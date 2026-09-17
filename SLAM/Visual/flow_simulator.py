"""Optical Flow 실습용 합성 데이터 생성기.

노트북은 알고리즘만 담고, 데이터를 만드는 잡일은 여기로 뺀다.
핵심은 **참값 flow 를 알고 있는 이미지 쌍**을 만드는 것이다. 참값이 없으면
"잔차가 줄었다"까지만 말할 수 있고 "맞게 수렴했다"는 말할 수 없다.

좌표 규약
---------
flow (u, v) 는 I0 의 픽셀 (x, y) 가 I1 에서 (x+u, y+v) 로 갔다는 뜻이다. 따라서
    I1(x + u, y + v) = I0(x, y)
이고, 이 관계 하나로 합성과 정렬의 방향이 모두 결정된다.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter

__all__ = [
    "FlowSpec", "ImagePair",
    "pixel_grid", "remap", "make_texture", "make_flow_field", "make_flow_data",
    "image_gradients", "window_sum", "warp_to_source", "endpoint_error",
    "downsample", "build_pyramid", "flow_to_rgb",
]


# ---------------------------------------------------------------- 기본 유틸

def pixel_grid(shape):
    """(H, W) 에 대한 정수 픽셀 좌표 (x, y). 둘 다 (H, W) 이다."""
    height, width = shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    return x.astype(float), y.astype(float)


def remap(image, dx, dy, order=1):
    """image 를 (x + dx, y + dy) 위치에서 샘플링한다.

    map_coordinates 는 (row, col) = (y, x) 순서를 받는다. 이 축 순서를 뒤집는 것이
    optical flow 구현에서 가장 흔한 버그라 한 곳에 가둬 둔다.
    """
    x, y = pixel_grid(image.shape)
    return map_coordinates(image, [y + dy, x + dx], order=order, mode="nearest")


# ---------------------------------------------------------------- 데이터 생성

@dataclass
class FlowSpec:
    """합성 이미지 쌍의 설정."""
    height: int = 160
    width: int = 200
    texture_scale: float = 2.0      # 클수록 부드러운 무늬 → gradient 가 약해진다
    noise: float = 0.0              # I1 에만 더하는 밝기 노이즈 (brightness constancy 위반)
    kind: str = "translation"       # translation | rotation | divergence
    amount: float = 3.0             # translation 이면 [px], 그 외에는 세기


@dataclass
class ImagePair:
    """참값을 아는 이미지 쌍."""
    I0: np.ndarray                  # (H, W) float, [0, 1]
    I1: np.ndarray                  # (H, W)
    flow: np.ndarray                # (H, W, 2) 참값 (u, v)
    spec: FlowSpec


def make_texture(shape, scale=2.0, seed=0):
    """모든 픽셀에 gradient 가 살아 있는 무늬를 만든다.

    LK 는 gradient 가 0 인 곳에서 아무것도 못 한다. 평평한 배경에 도형 몇 개를 그린
    그림을 쓰면 대부분의 픽셀에서 H 가 특이해져, 알고리즘이 아니라 데이터 때문에
    실패한다. 그래서 백색 잡음을 살짝 흐려 어디에나 무늬가 있게 만든다.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape)
    texture = gaussian_filter(noise, sigma=scale)
    texture += 0.4 * gaussian_filter(noise, sigma=scale * 4)     # 저주파를 섞어 대비를 준다
    texture -= texture.min()
    return texture / (texture.max() + 1e-12)


def make_flow_field(shape, kind="translation", amount=3.0):
    """참값 flow (H, W, 2) 를 만든다."""
    height, width = shape
    x, y = pixel_grid(shape)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0

    if kind == "translation":
        u = np.full(shape, amount)
        v = np.full(shape, amount * 0.5)
    elif kind == "rotation":
        angle = np.deg2rad(amount)                # amount 를 회전각[deg] 로 읽는다
        u = -(y - cy) * angle
        v = (x - cx) * angle
    elif kind == "divergence":
        scale = amount / max(cx, cy)
        u = (x - cx) * scale
        v = (y - cy) * scale
    else:
        raise ValueError(f"모르는 flow 종류: {kind}")

    return np.stack([u, v], axis=-1)


def make_flow_data(spec=None, seed=0):
    """(I0, I1, 참값 flow) 를 만든다.

    I1(x+u, y+v) = I0(x, y) 를 만족시켜야 하므로, I1 은 I0 를 **-flow** 로 샘플링해
    만든다. 부호를 뒤집으면 알고리즘이 정확히 반대 방향으로 수렴하는데, 그림만 봐서는
    잘 안 보인다.
    """
    spec = FlowSpec() if spec is None else spec
    shape = (spec.height, spec.width)

    I0 = make_texture(shape, scale=spec.texture_scale, seed=seed)
    flow = make_flow_field(shape, kind=spec.kind, amount=spec.amount)
    I1 = remap(I0, -flow[..., 0], -flow[..., 1])

    if spec.noise > 0:
        rng = np.random.default_rng(seed + 1)
        I1 = I1 + rng.normal(scale=spec.noise, size=shape)

    return ImagePair(I0=I0, I1=I1, flow=flow, spec=spec)


# ---------------------------------------------------------------- LK 보조

def image_gradients(image):
    """중앙차분으로 (I_x, I_y). 가장자리는 한쪽 차분으로 떨어진다."""
    I_y, I_x = np.gradient(image)       # np.gradient 는 축 순서대로 (row, col) 을 준다
    return I_x, I_y


def window_sum(array, size):
    """size x size 창 합. uniform_filter 는 평균이라 size^2 를 곱해 합으로 되돌린다.

    H 와 g 에 같은 상수가 곱해지므로 H δ = -g 의 해는 변하지 않는다. 그래도 합으로
    두는 편이 수식과 1:1 로 읽힌다.
    """
    return uniform_filter(array, size=size, mode="nearest") * (size * size)


def warp_to_source(I1, flow):
    """현재 flow 로 I1 을 I0 쪽으로 끌어온다. 수렴하면 I0 와 같아진다."""
    return remap(I1, flow[..., 0], flow[..., 1])


def endpoint_error(flow, flow_gt, margin=8):
    """EPE = 픽셀당 flow 벡터 오차의 평균 [px].

    가장자리는 warp 가 밖을 참조해 값이 오염되므로 margin 만큼 잘라내고 잰다.
    """
    d = flow[margin:-margin, margin:-margin] - flow_gt[margin:-margin, margin:-margin]
    return float(np.mean(np.linalg.norm(d, axis=-1)))


# ---------------------------------------------------------------- 피라미드

def downsample(image):
    """blur 후 2배 축소. blur 없이 버리면 aliasing 이 gradient 를 망가뜨린다."""
    return gaussian_filter(image, sigma=1.0)[::2, ::2]


def build_pyramid(image, levels):
    """[0] 이 원본, 뒤로 갈수록 거친 레벨."""
    pyramid = [image]
    for _ in range(levels - 1):
        pyramid.append(downsample(pyramid[-1]))
    return pyramid


# ---------------------------------------------------------------- 시각화

def flow_to_rgb(flow, max_magnitude=None):
    """flow 를 색으로. 색상(hue)=방향, 채도=크기. 두 flow 를 비교하려면
    max_magnitude 를 같은 값으로 고정해야 한다."""
    from matplotlib.colors import hsv_to_rgb

    u, v = flow[..., 0], flow[..., 1]
    magnitude = np.hypot(u, v)
    if max_magnitude is None:
        max_magnitude = magnitude.max()

    hsv = np.zeros(flow.shape[:2] + (3,))
    hsv[..., 0] = (np.arctan2(v, u) / (2 * np.pi)) % 1.0
    hsv[..., 1] = np.clip(magnitude / (max_magnitude + 1e-12), 0, 1)
    hsv[..., 2] = 1.0
    return hsv_to_rgb(hsv)
