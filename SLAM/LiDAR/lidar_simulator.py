import argparse
import os
import sys
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dataclasses import dataclass
from enum import Enum


# -*- coding: utf-8 -*-
"""Bundle Adjustment 핵심 — 회전 다양체, 투영, 해석적 야코비안.

표기
  T_i = (R_i, t_i)   카메라 i 의 world->camera 변환
  X_j                점 j 의 세계 좌표
  P   = R X + t      카메라 좌표계의 점
  pi(P)              픽셀 투영
갱신 규약 (SO(3) x R^3, 오른쪽 회전 섭동)
  R <- R exp(dphi^),  t <- t + drho
"""

def skew(v):

    v = np.asarray(v)

    if v.ndim == 1:
        x, y, z = v

        return np.array([
            [0, -z,  y],
            [z,  0, -x],
            [-y, x,  0]
        ])

    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]

    S = np.zeros((len(v), 3, 3))

    S[:, 0, 1] = -z
    S[:, 0, 2] =  y

    S[:, 1, 0] =  z
    S[:, 1, 2] = -x

    S[:, 2, 0] = -y
    S[:, 2, 1] =  x

    return S

def so3_exp(phi):
    """Rodrigues.  exp(phi^) in SO(3)."""
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.eye(3) + skew(phi)                 # 1차 근사로 충분
    K = skew(phi / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

def so3_log(R):
    """log: SO(3) -> R^3."""
    c = (np.trace(R) - 1) / 2
    th = np.arccos(np.clip(c, -1, 1))
    if th < 1e-12:
        return np.array([R[2, 1], R[0, 2], R[1, 0]])
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2],
                                             R[0, 2] - R[2, 0],
                                             R[1, 0] - R[0, 1]])

# ── 유사변환 정렬 ─────────────────────────────────────────
def umeyama_transform(src, dst):
    """src 를 dst 에 맞추는 최소제곱 유사변환. (scale, R, t) 를 돌려준다.

    BA 의 해는 게이지 자유도만큼 자유로우므로, 두 해를 비교하려면 먼저 이 변환으로
    같은 좌표계에 올려야 한다. 점과 카메라에 같은 변환을 걸 수 있도록 성분으로 준다.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U_, d_, Vt = np.linalg.svd(C)
    E = np.eye(3)
    if np.linalg.det(U_) * np.linalg.det(Vt) < 0: E[2, 2] = -1
    R_ = U_ @ E @ Vt
    var_s = (S ** 2).sum() / len(src)
    sc = float((d_ * np.diag(E)).sum() / var_s) if var_s > 0 else 1.0
    return sc, R_, (mu_d - sc * R_ @ mu_s)

def umeyama_align(src, dst):
    """유사변환(스케일+회전+평행이동)으로 src 를 dst 에 최소제곱 정렬. BA 비교의 전제."""
    sc, R_, t_ = umeyama_transform(src, dst)
    return (sc * (R_ @ src.T).T) + t_

# ── 투영 ─────────────────────────────────────────────────
class Camera:
    """핀홀 내부 파라미터."""
    def __init__(s, fx, fy, cx, cy):
        s.fx, s.fy, s.cx, s.cy = fx, fy, cx, cy

def project(cam, R, t, X):
    """pi(R X + t) -> 픽셀 (2,)"""
    P = R @ X + t
    return np.array([cam.fx * P[0] / P[2] + cam.cx,
                     cam.fy * P[1] / P[2] + cam.cy]), P

# ── 해석적 야코비안 ───────────────────────────────────────
def dpi_dP(cam, P):
    """d pi / d P   (2x3)"""
    x, y, z = P
    iz = 1.0 / z
    return np.array([[cam.fx * iz, 0, -cam.fx * x * iz * iz],
                     [0, cam.fy * iz, -cam.fy * y * iz * iz]])

def jacobians(cam, R, t, X):
    """(J_pose 2x6, J_point 2x3, residual 예측값 2)"""
    z_hat, P = project(cam, R, t, X)
    A = dpi_dP(cam, P)                    # 2x3
    # 오른쪽 회전 섭동:  P(d) = R exp(dphi^) X + t + drho  ~=  P - R X^ dphi + drho
    J_rho = A                             # dP/drho = I
    J_phi = -A @ R @ skew(X)              # dP/dphi = -R X^
    J_pose = np.hstack([J_rho, J_phi])    # 2x6, 순서 = [drho(3), dphi(3)]
    J_point = A @ R                       # dP/dX = R
    return J_pose, J_point, z_hat

# ── 수치 미분으로 검증 ────────────────────────────────────
def numeric_jacobians(cam, R, t, X, h=1e-7):
    Jp = np.zeros((2, 6)); Jx = np.zeros((2, 3))
    for k in range(3):                                     # drho
        e = np.zeros(3); e[k] = h
        zp, _ = project(cam, R, t + e, X)
        zm, _ = project(cam, R, t - e, X)
        Jp[:, k] = (zp - zm) / (2 * h)
    for k in range(3):                                     # dphi (오른쪽 섭동)
        e = np.zeros(3); e[k] = h
        zp, _ = project(cam, R @ so3_exp(e), t, X)
        zm, _ = project(cam, R @ so3_exp(-e), t, X)
        Jp[:, 3 + k] = (zp - zm) / (2 * h)
    for k in range(3):                                     # dX
        e = np.zeros(3); e[k] = h
        zp, _ = project(cam, R, t, X + e)
        zm, _ = project(cam, R, t, X - e)
        Jx[:, k] = (zp - zm) / (2 * h)
    return Jp, Jx


# ══════════════════════════════════════════════════════════════════
# 합성 장면 — 방(바닥 + 벽 넷) 안을 도는 LiDAR
# ══════════════════════════════════════════════════════════════════
ROOM_HALF = 2.5          # 방 반폭 [m]
ROOM_HEIGHT = 2.0        # 천장 높이 [m] (벽의 세로 범위)
# 이 거리 안의 표면만 본다 — 먼 구석은 빠져 스캔마다 겹침이 부분적이다. 단, 어느 포즈에서든
# 벽 넷이 다 보여야 한다 (가장 먼 벽까지 ROOM_HALF + 궤적 반경 1.2 = 3.7 m). 3.2 m 로 두면 벽 쪽
# 포즈가 바닥 + 평행한 벽 둘만 봐서 그 벽을 따라가는 평행이동이 관측되지 않았다 (README
# §10.3.3 의 복도 축퇴) — 실측: 잔차 RMSE 는 참값 수준(7.8 vs 7.6 mm)인데 포즈는 1.3 cm 틀렸다.
LIDAR_RANGE = 4.5

# (평면 위 한 점, 법선).  법선은 방 안쪽을 향한다.
ROOM_PLANES = [
    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),            # 바닥
    (np.array([-ROOM_HALF, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),     # 벽 x-
    (np.array([ROOM_HALF, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),     # 벽 x+
    (np.array([0.0, -ROOM_HALF, 0.0]), np.array([0.0, 1.0, 0.0])),     # 벽 y-
    (np.array([0.0, ROOM_HALF, 0.0]), np.array([0.0, -1.0, 0.0])),     # 벽 y+
]

# ══════════════════════════════════════════════════════════════════
# N 채널 회전형 LiDAR
# ══════════════════════════════════════════════════════════════════
@dataclass
class LidarSpec:
    """회전형 LiDAR 의 사양. 기본값은 Velodyne VLP-16 계열을 따랐다."""
    channels: int = 16                      # 수직 빔 개수
    azimuth_steps: int = 900                # 한 바퀴 분할 = 0.4 도 간격
    elevation_min_deg: float = -15.0
    elevation_max_deg: float = 15.0
    min_range: float = 0.3
    max_range: float = 30.0
    grazing_limit_deg: float = 85.0         # 입사각이 이보다 스치면 반사가 아예 안 돌아온다
    # 한 바퀴 도는 동안 센서가 실제로 지나는 거리 = 키프레임 간격의 몇 배인가.
    # 스캔 주기와 키프레임 간격은 전혀 다른 양이다. 10 Hz 센서가 0.5 m/s 로 움직이면 한
    # 바퀴에 0.05 m 를 지나는데, 이 궤적의 키프레임 간격은 0.75 m 라 약 0.07 이다.
    # 1.0 으로 두면 한 스캔이 키프레임 하나를 통째로 건너뛰어 왜곡이 1.8 m 까지 간다.
    sweep_fraction: float = 0.1
    range_noise: float = 0.1
    dropout: float = 0.1


class EObjectType(Enum):
    SPHERE = 0
    CUBE = 1
    PLANE = 2


class SceneObject:
    """장면에 놓이는 기본 도형 하나. 로컬 프레임에서 정의하고 (rotation, position) 으로 world 에 놓는다.

    size 의 뜻이 타입마다 다르다:
      SPHERE  radius              반지름
      CUBE    (hx, hy, hz)        반 변 길이. rotation 이 붙으면 기울어진 상자(OBB)다
      PLANE   (half_u, half_v)    로컬 +z 가 법선인 직사각형 판. 무한 평면이 아니라 유한한 판이다

    hit 은 광선을 하나가 아니라 묶음으로 받는다 — 회전형 LiDAR 한 바퀴가 채널 x 방위각이라
    광선이 한 번에 수만 개다. 광선마다 파이썬 루프를 돌면 스캔 하나에 수 초가 걸린다.
    """

    def __init__(self, type, position=(0.0, 0.0, 0.0), size=1.0, rotation=None, name=""):
        self.type = type
        self.position = np.asarray(position, dtype=float)
        self.size = np.atleast_1d(np.asarray(size, dtype=float))
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        self.name = name or type.name.lower()

    def half_extent(self):
        """CUBE 의 반 변 길이를 3 벡터로. 스칼라 하나만 준 경우를 흡수한다."""
        return self.size if len(self.size) == 3 else np.full(3, self.size[0])

    def hit(self, origin, directions):
        """광선 묶음과의 최근접 교차. origin (3,), directions 는 단위벡터 (k,3).

        돌려주는 것: (distance (k,), normal (k,3)).  맞지 않은 광선은 distance = inf 다.
        normal 은 world 단위벡터이고 항상 광선을 마주 보도록 부호를 맞춘다 — 판의 뒷면을
        때려도 입사각 판정이 뒤집히지 않게.
        """
        if self.type is EObjectType.PLANE:
            return self._hit_plane(origin, directions)
        if self.type is EObjectType.SPHERE:
            return self._hit_sphere(origin, directions)
        return self._hit_cube(origin, directions)

    def _hit_plane(self, origin, directions):
        # np.dot 이지 @ 가 아니다 — 이 파일 위쪽 build_normal_equation 과 같은 이유로,
        # macOS 의 numpy 2.x 는 (k,3) 행렬곱에 거짓 "divide by zero" 경고를 낸다.
        normal = self.rotation[:, 2]
        denominator = np.dot(directions, normal)
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = np.dot(self.position - origin, normal) / denominator
        # 광선과 평행하거나 뒤쪽이면 없는 것으로 친다
        distance = np.where((np.abs(denominator) > 1e-12) & (distance > 0), distance, np.inf)

        # 유한한 판이므로 로컬 (u, v) 가 범위 안인지 본다. inf 자리는 건드리지 않는다 —
        # inf * 0 = nan 이라 전부 계산하면 축에 평행한 광선이 조용히 nan 이 된다.
        inside = np.isfinite(distance)
        if inside.any():
            offset = origin + distance[inside, None] * directions[inside] - self.position
            u = np.dot(offset, self.rotation[:, 0])
            v = np.dot(offset, self.rotation[:, 1])
            inside[inside] = (np.abs(u) <= self.size[0]) & (np.abs(v) <= self.size[1])
        distance = np.where(inside, distance, np.inf)

        facing = np.where(denominator > 0, -1.0, 1.0)
        return distance, normal * facing[:, None]

    def _hit_sphere(self, origin, directions):
        radius = float(self.size[0])
        to_origin = origin - self.position
        b = np.dot(directions, to_origin)
        c = to_origin @ to_origin - radius * radius
        discriminant = b * b - c
        root = np.sqrt(np.maximum(discriminant, 0.0))
        near, far = -b - root, -b + root
        # 센서가 구 안에 있으면 near 가 음수다 — 그때는 뒷면(far)이 보이는 면이다
        distance = np.where(near > 1e-9, near, far)
        distance = np.where((discriminant > 0.0) & (distance > 1e-9), distance, np.inf)

        normal = np.zeros_like(directions)
        valid = np.isfinite(distance)
        if valid.any():
            outward = (origin + distance[valid, None] * directions[valid] - self.position) / radius
            facing = np.where(np.einsum("ij,ij->i", directions[valid], outward) > 0, -1.0, 1.0)
            normal[valid] = outward * facing[:, None]
        return distance, normal

    def _hit_cube(self, origin, directions):
        """슬랩 방법. 로컬 프레임으로 옮기면 축 정렬 상자라 축마다 구간 교집합이면 된다."""
        half = self.half_extent()
        origin_local = np.dot(self.rotation.T, origin - self.position)
        directions_local = np.dot(directions, self.rotation)
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = 1.0 / directions_local
            t_low = (-half - origin_local) * inverse
            t_high = (half - origin_local) * inverse
        t_near = np.minimum(t_low, t_high)
        t_far = np.maximum(t_low, t_high)

        # 축에 평행한 성분은 0 으로 나눈 값이라 못 쓴다. 그 축의 슬랩 안이면 제약이 없고,
        # 밖이면 어떤 t 에서도 못 맞는다.
        parallel = np.abs(directions_local) < 1e-12
        inside_slab = np.abs(origin_local) <= half
        t_near = np.where(parallel, np.where(inside_slab, -np.inf, np.inf), t_near)
        t_far = np.where(parallel, np.where(inside_slab, np.inf, -np.inf), t_far)

        enter, leave = t_near.max(1), t_far.min(1)
        distance = np.where(enter > 1e-9, enter, leave)      # 상자 안이면 나가는 면이 보인다
        distance = np.where((leave >= np.maximum(enter, 0.0)) & (distance > 1e-9), distance, np.inf)

        normal = np.zeros_like(directions)
        valid = np.isfinite(distance)
        if valid.any():
            point_local = origin_local + distance[valid, None] * directions_local[valid]
            # 어느 면인가 = 반 변 길이로 정규화했을 때 1 에 가장 가까운 축
            axis = np.argmax(np.abs(point_local) / half, axis=1)
            rows = np.arange(len(axis))
            local_normal = np.zeros((len(axis), 3))
            local_normal[rows, axis] = np.sign(point_local[rows, axis])
            world_normal = np.dot(local_normal, self.rotation.T)
            facing = np.where(np.einsum("ij,ij->i", directions[valid], world_normal) > 0, -1.0, 1.0)
            normal[valid] = world_normal * facing[:, None]
        return distance, normal


def surface_residual(obj, points):
    local = np.dot(points - obj.position, obj.rotation)
    if obj.type is EObjectType.SPHERE:
        return np.abs(np.linalg.norm(points - obj.position, axis=1) - obj.size[0])
    if obj.type is EObjectType.PLANE:
        return np.abs(local[:, 2])
    half = obj.half_extent()
    return np.abs((np.abs(local) / half).max(1) - 1.0) * half.min()


def rotation_from_normal(normal, up=(0.0, 0.0, 1.0)):
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    up = np.asarray(up, dtype=float)
    if abs(up @ normal) > 0.9:                       # 바닥·천장이면 up 을 바꿔 잡는다
        up = np.array([0.0, 1.0, 0.0])
    u = np.cross(up, normal)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(normal, u), normal])

def make_room_objects(props=True):
    half, height = ROOM_HALF, ROOM_HEIGHT
    objects = [
        SceneObject(EObjectType.PLANE, (0, 0, 0), (half, half),
                    rotation_from_normal((0, 0, 1)), "floor"),
        SceneObject(EObjectType.PLANE, (0, 0, height), (half, half),
                    rotation_from_normal((0, 0, -1)), "ceiling"),
        SceneObject(EObjectType.PLANE, (-half, 0, height / 2), (half, height / 2),
                    rotation_from_normal((1, 0, 0)), "wall x-"),
        SceneObject(EObjectType.PLANE, (half, 0, height / 2), (half, height / 2),
                    rotation_from_normal((-1, 0, 0)), "wall x+"),
        SceneObject(EObjectType.PLANE, (0, -half, height / 2), (half, height / 2),
                    rotation_from_normal((0, 1, 0)), "wall y-"),
        SceneObject(EObjectType.PLANE, (0, half, height / 2), (half, height / 2),
                    rotation_from_normal((0, -1, 0)), "wall y+"),
    ]
    if props:
        objects += [
            SceneObject(EObjectType.SPHERE, (1.45, -1.15, 0.45), 0.45, None, "sphere"),
            SceneObject(EObjectType.CUBE, (-1.55, 1.30, 0.35), (0.35, 0.35, 0.35),
                        so3_exp(np.array([0.0, 0.0, 0.6])), "cube"),
            SceneObject(EObjectType.CUBE, (0.25, 1.85, 0.60), (0.55, 0.12, 0.60),
                        so3_exp(np.array([0.0, 0.0, -0.25])), "panel"),
        ]
    return objects


def interpolate_pose(R_from, t_from, R_to, t_to, fraction, base=None):
    """(R_from, t_from) 에서 (R_to, t_to) 로 fraction 만큼 간 포즈.

    회전은 SO(3) 안에서 간다 — 행렬을 선형보간하면 군을 벗어난다. base 를 주면 그 포즈에
    같은 크기의 운동을 얹는다 (마지막 키프레임처럼 다음 구간이 없을 때 직전 운동을 잇는 용도).
    """
    turn = fraction * so3_log(R_from.T @ R_to)
    shift = fraction * (t_to - t_from)
    R_base, t_base = (R_from, t_from) if base is None else base
    return R_base @ so3_exp(turn), t_base + shift

@dataclass
class LidarScan:
    """한 바퀴 스캔의 원시 반환.

    평면 라벨이 없다. 실제 센서가 주는 것은 (거리, 채널 고도, 방위각) 뿐이고 points 는 그
    구면좌표를 직교로 푼 것이다 — 포즈가 들어가지 않는다.
    object_index 와 normal 은 참값이라 채점·그림용이지 풀이에 쓰라고 있는 게 아니다.
    """
    pose_index: int
    points: np.ndarray                      # (k,3) 센서 프레임
    ranges: np.ndarray                      # (k,)
    ring: np.ndarray                        # (k,) 채널 번호
    azimuth_index: np.ndarray               # (k,) 방위각 칸 번호
    azimuth: np.ndarray                     # (k,) [rad]
    object_index: np.ndarray                # (k,) 참값
    normal: np.ndarray                      # (k,3) 참값 world 법선
    world_truth: np.ndarray                 # (k,3) 발사 시점 포즈로 푼 참 위치

def beam_directions(spec):
    """센서 프레임 빔 방향 (channels, azimuth_steps, 3) 과 고도·방위각.

    고도 e, 방위각 a 에 대해 d = (cos e cos a, cos e sin a, sin e).
    """
    elevation = np.radians(np.linspace(spec.elevation_min_deg, spec.elevation_max_deg, spec.channels))
    azimuth = np.linspace(0.0, 2.0 * np.pi, spec.azimuth_steps, endpoint=False)
    cos_elevation, sin_elevation = np.cos(elevation)[:, None], np.sin(elevation)[:, None]
    directions = np.stack([cos_elevation * np.cos(azimuth)[None, :],
                           cos_elevation * np.sin(azimuth)[None, :],
                           np.broadcast_to(sin_elevation, (spec.channels, spec.azimuth_steps))],
                          axis=-1)
    return directions, elevation, azimuth

def firing_segments(R, t, n_azimuth, motion, chunks=36):
    """방위각 구간별 (열 인덱스, 발사 포즈).

    motion 이 없으면 한 덩어리 = 강체 스캔. 있으면 다음 포즈까지 방위각에 비례해 보간한다 —
    회전형 LiDAR 는 한 바퀴 도는 동안(보통 100 ms) 계속 움직이므로 한 스캔의 점들이 서로
    다른 포즈에서 측정된 것이다 (README §10.4). 구간으로 끊는 이유는 속도다: 방위각마다
    포즈를 새로 만들면 스캔 하나에 numpy 호출이 수천 번이다.
    """
    if motion is None:
        return [(np.arange(n_azimuth), R, t)]
    R_next, t_next = motion
    turn = so3_log(R.T @ R_next)
    bounds = np.unique(np.linspace(0, n_azimuth, min(chunks, n_azimuth) + 1).astype(int))
    segments = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        fraction = 0.5 * (low + high) / n_azimuth
        segments.append((np.arange(low, high),
                         R @ so3_exp(fraction * turn), t + fraction * (t_next - t)))
    return segments


def cast_rays(objects, origin, directions):
    """광선마다 가장 가까운 물체를 고른다. (distance, normal, object_index) — 못 맞으면 -1."""
    distance = np.full(len(directions), np.inf)
    normal = np.zeros_like(directions)
    object_index = np.full(len(directions), -1, dtype=int)
    for index, obj in enumerate(objects):
        hit_distance, hit_normal = obj.hit(origin, directions)
        closer = hit_distance < distance
        distance = np.where(closer, hit_distance, distance)
        normal = np.where(closer[:, None], hit_normal, normal)
        object_index = np.where(closer, index, object_index)
    return distance, normal, object_index


def scan_once(objects, R, t, spec, rng, pose_index=0, motion=None):
    """포즈 (R, t) 에서 한 바퀴 돌며 광선을 쏜다."""
    directions, _, azimuth = beam_directions(spec)
    shape = (spec.channels, spec.azimuth_steps)

    distance = np.full(shape, np.inf)
    normal = np.zeros(shape + (3,))
    object_index = np.full(shape, -1, dtype=int)
    world_direction = np.zeros(shape + (3,))
    world_origin = np.zeros(shape + (3,))

    for columns, R_fire, t_fire in firing_segments(R, t, spec.azimuth_steps, motion):
        rays = np.dot(directions[:, columns, :].reshape(-1, 3), R_fire.T)   # 센서 -> world
        hit_distance, hit_normal, hit_object = cast_rays(objects, t_fire, rays)
        block = (spec.channels, len(columns))
        distance[:, columns] = hit_distance.reshape(block)
        normal[:, columns, :] = hit_normal.reshape(block + (3,))
        object_index[:, columns] = hit_object.reshape(block)
        world_direction[:, columns, :] = rays.reshape(block + (3,))
        world_origin[:, columns, :] = t_fire

    # 1. 사거리 밖은 반환이 없다
    returned = (distance > spec.min_range) & (distance < spec.max_range)
    # 2. 스치는 입사각은 빛이 안 돌아온다 — 벽을 비스듬히 보는 채널이 비는 이유
    cosine = np.abs(np.einsum("caj,caj->ca", world_direction, normal))
    returned &= cosine > np.cos(np.radians(spec.grazing_limit_deg))
    # 3. 무작위 미수신
    if spec.dropout > 0.0:
        returned &= rng.random(shape) >= spec.dropout

    measured = distance + rng.normal(scale=spec.range_noise, size=shape)
    ring, column = np.nonzero(returned)
    ranges = measured[ring, column]

    return LidarScan(
        pose_index=pose_index,
        points=ranges[:, None] * directions[ring, column],
        ranges=ranges,
        ring=ring,
        azimuth_index=column,
        azimuth=azimuth[column],
        object_index=object_index[ring, column],
        normal=normal[ring, column],
        world_truth=world_origin[ring, column] + ranges[:, None] * world_direction[ring, column])


def circular_trajectory(n_pose, radius=1.2):
    Rs, ts = [], []
    for i in range(n_pose):
        angle = 2.0 * np.pi * i / n_pose
        ts.append(np.array([radius * np.cos(angle), radius * np.sin(angle),
                            0.8 + 0.2 * np.sin(2 * angle)]))
        Rs.append(so3_exp(np.array([0.0, 0.0, angle + 0.5 * np.pi]))
                  @ so3_exp(np.array([0.05 * np.sin(angle), 0.0, 0.0])))
    return np.array(Rs), np.array(ts)

def make_lidar_data(objects, n_pose=10, spec=None, seed=0, motion=False):
    spec = LidarSpec() if spec is None else spec
    rng = np.random.default_rng(seed)
    Rs, ts = circular_trajectory(n_pose)
    scans = []
    for i in range(n_pose):
        step = None
        if motion:
            following = min(i + 1, n_pose - 1)      # 마지막 포즈는 직전 구간의 운동을 이어 쓴다
            leading = following - 1 if following == i else i
            step = interpolate_pose(Rs[leading], ts[leading], Rs[following], ts[following],
                                    spec.sweep_fraction, base=(Rs[i], ts[i]))
        scans.append(scan_once(objects, Rs[i], ts[i], spec, rng, pose_index=i, motion=step))
    return Rs, ts, scans
