# -*- coding: utf-8 -*-
"""LiDAR BA 를 눈으로 본다 — point-to-plane 잔차로 포즈만 푸는 BA 의 수렴 애니메이션.

카메라 BA(`ba_viz.py`)와 달리 "같은 점"이 없다 (README §10.3). 그래서 점을 변수로 두지
않고, 각 점이 속한 **평면**에 대한 거리를 잔차로 쓴다:

  P_world = R_i P_lidar + t_i                       (T_i 는 LiDAR->World)
  r       = n^T P_world + d                         평면 (n, d) 까지의 부호 있는 거리
  J_T     = [ -n^T [P_world]_x   n^T ]              d r / d[dtheta, dt]   (1x6)

섭동 규약은 문서 그대로 **왼쪽(world) 섭동**이다:
  P_world(d) = exp(dtheta^) (R P + t) + dt  ~=  P_world - [P_world]_x dtheta + dt
  갱신:  R <- exp(dtheta^) R,   t <- exp(dtheta^) t + dt
★ `ba_core.py` 는 오른쪽 섭동(R exp(dphi^))이다. 두 규약을 섞으면 야코비안의 회전 블록이
  틀린다 — `--selftest` 가 중앙차분으로 이 블록을 직접 대조한다.

랜드마크는 점이 아니라 **평면 (n, d)** 다 — 문서 §1.3 의 X_j 자리에 평면이 앉는다. 초기값은
모든 스캔의 점을 초기 포즈로 world 에 올려 산란행렬의 최소고유벡터로 닫힌형 맞춤(README
§10.3.2)하고, 그 뒤로는 포즈와 **한 시스템에서** LM 으로 푼다 (평면 블록 J = [P^T B, 1],
B 는 법선의 접공간 기저). 포즈 0 을 게이지로 고정한다 — 평면은 6 자유도를 다 관측하므로
카메라 BA 와 달리 스케일은 자유롭지 않다.

뷰가 둘이고 **데이터 생성기가 둘**이다. 이 차이가 이 파일의 요점이다:

  make_lidar_scene   평면 위에 점을 직접 뿌린다. 각 점에 참값 평면 번호를 붙여 준다.
                     converge 뷰와 BA 가 이걸 쓴다 — 데이터 연관이 이미 풀린 상태다.
  make_lidar_data    장면에 SceneObject 를 놓고 N 채널 회전형 LiDAR 광선을 쏜다.
                     나오는 것은 (거리, 빔 각도) 뿐 — 평면 번호가 없다. raw 뷰가 이걸 쓴다.

★ 두 번째가 실데이터의 형태다. 첫 번째로 BA 를 돌리면 쉬워 보이는 이유가 여기 있다 —
  실측: 초기 오차 175 mm 에서 참값 라벨은 1.5 mm 로 수렴하지만, 같은 장면에서 BALM 식
  복셀 연관은 평면을 2개밖에 못 찾아 한 발짝도 못 움직인다. 연관이 쓸 만해지는 것은 초기
  오차 100 mm 아래부터다. 두 번째 데이터를 BA 에 넣으려면 연관 단계를 먼저 붙여야 한다.

실행:
  python ba_lidar.py                    # 수렴 뷰 — 슬라이더 + 재생
  python ba_lidar.py raw                # 회전형 LiDAR 원시 점 — 포즈 슬라이더 + 재생
  python ba_lidar.py raw --motion       # 스캔 내 운동 왜곡을 켠다
  python ba_lidar.py raw --channels 64 --azimuth-steps 1800
  python ba_lidar.py --save figures     # 창 없이 PNG 저장 (헤드리스)
  python ba_lidar.py --selftest         # 야코비안·수렴·레이캐스팅·콜백 확인
"""
import argparse
import os
import sys
from dataclasses import dataclass
from enum import Enum

import numpy as np

from scipy.spatial import cKDTree

from ba_core import so3_exp, so3_log
from ba_solver import huber_weight


np.set_printoptions(precision=3, suppress=True)


# ── 인자를 먼저 읽는다: --save 면 창 없는 백엔드를 pyplot import 전에 골라야 한다 ──
def parse_args(argv):
    p = argparse.ArgumentParser(description="LiDAR Bundle Adjustment 시각화")
    p.add_argument("view", nargs="?", default="converge", choices=["converge", "raw"],
                   help="converge = BA 수렴 애니메이션(기본), raw = 회전형 LiDAR 원시 점")
    p.add_argument("--save", metavar="DIR", default=None,
                   help="창을 띄우지 않고 이 폴더에 PNG 를 저장한다")
    p.add_argument("--seed", type=int, default=0, help="장면 시드")
    p.add_argument("--n-pose", type=int, default=10, help="LiDAR 포즈 수")
    p.add_argument("--n-pt", type=int, default=120, help="스캔당 평면 하나에서 찍는 점 수")
    p.add_argument("--iters", type=int, default=20, help="LM 최대 반복")
    p.add_argument("--noise", type=float, default=0.01, help="LiDAR 거리 잡음 [m]")
    p.add_argument("--rot", type=float, default=0.05, help="초기 회전 오차 표준편차 [rad]")
    p.add_argument("--trans", type=float, default=0.15, help="초기 평행이동 오차 표준편차 [m]")
    p.add_argument("--huber", type=float, default=None,
                   help="Huber 문턱 [m] (기본 없음 = 최소제곱)")
    p.add_argument("--selftest", action="store_true",
                   help="그리지 않고 야코비안·수렴·레이캐스팅·콜백 일관성만 확인한다")
    # ── raw 뷰 전용 — N 채널 회전형 LiDAR 사양 ──
    p.add_argument("--channels", type=int, default=8, help="raw: 수직 채널 수")
    p.add_argument("--azimuth-steps", type=int, default=450, help="raw: 한 바퀴의 방위각 분할")
    p.add_argument("--elevation", type=float, nargs=2, default=(-15.0, 15.0),
                   metavar=("MIN", "MAX"), help="raw: 수직 시야각 범위 [deg]")
    p.add_argument("--max-range", type=float, default=30.0, help="raw: 최대 사거리 [m]")
    p.add_argument("--dropout", type=float, default=0.0, help="raw: 무작위 미수신 비율")
    p.add_argument("--motion", action="store_true",
                   help="raw: 한 바퀴 도는 동안 센서가 움직인다 (스캔 내 운동 왜곡)")
    p.add_argument("--no-props", action="store_true",
                   help="raw: 방만 두고 구·정육면체 소품을 빼다")
    return p.parse_args(argv)


ARGS = parse_args(sys.argv[1:])

import matplotlib
if ARGS.save is not None or ARGS.selftest:
    matplotlib.use("Agg")           # 창 없는 렌더러. pyplot import 보다 먼저여야 한다
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

COLOR_TRUE = "#9aa0a6"
COLOR_POSE = "#1a73e8"
COLOR_ERR = "#d93025"
PLANE_COLORS = ["#ff7043", "#1e8e3e", "#8e24aa", "#fbbc04", "#00acc1", "#6d4c41"]


def use_korean_font():
    """한글 폰트가 있으면 쓴다. 없으면 라벨이 네모로 나오므로 영문으로 떨어진다."""
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic",
                 "Malgun Gothic", "Noto Sans CJK KR", "Arial Unicode MS"):
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["mathtext.fontset"] = "dejavusans"
            return True
    return False


KOREAN = use_korean_font()


def L(korean, english):
    """한글 폰트가 없는 환경에서도 읽히도록."""
    return korean if KOREAN else english


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


def plane_basis(normal):
    """법선에 수직인 정규직교 기저 둘 (2x3)."""
    return np.linalg.svd(normal.reshape(1, 3))[2][1:]


def sample_plane_patch(rng, point_on_plane, normal, count):
    """방 크기 안에서 평면 위 점을 균일하게 찍는다 (world 좌표)."""
    basis = plane_basis(normal)
    # 바닥은 x·y 가 방 안, 벽은 수평 방향이 방 안 + 높이가 [0, ROOM_HEIGHT]
    if abs(normal[2]) > 0.5:
        uv = rng.uniform(-ROOM_HALF, ROOM_HALF, size=(count, 2))
    else:
        uv = np.column_stack([rng.uniform(-ROOM_HALF, ROOM_HALF, size=count),
                              rng.uniform(0.0, ROOM_HEIGHT, size=count)])
        # basis 의 어느 행이 수직축인지 모르므로 world 로 옮긴 뒤 높이를 다시 정한다
        points = point_on_plane + uv[:, :1] * basis[np.argmin(np.abs(basis[:, 2]))]
        points[:, 2] = uv[:, 1]
        return points
    return point_on_plane + uv @ basis


def circular_trajectory(n_pose, radius=1.2):
    Rs, ts = [], []
    for i in range(n_pose):
        angle = 2.0 * np.pi * i / n_pose
        ts.append(np.array([radius * np.cos(angle), radius * np.sin(angle),
                            0.8 + 0.2 * np.sin(2 * angle)]))
        Rs.append(so3_exp(np.array([0.0, 0.0, angle + 0.5 * np.pi]))
                  @ so3_exp(np.array([0.05 * np.sin(angle), 0.0, 0.0])))
    return np.array(Rs), np.array(ts)


def make_lidar_scene(n_pose=10, n_pt=120, noise=0.01, seed=0):
    """합성 LiDAR 포즈 / 평면 / 스캔을 생성한다.
    
    돌려주는 것
      Rs, ts     참값 포즈 (LiDAR->World)
      planes     [(n, d)]  참값 평면,  n^T p + d = 0
      scans      [(pose_idx, plane_idx, P_lidar (k,3))]   포즈별·평면별 점 묶음
    
    점은 각 포즈에서 LIDAR_RANGE 안의 표면에서만 찍는다 — 실제 스캔처럼 겹침이 부분적이다.
    잡음은 빔 방향(센서에서 점으로 향하는 방향)에 준다: LiDAR 는 거리를 재기 때문이다.
    """
    rng = np.random.default_rng(seed)

    Rs, ts = circular_trajectory(n_pose)

    planes = [(normal / np.linalg.norm(normal), float(-normal @ point)) for point, normal in ROOM_PLANES]
    
    scans = []
    for i in range(n_pose):
        for j, (point_on_plane, normal) in enumerate(ROOM_PLANES):
            world = sample_plane_patch(rng, point_on_plane, normal, n_pt * 3)
            ray = world - ts[i]
            distance = np.linalg.norm(ray, axis=1)
            visible = distance < LIDAR_RANGE
            world, ray, distance = world[visible][:n_pt], ray[visible][:n_pt], distance[visible][:n_pt]
            if len(world) < 8:
                continue
            world = world + (ray / distance[:, None]) * rng.normal(scale=noise, size=(len(world), 1))
            local = (world - ts[i]) @ Rs[i]
            scans.append((i, j, local))

    # 축퇴는 조용히 틀린 해로 수렴한다 — 포즈마다 보이는 법선이 3차원을 펼치는지 여기서 막는다.
    for i in range(n_pose):
        normals = np.array([planes[j][0] for (pose, j, _) in scans if pose == i])
        if len(normals) == 0 or np.linalg.matrix_rank(normals, tol=1e-6) < 3:
            raise RuntimeError(f"make_lidar_scene: pose {i} sees planes whose normals do not span 3D "
                               f"- its translation is unobservable (raise LIDAR_RANGE)")
    return Rs, ts, planes, scans


def perturb_poses(Rs, ts, seed=1, rot=0.05, trans=0.15):
    """참값을 흔들어 초기 추정을 만든다. 포즈 0 은 게이지라 그대로 둔다."""
    rng = np.random.default_rng(seed)
    Rs_perturbed = Rs.copy()
    ts_perturbed = ts.copy()
    for i in range(1, len(Rs)):
        Rs_perturbed[i] = so3_exp(rng.normal(scale=rot, size=3)) @ Rs[i]
        ts_perturbed[i] = ts[i] + rng.normal(scale=trans, size=3)
    return Rs_perturbed, ts_perturbed


# ══════════════════════════════════════════════════════════════════
# 잔차 · 야코비안 — 문서 §1.2 / §1.3 그대로
# ══════════════════════════════════════════════════════════════════
def to_world(R, t, P_lidar):
    """P_world = R P_lidar + t   ((k,3) 벡터화)

    np.dot 이지 @ 가 아니다 — macOS 의 numpy 2.x 는 큰 (k,3) 행렬곱에 거짓 "divide by zero"
    경고를 낸다 (값은 유한). 회전형 LiDAR 스캔은 점이 수만 개라 이 경로로 들어온다.
    """
    return np.dot(P_lidar, R.T) + t


def residual_point_to_plane(R, t, P_lidar, normal, offset):
    """r = n^T (R P + t) + d   (k,)"""
    return to_world(R, t, P_lidar) @ normal + offset


def jacobian_pose(P_world, normal):
    """J_T = [ -n^T [P_world]_x ,  n^T ]   (k x 6),  열 순서 = [dtheta(3), dt(3)]

    -n^T [P]_x  는  (P x n)^T 과 같다 — 점마다 3x3 을 만들지 않고 외적 한 번으로 얻는다.
    """
    return np.hstack([np.cross(P_world, normal), np.broadcast_to(normal, P_world.shape)])


def apply_left_update(R, t, delta):
    """R <- exp(dtheta^) R,   t <- exp(dtheta^) t + dt      (delta = [dtheta, dt])"""
    E = so3_exp(delta[:3])
    return E @ R, E @ t + delta[3:]


def numeric_jacobian_pose(R, t, P_lidar, normal, offset, h=1e-7):
    """왼쪽 섭동 규약으로 중앙차분. (k x 6)"""
    J = np.zeros((len(P_lidar), 6))
    for k in range(6):
        e = np.zeros(6); e[k] = h
        Rp, tp = apply_left_update(R, t, e)
        Rm, tm = apply_left_update(R, t, -e)
        J[:, k] = (residual_point_to_plane(Rp, tp, P_lidar, normal, offset)
                   - residual_point_to_plane(Rm, tm, P_lidar, normal, offset)) / (2 * h)
    return J


# ══════════════════════════════════════════════════════════════════
# 평면 — 초기값은 닫힌형(README §10.3.2), 그 뒤로는 포즈와 함께 변수
# ══════════════════════════════════════════════════════════════════
def fit_planes(Rs, ts, scans, n_plane):
    gathered = [[] for _ in range(n_plane)]
    for i, j, local in scans:
        gathered[j].append(to_world(Rs[i], ts[i], local))
    planes = []
    for j in range(n_plane):
        points = np.vstack(gathered[j])
        centre = points.mean(0)
        centred = points - centre
        scatter = centred.T @ centred / len(points)
        normal = np.linalg.eigh(scatter)[1][:, 0]
        planes.append((normal, float(-normal @ centre)))
    return planes


def normal_tangent_basis(normal):
    """||n|| = 1 을 지키는 2 차원 섭동 기저 B (3x2):  n <- normalize(n + B dn)."""
    return plane_basis(normal).T


def jacobian_plane(P_world, normal):
    """d r / d[dn(2), dd(1)] = [ P_world^T B ,  1 ]   (k x 3)"""
    return np.hstack([P_world @ normal_tangent_basis(normal), np.ones((len(P_world), 1))])


def apply_plane_update(normal, offset, delta):
    """n <- normalize(n + B dn),   d <- d + dd      (delta = [dn(2), dd])"""
    normal = normal + normal_tangent_basis(normal) @ delta[:2]
    return normal / np.linalg.norm(normal), offset + float(delta[2])


def numeric_jacobian_plane(R, t, P_lidar, normal, offset, h=1e-7):
    J = np.zeros((len(P_lidar), 3))
    for k in range(3):
        e = np.zeros(3); e[k] = h
        np_, dp = apply_plane_update(normal, offset, e)
        nm, dm = apply_plane_update(normal, offset, -e)
        J[:, k] = (residual_point_to_plane(R, t, P_lidar, np_, dp)
                   - residual_point_to_plane(R, t, P_lidar, nm, dm)) / (2 * h)
    return J


def all_residuals(Rs, ts, planes, scans):
    """모든 스캔·모든 점의 잔차를 한 벡터로. (N,)"""
    return np.concatenate([residual_point_to_plane(Rs[i], ts[i], local, *planes[j])
                           for i, j, local in scans])


def cost(Rs, ts, planes, scans, huber=None):
    """0.5 * sum rho(r^2).  huber 가 없으면 최소제곱."""
    r = all_residuals(Rs, ts, planes, scans)
    s = r ** 2
    if huber is None:
        return 0.5 * float(s.sum())
    d2 = huber ** 2
    rho = np.where(s <= d2, s, 2 * huber * np.sqrt(s) - d2)
    return 0.5 * float(rho.sum())


# ══════════════════════════════════════════════════════════════════
# 포즈 + 평면을 한꺼번에 푸는 LM
# ══════════════════════════════════════════════════════════════════
# 왜 평면을 변수로 드는가 — "평면은 반복마다 닫힌형으로 다시 맞추고 포즈만 GN" 이 먼저였다.
# 야코비안은 맞았는데(중앙차분 일치) 수렴이 느렸다: 포즈 1..m-1 이 같은 방향으로 어긋나 있으면
# 그 점들로 맞춘 평면도 같이 끌려가서, 잔차는 어긋남의 1/m 만 보이고 오차는 반복마다 (m-1)/m
# 배로만 준다. 실측(포즈 10, 잡음 10 mm, 20 반복): 잔차 RMSE 는 참값 수준(7.9 vs 7.7 mm)인데
# 포즈는 1.2 cm 남아 있었다. 평면을 변수에 넣고 한 시스템으로 풀면 그 결합이 H 에 들어간다.
def parameter_layout(n_pose, n_plane):
    """미지수 배치: 포즈 i -> [6i, 6i+6),  평면 j -> [6m + 3j, 6m + 3j + 3).  포즈 0 은 뺀다."""
    n_total = 6 * n_pose + 3 * n_plane
    free = np.arange(6, n_total)
    return n_total, free


def build_normal_equation(Rs, ts, planes, scans, huber=None):
    """H = sum J^T w J,   g = sum J^T w r.   J 한 행은 [포즈 블록 1x6 | 평면 블록 1x3].

    밀집으로 만든다 — 포즈 수십, 평면 몇 개라 (6m + 3p)^2 이 작다. 스캔이 커지면
    카메라 BA 의 점 소거(ba_solver.py, Schur)와 똑같이 평면 블록을 소거하면 된다.
    """
    m, n_plane = len(Rs), len(planes)
    n_total, _ = parameter_layout(m, n_plane)
    H = np.zeros((n_total, n_total))
    g = np.zeros(n_total)
    for i, j, local in scans:
        normal, offset = planes[j]
        P_world = to_world(Rs[i], ts[i], local)
        r = P_world @ normal + offset
        J = np.hstack([jacobian_pose(P_world, normal), jacobian_plane(P_world, normal)])
        w = np.ones(len(r)) if huber is None else huber_weight(np.abs(r), huber)
        Jw = J * w[:, None]
        rows = np.r_[6 * i:6 * i + 6, 6 * m + 3 * j:6 * m + 3 * j + 3]
        # np.dot 이지 @ 가 아니다: macOS 의 numpy 2.x 는 이 모양의 J^T J 에 거짓 "divide by zero"
        # 경고를 낸다 (값은 유한 — np.dot 과 einsum 은 조용하다).
        H[np.ix_(rows, rows)] += np.dot(Jw.T, J)
        g[rows] += np.dot(Jw.T, r)
    return H, g


def apply_state_update(Rs, ts, planes, delta):
    """delta (6m + 3p,) 를 포즈(왼쪽 섭동)와 평면(접공간 섭동)에 나눠 적용한다."""
    m = len(Rs)
    Rs_new, ts_new = Rs.copy(), ts.copy()
    for i in range(m):
        Rs_new[i], ts_new[i] = apply_left_update(Rs[i], ts[i], delta[6 * i:6 * i + 6])
    planes_new = [apply_plane_update(normal, offset, delta[6 * m + 3 * j:6 * m + 3 * j + 3])
                  for j, (normal, offset) in enumerate(planes)]
    return Rs_new, ts_new, planes_new


def lidar_bundle_adjust(Rs, ts, scans, n_plane, n_iter=20, huber=None,
                        lam=1e-3, on_step=None):
    """포즈 + 평면 LM. 평면 초기값은 닫힌형 맞춤, 포즈 0 은 고정(게이지).

    on_step(iteration, Rs, ts, planes, cost, accepted, lam) 이 반복마다 불린다.
    넘기는 배열은 내부 것 그대로이므로 받는 쪽에서 복사해야 한다.
    """
    Rs, ts = Rs.copy(), ts.copy()
    planes = fit_planes(Rs, ts, scans, n_plane)
    _, free = parameter_layout(len(Rs), n_plane)
    history = []
    old_cost = cost(Rs, ts, planes, scans, huber)

    for iteration in range(n_iter):
        H, g = build_normal_equation(Rs, ts, planes, scans, huber)
        Hf = H[np.ix_(free, free)]
        Hf = Hf + lam * np.diag(np.diag(Hf))               # LM 감쇠 (Marquardt 형)
        try:
            delta_free = np.linalg.solve(Hf, -g[free])
        except np.linalg.LinAlgError:
            lam *= 10.0
            history.append(old_cost)
            if on_step is not None:
                on_step(iteration, Rs, ts, planes, old_cost, False, lam)
            continue

        delta = np.zeros(len(g))
        delta[free] = delta_free                           # 포즈 0 은 0 — 그대로
        Rs_new, ts_new, planes_new = apply_state_update(Rs, ts, planes, delta)
        new_cost = cost(Rs_new, ts_new, planes_new, scans, huber)

        accepted = new_cost < old_cost
        if accepted:
            Rs, ts, planes = Rs_new, ts_new, planes_new
            converged = abs(old_cost - new_cost) < 1e-12 * max(1.0, old_cost)
            old_cost = new_cost
            lam = max(lam / 10.0, 1e-9)
        else:
            converged = False
            lam *= 10.0

        history.append(old_cost)
        if on_step is not None:
            on_step(iteration, Rs, ts, planes, old_cost, accepted, lam)
        if converged:
            break

    return Rs, ts, planes, history


# ══════════════════════════════════════════════════════════════════
# 평가 — 포즈 오차 (포즈 0 고정이라 정렬 없이 바로 비교한다)
# ══════════════════════════════════════════════════════════════════
def pose_errors(Rs, ts, Rs_true, ts_true):
    """(평행이동 오차 [m], 회전 오차 [deg]) 포즈별."""
    trans = np.linalg.norm(ts - ts_true, axis=1)
    rot = np.array([np.degrees(np.linalg.norm(so3_log(R @ Rt.T)))
                    for R, Rt in zip(Rs, Rs_true)])
    return trans, rot


def collect_iterates(Rs0, ts0, scans, n_plane, n_iter, huber=None):
    """초기추정 + LM 반복별 상태를 모은다. on_step 이 넘기는 배열은 반드시 복사한다."""
    planes0 = fit_planes(Rs0, ts0, scans, n_plane)
    snaps = [(Rs0.copy(), ts0.copy(), planes0, float(cost(Rs0, ts0, planes0, scans, huber)),
              True, None)]

    def record(it, Rs, ts, planes, c, accepted, lam):
        snaps.append((Rs.copy(), ts.copy(), list(planes), float(c), bool(accepted), float(lam)))

    result = lidar_bundle_adjust(Rs0, ts0, scans, n_plane, n_iter=n_iter, huber=huber,
                                 on_step=record)
    return snaps, result


# ══════════════════════════════════════════════════════════════════
# 그리기
# ══════════════════════════════════════════════════════════════════
AXIS_LENGTH = 0.35


def pose_axes_polyline(Rs, ts, length=AXIS_LENGTH):
    """포즈마다 x 축(진행 방향) 살 하나 + 궤적. NaN 으로 끊은 폴리라인 하나로 묶는다.

    아티스트를 포즈마다 만들지 않아야 슬라이더 재생이 끊기지 않는다.
    """
    nan = np.full((1, 3), np.nan)
    pieces = [ts, nan]
    for R, t in zip(Rs, ts):
        pieces += [t[None], (t + length * R[:, 0])[None], nan]
    return np.vstack(pieces)


def world_points_by_plane(Rs, ts, scans, n_plane):
    """평면별로 모든 스캔의 world 점을 모은다 (그리기용)."""
    gathered = [[] for _ in range(n_plane)]
    for i, j, local in scans:
        gathered[j].append(to_world(Rs[i], ts[i], local))
    return [np.vstack(g) for g in gathered]


def setup_3d_axes(ax, pad=0.3):
    half = ROOM_HALF + pad
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(-pad, ROOM_HEIGHT + pad)
    ax.set_box_aspect((1, 1, (ROOM_HEIGHT + 2 * pad) / (2 * half)))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.tick_params(labelsize=7)


def view_converge(args):
    Rs_true, ts_true, planes_true, scans = make_lidar_scene(n_pose=args.n_pose, n_pt=args.n_pt, noise=args.noise, seed=args.seed)
    n_plane = len(ROOM_PLANES)
    Rs0, ts0 = perturb_poses(Rs_true, ts_true, seed=args.seed + 1, rot=args.rot, trans=args.trans)
    n_points = sum(len(local) for _, _, local in scans)

    snaps, (Rf, tf, planes_f, _) = collect_iterates(Rs0, ts0, scans, n_plane, args.iters, args.huber)
    costs = [s[3] for s in snaps]
    accepted = [s[4] for s in snaps]

    # 잔차 히스토그램의 축을 초기 상태로 고정한다. 프레임마다 다시 맞추면 "줄어든다"는
    # 사실 자체가 안 보인다.
    initial_abs = np.abs(all_residuals(Rs0, ts0, snaps[0][2], scans))
    residual_xmax = float(np.percentile(initial_abs, 99)) * 1.05
    residual_xmin = args.noise * 0.05

    fig = plt.figure(figsize=(14.5, 9.0))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.155, hspace=0.38, wspace=0.20)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_cost = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, 0])
    ax_hist = fig.add_subplot(grid[1, 1])

    # ── 좌상: 3D 씬 — 평면별 점 + 포즈 ──
    setup_3d_axes(ax3d)
    truth_poly = pose_axes_polyline(Rs_true, ts_true)
    ax3d.plot(truth_poly[:, 0], truth_poly[:, 1], truth_poly[:, 2],
              color=COLOR_TRUE, lw=1.6, ls="--", label=L("참값 포즈", "true poses"))
    point_lines = []
    for j in range(n_plane):
        (line,) = ax3d.plot([], [], [], ".", color=PLANE_COLORS[j % len(PLANE_COLORS)],
                            ms=2.0, alpha=0.55, mew=0)
        point_lines.append(line)
    (line_pose,) = ax3d.plot([], [], [], color=COLOR_POSE, lw=1.4,
                             label=L("추정 포즈", "estimated poses"))
    (line_err,) = ax3d.plot([], [], [], color=COLOR_ERR, lw=1.0, alpha=0.8,
                            label=L("포즈 오차", "pose error"))
    ax3d.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax3d.view_init(elev=28, azim=-55)

    # ── 우상: cost 수렴 곡선 ──
    ax_cost.set_yscale("log")
    ax_cost.plot(range(len(costs)), costs, "-", color="#5f6368", lw=1.2, zorder=1)
    reject_x = [k for k in range(1, len(costs)) if not accepted[k]]
    ax_cost.plot(range(len(costs)), costs, "o", color=COLOR_POSE, ms=4, zorder=2,
                 label=L("채택", "accepted"))
    if reject_x:
        ax_cost.plot(reject_x, [costs[k] for k in reject_x], "x", color=COLOR_ERR,
                     ms=7, mew=1.6, zorder=3, label=L("거부 (lam 증가)", "rejected (lam up)"))
    (cost_marker,) = ax_cost.plot([], [], "o", color=COLOR_ERR, ms=11, mfc="none", mew=2, zorder=4)
    ax_cost.set_xlabel(L("LM 반복", "LM iteration"))
    ax_cost.set_ylabel(L("cost (로그)", "cost (log)"))
    ax_cost.set_title(L("수렴 곡선  0.5·Σρ(r²)", "convergence  0.5*sum rho(r^2)"), fontsize=10)
    ax_cost.grid(alpha=0.25)
    ax_cost.legend(fontsize=8)

    # ── 좌하: 포즈별 오차 ──
    pose_index = np.arange(len(Rs_true))
    trans0, rot0 = pose_errors(Rs0, ts0, Rs_true, ts_true)
    ax_err.set_xlabel(L("포즈", "pose"))
    ax_err.set_ylabel(L("평행이동 오차 [m]", "translation error [m]"), color=COLOR_POSE)
    ax_err.set_ylim(0, max(trans0.max() * 1.15, 1e-3))
    ax_err.set_xticks(pose_index)
    ax_err.grid(alpha=0.25)
    ax_err.bar(pose_index - 0.2, trans0, width=0.4, color=COLOR_POSE, alpha=0.25,
               label=L("초기 평행이동", "initial trans"))
    bars_trans = ax_err.bar(pose_index - 0.2, trans0, width=0.4, color=COLOR_POSE,
                            label=L("현재 평행이동", "current trans"))
    ax_err_rot = ax_err.twinx()
    ax_err_rot.set_ylabel(L("회전 오차 [deg]", "rotation error [deg]"), color=COLOR_ERR)
    ax_err_rot.set_ylim(0, max(rot0.max() * 1.15, 1e-3))
    ax_err_rot.bar(pose_index + 0.2, rot0, width=0.4, color=COLOR_ERR, alpha=0.25,
                   label=L("초기 회전", "initial rot"))
    bars_rot = ax_err_rot.bar(pose_index + 0.2, rot0, width=0.4, color=COLOR_ERR,
                              label=L("현재 회전", "current rot"))
    handles = [ax_err.containers[0], bars_trans, ax_err_rot.containers[0], bars_rot]
    ax_err.legend(handles, [h.get_label() for h in handles], fontsize=7, loc="upper right")

    # ── 우하: 잔차 히스토그램 (로그 x) ──
    ax_hist.set_xscale("log")
    ax_hist.set_xlim(residual_xmin, residual_xmax)
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel(L("|r| = 평면까지 거리 [m]", "|r| = distance to plane [m]"))
    ax_hist.set_ylabel(L("점 수 (로그)", "points (log)"))
    ax_hist.grid(alpha=0.25)
    hist_bins = np.logspace(np.log10(residual_xmin), np.log10(residual_xmax), 46)
    ticks = [t for t in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
             if residual_xmin <= t <= residual_xmax]
    ax_hist.set_xticks(ticks)
    ax_hist.set_xticklabels([("%g" % t) for t in ticks])
    ax_hist.minorticks_off()
    ax_hist.axvline(args.noise, color=COLOR_TRUE, ls="--", lw=1.0)
    from matplotlib.transforms import blended_transform_factory
    ax_hist.text(args.noise, 0.96, L(" 잡음 σ", " noise sigma"), color="#5f6368",
                 fontsize=8, va="top", ha="left",
                 transform=blended_transform_factory(ax_hist.transData, ax_hist.transAxes))

    slider_axes = fig.add_axes([0.10, 0.045, 0.56, 0.022])
    slider = Slider(slider_axes, L("반복", "iter"), 0, len(snaps) - 1, valinit=0, valstep=1)
    button_axes = fig.add_axes([0.72, 0.038, 0.09, 0.038])
    button = Button(button_axes, L("▶ 재생", "> play"))

    state = {"playing": False, "bars": None}

    def draw(k):
        Rs_k, ts_k, planes_k, cost_k, accepted_k, lam_k = snaps[k]

        for j, points in enumerate(world_points_by_plane(Rs_k, ts_k, scans, n_plane)):
            point_lines[j].set_data_3d(points[:, 0], points[:, 1], points[:, 2])
        poly = pose_axes_polyline(Rs_k, ts_k)
        line_pose.set_data_3d(poly[:, 0], poly[:, 1], poly[:, 2])
        nan = np.full((1, 3), np.nan)
        segments = np.vstack([v for a, b in zip(ts_k, ts_true) for v in (a[None], b[None], nan)])
        line_err.set_data_3d(segments[:, 0], segments[:, 1], segments[:, 2])

        cost_marker.set_data([k], [cost_k])

        trans_k, rot_k = pose_errors(Rs_k, ts_k, Rs_true, ts_true)
        for bar, value in zip(bars_trans, trans_k):
            bar.set_height(value)
        for bar, value in zip(bars_rot, rot_k):
            bar.set_height(value)

        abs_r = np.abs(all_residuals(Rs_k, ts_k, planes_k, scans))
        if state["bars"] is not None:
            state["bars"].remove()
        counts, _ = np.histogram(abs_r, bins=hist_bins)
        state["bars"] = ax_hist.bar(hist_bins[:-1], np.maximum(counts, 1e-9),
                                    width=np.diff(hist_bins), align="edge",
                                    color=COLOR_POSE, alpha=0.85)
        ax_hist.set_ylim(0.7, max(10, counts.max() * 1.5))
        rmse = float(np.sqrt((abs_r ** 2).mean()))

        tag = "" if k == 0 else ("  " + (L("채택", "accepted") if accepted_k
                                         else L("거부 → lam 증가", "rejected -> lam up")))
        lam_text = "" if lam_k is None else f"   lam={lam_k:.1e}"
        ax3d.set_title(L(f"스캔을 현재 포즈로 world 에 올린 것   반복 {k}/{len(snaps)-1}\n"
                         f"평행이동 오차 평균 {trans_k[1:].mean():.4f} m   회전 {rot_k[1:].mean():.3f}°",
                         f"scans in world at current poses   iter {k}/{len(snaps)-1}\n"
                         f"mean trans err {trans_k[1:].mean():.4f} m   rot {rot_k[1:].mean():.3f} deg"),
                       fontsize=9)
        ax_err.set_title(L("포즈별 오차 (포즈 0 = 게이지, 고정)",
                           "per-pose error (pose 0 = gauge, fixed)"), fontsize=10)
        ax_hist.set_title(L(f"잔차 분포   RMSE = {rmse * 1000:.2f} mm",
                            f"residuals   RMSE = {rmse * 1000:.2f} mm"), fontsize=10)
        fig.suptitle(L(f"LiDAR BA (point-to-plane) — 포즈 {len(Rs_true)}개 · 평면 {n_plane}개 · "
                       f"점 {n_points:,}개   |   cost = {cost_k:.4g}{lam_text}{tag}",
                       f"LiDAR BA (point-to-plane) - {len(Rs_true)} poses / {n_plane} planes / "
                       f"{n_points:,} points   |   cost = {cost_k:.4g}{lam_text}{tag}"),
                     fontsize=12, y=0.975)
        fig.canvas.draw_idle()

    slider.on_changed(lambda value: draw(int(value)))
    draw(0)

    timer = fig.canvas.new_timer(interval=250)

    def tick():
        k = int(slider.val)
        slider.set_val(0 if k >= len(snaps) - 1 else k + 1)

    timer.add_callback(tick)

    def toggle(_event):
        state["playing"] = not state["playing"]
        if state["playing"]:
            timer.start(); button.label.set_text(L("❙❙ 정지", "|| stop"))
        else:
            timer.stop(); button.label.set_text(L("▶ 재생", "> play"))
        fig.canvas.draw_idle()

    button.on_clicked(toggle)

    trans_f, rot_f = pose_errors(Rf, tf, Rs_true, ts_true)
    rmse_of = lambda Rs_, ts_, planes_: float(np.sqrt((all_residuals(Rs_, ts_, planes_, scans) ** 2).mean()))
    print(f"  {L('초기', 'initial')}: RMSE {rmse_of(Rs0, ts0, snaps[0][2]) * 1000:7.2f} mm"
          f"   {L('평행이동', 'trans')} {trans0[1:].mean():.4f} m   {L('회전', 'rot')} {rot0[1:].mean():.3f}°")
    print(f"  {L('BA 후', 'final')}  : RMSE {rmse_of(Rf, tf, planes_f) * 1000:7.2f} mm"
          f"   {L('평행이동', 'trans')} {trans_f[1:].mean():.4f} m   {L('회전', 'rot')} {rot_f[1:].mean():.3f}°")
    print(f"  {L('참값', 'truth')}  : RMSE {rmse_of(Rs_true, ts_true, planes_true) * 1000:7.2f} mm"
          f"   <- {L('주입 잡음', 'injected noise')} {args.noise * 1000:.0f} mm")
    print(f"  {L('LM 반복', 'LM iterations')} = {len(snaps)-1}"
          f"  ({L('거부', 'rejected')} {sum(1 for a in accepted[1:] if not a)})")

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        for tag, k in (("initial", 0), ("mid", len(snaps) // 2), ("final", len(snaps) - 1)):
            draw(k)
            slider.set_val(k)
            path = os.path.join(args.save, f"lidar_{tag}.png")
            fig.savefig(path, dpi=110)
            print(f"  saved {path}")
        return
    plt.show()


# ══════════════════════════════════════════════════════════════════
# 뷰 2 — 회전형 LiDAR 의 원시 점
# ══════════════════════════════════════════════════════════════════
def object_wireframe(obj, circle_steps=48):
    """물체 하나의 외곽선을 NaN 으로 끊은 폴리라인 하나로.

    view_converge 의 pose_axes_polyline 과 같은 이유다 — 물체마다 아티스트를 만들면
    슬라이더 재생이 끊긴다.
    """
    nan = np.full((1, 3), np.nan)
    if obj.type is EObjectType.PLANE:
        half_u, half_v = obj.size[0], obj.size[1]
        corners = np.array([[-half_u, -half_v, 0.0], [half_u, -half_v, 0.0],
                            [half_u, half_v, 0.0], [-half_u, half_v, 0.0], [-half_u, -half_v, 0.0]])
        return np.vstack([np.dot(corners, obj.rotation.T) + obj.position, nan])
    if obj.type is EObjectType.SPHERE:
        radius = obj.size[0]
        angle = np.linspace(0.0, 2.0 * np.pi, circle_steps)
        cos, sin, zero = radius * np.cos(angle), radius * np.sin(angle), np.zeros(circle_steps)
        rings = [np.column_stack([cos, sin, zero]),
                 np.column_stack([cos, zero, sin]),
                 np.column_stack([zero, cos, sin])]
        return np.vstack([piece for ring in rings for piece in (ring + obj.position, nan)])
    half = obj.half_extent()
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    corners = signs * half
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    pieces = []
    for a, b in edges:
        pieces += [corners[[a, b]], nan]
    return np.dot(np.vstack(pieces), obj.rotation.T) + obj.position


def scene_wireframe(objects):
    """장면 전체를 배열 하나로."""
    return np.vstack([object_wireframe(obj) for obj in objects])


def view_raw(args):
    """N 채널 회전형 LiDAR 가 실제로 주는 것을 본다 — 링 구조, 거리 영상, 누적 점군."""
    spec = LidarSpec(channels=args.channels, 
                     azimuth_steps=args.azimuth_steps,
                     elevation_min_deg=args.elevation[0], 
                     elevation_max_deg=args.elevation[1],
                     max_range=args.max_range, 
                     range_noise=args.noise, 
                     dropout=args.dropout)
    objects = make_room_objects(props=not args.no_props)
    Rs, ts, scans = make_lidar_data(objects, 
                                    n_pose=args.n_pose, 
                                    spec=spec,
                                    seed=args.seed, 
                                    motion=args.motion)

    ring_colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, spec.channels))
    total_rays = spec.channels * spec.azimuth_steps

    fig = plt.figure(figsize=(14.5, 9.0))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.155, hspace=0.38, wspace=0.22)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_image = fig.add_subplot(grid[0, 1])
    ax_top = fig.add_subplot(grid[1, 0])
    ax_ring = fig.add_subplot(grid[1, 1])

    # ── 좌상: 한 스캔을 world 에서, 채널별 색 ──
    setup_3d_axes(ax3d)
    wire = scene_wireframe(objects)
    ax3d.plot(wire[:, 0], wire[:, 1], wire[:, 2], color=COLOR_TRUE, lw=1.0, alpha=0.8)
    trajectory = pose_axes_polyline(Rs, ts)
    ax3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
              color=COLOR_TRUE, lw=1.2, ls="--", alpha=0.7)
    ring_lines = []
    for channel in range(spec.channels):
        (line,) = ax3d.plot([], [], [], ".", color=ring_colors[channel], ms=1.6, mew=0, alpha=0.85)
        ring_lines.append(line)
    (sensor_marker,) = ax3d.plot([], [], [], "o", color=COLOR_ERR, ms=7, mew=0)
    ax3d.view_init(elev=26, azim=-58)

    # ── 우상: 거리 영상 — 회전형 LiDAR 의 원시 형태 그대로 (행 = 채널, 열 = 방위각) ──
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#2b2b2b")                 # 반환 없음
    blank = np.full((spec.channels, spec.azimuth_steps), np.nan)
    image = ax_image.imshow(blank, aspect="auto", origin="lower", cmap=colormap,
                            extent=(0.0, 360.0, -0.5, spec.channels - 0.5),
                            vmin=0.0, vmax=min(spec.max_range, 1.15 * max(s.ranges.max() for s in scans)))
    ax_image.set_xlabel(L("방위각 [deg]", "azimuth [deg]"))
    ax_image.set_ylabel(L("채널", "channel"))
    fig.colorbar(image, ax=ax_image, pad=0.02, label=L("거리 [m]", "range [m]"))

    # ── 좌하: 위에서 본 누적 점군 — 포즈별 색 ──
    ax_top.set_aspect("equal")
    ax_top.set_xlim(-ROOM_HALF - 0.3, ROOM_HALF + 0.3)
    ax_top.set_ylim(-ROOM_HALF - 0.3, ROOM_HALF + 0.3)
    ax_top.set_xlabel("x [m]"); ax_top.set_ylabel("y [m]")
    ax_top.grid(alpha=0.25)
    pose_colors = plt.get_cmap("plasma")(np.linspace(0.1, 0.9, len(scans)))
    stride = max(1, sum(len(s.points) for s in scans) // 60000)
    for scan in scans:
        world = to_world(Rs[scan.pose_index], ts[scan.pose_index], scan.points)[::stride]
        ax_top.plot(world[:, 0], world[:, 1], ".", ms=1.2, mew=0, alpha=0.7,
                    color=pose_colors[scan.pose_index], rasterized=True)
    # 물체 윤곽을 얹는다 — 점만 있으면 무엇을 보고 있는지 알 수 없다
    ax_top.plot(wire[:, 0], wire[:, 1], color="#3c4043", lw=0.9, alpha=0.9)
    ax_top.plot(ts[:, 0], ts[:, 1], "-o", color=COLOR_ERR, ms=3, lw=1.0, label=L("궤적", "trajectory"))
    ax_top.legend(fontsize=8, loc="upper right")

    # ── 우하: 물체별 반환 수 ──
    # 채널별로 세면 닫힌 방에서는 전부 한 바퀴 빔 수라 아무것도 안 보인다. 물체별로 세면
    # 포즈가 움직일 때 무엇이 무엇을 가리는지가 드러난다.
    object_colors = plt.get_cmap("tab10")(np.arange(len(objects)) % 10)
    ax_object = ax_ring
    ax_object.set_xlabel(L("반환 점 수", "returns"))
    ax_object.grid(alpha=0.25, axis="x")
    ax_object.set_yticks(np.arange(len(objects)))
    ax_object.set_yticklabels([obj.name for obj in objects], fontsize=8)
    ax_object.invert_yaxis()
    object_bars = ax_object.barh(np.arange(len(objects)), np.zeros(len(objects)),
                                 color=object_colors, height=0.75)
    ax_object.set_xlim(0, max(np.bincount(s.object_index, minlength=len(objects)).max()
                              for s in scans) * 1.1)

    slider_axes = fig.add_axes([0.10, 0.045, 0.56, 0.022])
    slider = Slider(slider_axes, L("포즈", "pose"), 0, len(scans) - 1, valinit=0, valstep=1)
    button_axes = fig.add_axes([0.72, 0.038, 0.09, 0.038])
    button = Button(button_axes, L("▶ 재생", "> play"))
    state = {"playing": False}

    def draw(k):
        scan = scans[k]
        world = to_world(Rs[k], ts[k], scan.points)
        for channel in range(spec.channels):
            selected = world[scan.ring == channel]
            ring_lines[channel].set_data_3d(selected[:, 0], selected[:, 1], selected[:, 2])
        sensor_marker.set_data_3d(ts[k, 0:1], ts[k, 1:2], ts[k, 2:3])

        grid_image = np.full((spec.channels, spec.azimuth_steps), np.nan)
        grid_image[scan.ring, scan.azimuth_index] = scan.ranges
        image.set_data(grid_image)

        counts = np.bincount(scan.object_index, minlength=len(objects))
        for bar, value in zip(object_bars, counts):
            bar.set_width(value)

        returned = len(scan.points)
        ax3d.set_title(L(f"스캔 {k} 을 참값 포즈로 world 에   채널별 색\n"
                         f"반환 {returned:,} / {total_rays:,} 빔 ({100 * returned / total_rays:.1f} %)",
                         f"scan {k} in world at true pose, coloured by channel\n"
                         f"{returned:,} / {total_rays:,} beams returned ({100 * returned / total_rays:.1f} %)"),
                       fontsize=9)
        ax_image.set_title(L(f"거리 영상 — 원시 반환 그대로   중앙값 {np.median(scan.ranges):.2f} m",
                             f"range image - raw returns   median {np.median(scan.ranges):.2f} m"),
                           fontsize=10)
        ax_object.set_title(L("물체별 반환 수 — 포즈가 움직이면 가림이 바뀐다",
                              "returns per object - occlusion changes with pose"), fontsize=10)
        fig.canvas.draw_idle()

    ax_top.set_title(L(f"위에서 본 누적 점군 (포즈 {len(scans)}개, {stride} 점마다 하나)",
                       f"accumulated cloud from above ({len(scans)} poses, every {stride})"), fontsize=10)
    motion_note = L("  운동 왜곡 켜짐", "  motion distortion on") if args.motion else ""
    fig.suptitle(L(f"회전형 LiDAR 원시 점 — {spec.channels} 채널 · 방위각 {spec.azimuth_steps} 분할 · "
                   f"고도 {spec.elevation_min_deg:g}~{spec.elevation_max_deg:g}° · "
                   f"물체 {len(objects)}개{motion_note}",
                   f"rotating LiDAR raw points - {spec.channels} channels / {spec.azimuth_steps} azimuth / "
                   f"elevation {spec.elevation_min_deg:g}..{spec.elevation_max_deg:g} deg / "
                   f"{len(objects)} objects{motion_note}"), fontsize=12, y=0.975)

    slider.on_changed(lambda value: draw(int(value)))
    draw(0)

    timer = fig.canvas.new_timer(interval=300)

    def tick():
        k = int(slider.val)
        slider.set_val(0 if k >= len(scans) - 1 else k + 1)

    timer.add_callback(tick)

    def toggle(_event):
        state["playing"] = not state["playing"]
        if state["playing"]:
            timer.start(); button.label.set_text(L("❙❙ 정지", "|| stop"))
        else:
            timer.stop(); button.label.set_text(L("▶ 재생", "> play"))
        fig.canvas.draw_idle()

    button.on_clicked(toggle)

    hit_counts = np.bincount(np.concatenate([s.object_index for s in scans]), minlength=len(objects))
    total_points = sum(len(s.points) for s in scans)
    print(f"  {L('스캔', 'scans')} {len(scans)}   {L('점', 'points')} {total_points:,}"
          f"   {L('반환율', 'return rate')} {100 * total_points / (total_rays * len(scans)):.1f} %")
    for obj, count in zip(objects, hit_counts):
        print(f"    {obj.name:10s} {count:8,d}  ({100 * count / total_points:5.1f} %)")
    if args.motion:
        offset = np.concatenate([np.linalg.norm(to_world(Rs[s.pose_index], ts[s.pose_index], s.points)
                                                - s.world_truth, axis=1) for s in scans])
        print(f"  {L('강체 가정의 오차', 'rigid-scan error')}: "
              f"{L('평균', 'mean')} {1000 * offset.mean():.1f} mm   "
              f"{L('최대', 'max')} {1000 * offset.max():.1f} mm")

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        for k in (0, len(scans) // 2):
            draw(k)
            slider.set_val(k)
            path = os.path.join(args.save, f"raw_pose{k}.png")
            fig.savefig(path, dpi=110)
            print(f"  saved {path}")
        return
    plt.show()


# ══════════════════════════════════════════════════════════════════
# 자기검증 — 야코비안 · 수렴 · 레이캐스팅 · 콜백
# ══════════════════════════════════════════════════════════════════
def selftest(args):
    """
    1. 문서의 J_T = [-n^T [P]_x, n^T] 가 왼쪽 섭동의 중앙차분과 맞는가 (무작위 200회).
       회전 블록은 규약에 민감하다 — 오른쪽 섭동으로 재면 이 검사가 빨개진다 (뮤테이션 확인:
       외적 순서를 뒤집으면 상대오차 2.0 으로 실패).
    2. 평면 블록 J = [P^T B, 1] 이 접공간 섭동의 중앙차분과 맞는가.
    3. 합성 방에서 BA 가 포즈를 잡음 수준으로 되돌리는가. 문턱은 잡음 10 mm · 평면당 점 80 개에서
       실측한 값(평행이동 평균 2 mm)의 여유 있는 배수다 — 교대식(평면 재맞춤 + 포즈만)이면
       20 반복 뒤 1.2 cm 가 남아 이 검사가 빨개진다.
    4. on_step 이 넘기는 상태와 cost 가 짝이 맞는가 (ba_viz.py 와 같은 검사).
    """
    rng = np.random.default_rng(0)
    worst_pose = worst_plane = 0.0
    for _ in range(200):
        R = so3_exp(rng.normal(scale=0.8, size=3))
        t = rng.normal(scale=1.5, size=3)
        P = rng.normal(scale=2.0, size=(5, 3))
        normal = rng.normal(size=3); normal /= np.linalg.norm(normal)
        offset = float(rng.normal())
        P_world = to_world(R, t, P)
        numeric = numeric_jacobian_pose(R, t, P, normal, offset)
        worst_pose = max(worst_pose, np.abs(jacobian_pose(P_world, normal) - numeric).max()
                         / max(1.0, np.abs(numeric).max()))
        numeric = numeric_jacobian_plane(R, t, P, normal, offset)
        worst_plane = max(worst_plane, np.abs(jacobian_plane(P_world, normal) - numeric).max()
                          / max(1.0, np.abs(numeric).max()))

    Rs_true, ts_true, _, scans = make_lidar_scene(n_pose=8, n_pt=80, noise=0.01, seed=3)
    n_plane = len(ROOM_PLANES)
    Rs0, ts0 = perturb_poses(Rs_true, ts_true, seed=4, rot=0.05, trans=0.15)
    snaps, (Rf, tf, _, hist) = collect_iterates(Rs0, ts0, scans, n_plane, 20)
    trans0, rot0 = pose_errors(Rs0, ts0, Rs_true, ts_true)
    trans_f, rot_f = pose_errors(Rf, tf, Rs_true, ts_true)
    converge_ok = trans_f[1:].max() < 0.006 and rot_f[1:].max() < 0.1

    worst_pair = 0.0
    for (Rs_k, ts_k, planes_k, cost_k, _, _) in snaps:
        worst_pair = max(worst_pair, abs(cost_k - cost(Rs_k, ts_k, planes_k, scans)))
    timing_ok = True
    for k in range(1, len(snaps)):
        changed = not np.array_equal(snaps[k][1], snaps[k - 1][1])
        if changed != snaps[k][4]:
            timing_ok = False
    gauge_ok = np.array_equal(Rf[0], Rs_true[0]) and np.array_equal(tf[0], ts_true[0])

    checks = [
        (f"J_T (포즈) 해석 vs 중앙차분 (최대 상대오차 {worst_pose:.2e})", worst_pose < 1e-5),
        (f"J (평면) 해석 vs 중앙차분 (최대 상대오차 {worst_plane:.2e})", worst_plane < 1e-5),
        (f"수렴 {len(hist)}반복: 평행이동 {trans0[1:].mean()*1000:.1f} -> {trans_f[1:].mean()*1000:.2f} mm "
         f"(최대 {trans_f[1:].max()*1000:.2f}), 회전 {rot0[1:].mean():.2f} -> {rot_f[1:].mean():.3f} deg "
         f"(최대 {rot_f[1:].max():.3f})", converge_ok),
        (f"상태-cost 짝 (최대오차 {worst_pair:.2e})", worst_pair < 1e-9),
        ("상태는 채택된 반복에서만 바뀐다", timing_ok),
        ("on_step 횟수 == hist 길이", len(snaps) - 1 == len(hist)),
        ("마지막 스냅샷 == 반환값", np.allclose(snaps[-1][0], Rf) and np.allclose(snaps[-1][1], tf)),
        ("포즈 0 은 게이지로 그대로", gauge_ok),
    ]
    checks += raycast_checks()
    checks += bundle_checks()

    print("LiDAR BA 자기검증")
    ok = True
    for name, passed in checks:
        print(f"  {name:88s} {'OK' if passed else 'FAIL'}")
        ok &= bool(passed)
    print("  ==> " + ("전부 통과" if ok else "실패 있음"))
    return 0 if ok else 1


def raycast_checks():
    """SceneObject 의 레이캐스팅과 회전형 LiDAR 를 검증한다.

    주 단언은 '반환된 점이 정말 그 물체 표면 위에 있는가'(surface_residual)다 — 거리만
    대조하면 법선 버그가 통과하고, 법선만 보면 교차 판정 버그가 통과한다. 실제로 겪은 것:
    평면 판정에서 inf 자리까지 로컬 좌표를 계산하면 축에 평행한 광선이 nan 이 되는데,
    거리 비교만으로는 안 잡힌다.
    """
    checks = []

    # 1. 해석해와 대조 — 중심을 정면으로 겨눈 광선은 거리·법선이 손으로 나온다
    origin = np.zeros(3)
    forward = np.array([[1.0, 0.0, 0.0]])
    sphere = SceneObject(EObjectType.SPHERE, (5, 0, 0), 1.0)
    cube = SceneObject(EObjectType.CUBE, (5, 0, 0), (1.0, 1.0, 1.0))
    plane = SceneObject(EObjectType.PLANE, (0, 0, 0), (1.0, 1.0), rotation_from_normal((0, 0, 1)))
    analytic = []
    for obj in (sphere, cube):
        distance, normal = obj.hit(origin, forward)
        analytic.append(abs(distance[0] - 4.0) < 1e-12
                        and np.allclose(normal[0], [-1.0, 0.0, 0.0], atol=1e-12))
    distance, normal = plane.hit(np.array([0.0, 0.0, 2.0]), np.array([[0.0, 0.0, -1.0]]))
    analytic.append(abs(distance[0] - 2.0) < 1e-12 and np.allclose(normal[0], [0, 0, 1], atol=1e-12))
    # 유한한 판이므로 범위 밖은 맞지 않는다. 판을 무한 평면으로 다루면 여기가 빨개진다.
    outside, _ = plane.hit(np.array([1.5, 0.0, 2.0]), np.array([[0.0, 0.0, -1.0]]))
    analytic.append(not np.isfinite(outside[0]))
    # 뒤쪽 물체는 맞지 않는다
    behind, _ = sphere.hit(origin, np.array([[-1.0, 0.0, 0.0]]))
    analytic.append(not np.isfinite(behind[0]))
    checks.append(("레이캐스팅 해석해 (구·정육면체·판·범위밖·뒤쪽)", all(analytic)))

    # 1b. 뒷면과 내부 — 법선은 언제나 광선을 마주 봐야 한다.
    # 닫힌 방만 쏘면 이 분기가 한 번도 안 돌아서, 부호를 지워도 검사가 통과했다 (뮤테이션 확인).
    facing_cases = []
    back_distance, back_normal = plane.hit(np.array([0.0, 0.0, -2.0]), np.array([[0.0, 0.0, 1.0]]))
    facing_cases.append(abs(back_distance[0] - 2.0) < 1e-12
                        and np.allclose(back_normal[0], [0.0, 0.0, -1.0], atol=1e-12))
    for obj in (SceneObject(EObjectType.SPHERE, (0, 0, 0), 1.0),
                SceneObject(EObjectType.CUBE, (0, 0, 0), (1.0, 1.0, 1.0))):
        inside_distance, inside_normal = obj.hit(origin, forward)      # 물체 안에서 바깥을 본다
        facing_cases.append(abs(inside_distance[0] - 1.0) < 1e-12
                            and np.allclose(inside_normal[0], [-1.0, 0.0, 0.0], atol=1e-12))
    checks.append(("판 뒷면·물체 내부에서도 법선이 광선을 마주 본다", all(facing_cases)))

    # 2. 방 + 소품에 무작위 광선 — 맞은 점이 그 물체 표면 위에 있는가
    objects = make_room_objects()
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(4000, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    sensor = np.array([0.3, -0.2, 0.9])
    distance, normal, object_index = cast_rays(objects, sensor, directions)
    hit = np.isfinite(distance)
    points = sensor + distance[hit, None] * directions[hit]
    worst_surface = max(surface_residual(objects[k], points[object_index[hit] == k]).max()
                        for k in np.unique(object_index[hit]))
    unit_error = np.abs(np.linalg.norm(normal[hit], axis=1) - 1.0).max()
    facing = bool((np.einsum("ij,ij->i", directions[hit], normal[hit]) < 0).all())
    # 닫힌 방이므로 모든 광선이 무언가를 맞아야 한다 — 하나라도 새면 판의 이음매가 벌어진 것
    checks.append((f"광선이 표면에 닿는가 (잔차 최대 {worst_surface:.1e} m, 미스 {(~hit).sum()})",
                   worst_surface < 1e-9 and (~hit).sum() == 0))
    checks.append((f"법선이 단위이고 광선을 마주 보는가 (길이오차 {unit_error:.1e})",
                   unit_error < 1e-9 and facing))

    # 3. 가림 — 가까운 물체가 먼 것을 가린다. 벽 앞의 구를 겨누면 구가 나와야 한다
    wall = SceneObject(EObjectType.PLANE, (4, 0, 0), (3.0, 3.0), rotation_from_normal((-1, 0, 0)))
    blocker = SceneObject(EObjectType.SPHERE, (2, 0, 0), 0.5)
    spread = np.zeros((200, 3)); spread[:, 0] = 1.0
    spread[:, 1] = np.linspace(-0.2, 0.2, 200)
    spread /= np.linalg.norm(spread, axis=1)[:, None]
    _, _, picked = cast_rays([wall, blocker], sensor * 0.0, spread)
    checks.append(("가림 — 벽 앞의 구가 이긴다", bool((picked == 1).all())))

    # 4. 회전형 스캔의 원시 측정 일관성
    spec = LidarSpec(range_noise=0.0, azimuth_steps=180)
    Rs, ts, scans = make_lidar_data(objects, n_pose=3, spec=spec, seed=0)
    scan = scans[1]
    directions_all, _, _ = beam_directions(spec)
    expected = scan.ranges[:, None] * directions_all[scan.ring, scan.azimuth_index]
    range_ok = np.abs(np.linalg.norm(scan.points, axis=1) - scan.ranges).max() < 1e-9
    beam_ok = np.abs(scan.points - expected).max() < 1e-12
    # 운동이 없으면 참 포즈로 올린 것이 참 위치와 같아야 한다
    rigid_ok = np.abs(to_world(Rs[1], ts[1], scan.points) - scan.world_truth).max() < 1e-9
    world = to_world(Rs[1], ts[1], scan.points)
    worst_scan = max(surface_residual(objects[k], world[scan.object_index == k]).max()
                     for k in np.unique(scan.object_index))
    checks.append((f"스캔 원시값 = (거리, 빔각) 뿐 (|p|-r {'OK' if range_ok else 'FAIL'}, "
                   f"빔각 {'OK' if beam_ok else 'FAIL'}, 표면 잔차 {worst_scan:.1e} m)",
                   range_ok and beam_ok and rigid_ok and worst_scan < 1e-9))

    # 5. 운동 왜곡은 켰을 때만 생긴다 — 껐는데 생기면 포즈 보간이 새는 것이다
    _, _, moving = make_lidar_data(objects, n_pose=3, spec=spec, seed=0, motion=True)
    skew = np.linalg.norm(to_world(Rs[1], ts[1], moving[1].points) - moving[1].world_truth, axis=1)
    checks.append((f"운동 왜곡: 끄면 0, 켜면 보인다 (평균 {1000 * skew.mean():.0f} mm)",
                   rigid_ok and skew.mean() > 0.01))

    # 6. 평면 라벨이 스캔에 없다 — 이게 이 생성기의 존재 이유다
    fields = set(LidarScan.__dataclass_fields__)
    checks.append(("스캔에 평면 라벨이 없다", "plane_index" not in fields and "planes" not in fields))
    return checks

# ══════════════════════════════════════════════════════════════════
# 장면 도형 — SceneObject 와 레이캐스팅
# ══════════════════════════════════════════════════════════════════
# 위쪽 make_lidar_scene 은 평면 위에 점을 직접 뿌린다. 그건 "어느 점이 어느 평면인지" 를
# 생성기가 알려주는 것이라, 실데이터에 없는 정보를 BA 에 넘긴다. 여기서는 반대로 간다 —
# 장면에 물체를 놓고 광선을 쏴서, 실제 센서가 주는 것(거리 + 빔 각도)만 남긴다.
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
    """점이 그 물체 표면에서 얼마나 떨어져 있는가 [m]. 레이캐스팅이 맞으면 0 이다.

    거리만 대조하는 검사는 법선 버그를 통과시킨다. 이 값은 교차 판정·거리·로컬 프레임 변환
    중 어느 하나만 틀려도 커지므로 자기검증의 주 단언으로 쓴다.
    """
    local = np.dot(points - obj.position, obj.rotation)
    if obj.type is EObjectType.SPHERE:
        return np.abs(np.linalg.norm(points - obj.position, axis=1) - obj.size[0])
    if obj.type is EObjectType.PLANE:
        return np.abs(local[:, 2])
    half = obj.half_extent()
    return np.abs((np.abs(local) / half).max(1) - 1.0) * half.min()


def rotation_from_normal(normal, up=(0.0, 0.0, 1.0)):
    """로컬 +z 가 normal, 로컬 +y 가 되도록 up 을 향하는 정규직교 회전.

    plane_basis 의 임의 기저를 그대로 쓰면 판의 '가로'가 무엇인지 정해지지 않아 size 를
    읽을 수 없다. 여기서 축을 못박는다.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    up = np.asarray(up, dtype=float)
    if abs(up @ normal) > 0.9:                       # 바닥·천장이면 up 을 바꿔 잡는다
        up = np.array([0.0, 1.0, 0.0])
    u = np.cross(up, normal)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(normal, u), normal])


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


def interpolate_pose(R_from, t_from, R_to, t_to, fraction, base=None):
    """(R_from, t_from) 에서 (R_to, t_to) 로 fraction 만큼 간 포즈.

    회전은 SO(3) 안에서 간다 — 행렬을 선형보간하면 군을 벗어난다. base 를 주면 그 포즈에
    같은 크기의 운동을 얹는다 (마지막 키프레임처럼 다음 구간이 없을 때 직전 운동을 잇는 용도).
    """
    turn = fraction * so3_log(R_from.T @ R_to)
    shift = fraction * (t_to - t_from)
    R_base, t_base = (R_from, t_from) if base is None else base
    return R_base @ so3_exp(turn), t_base + shift


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

from typing import List

def show_point_cloud(scans: List[np.ndarray]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, points in enumerate(scans):
        color = np.random.rand(3)

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=2,
            color=color,
            label=f"Frame {i}"
        )

    ax.legend()
    plt.show()


def estimate_normals(points, k=20):
    """각 점의 법선 = 가장 가까운 이웃 k 개의 산란행렬 최소고유벡터.

    이웃은 KD-tree 로 찾는다. 전수 비교는 점마다 전체를 정렬해서 O(N^2 log N) 이라 밀도를
    올리는 순간 못 쓴다 — 실측(스캔당 1914 점): 0.55 초에서 0.01 초로.

    법선의 부호는 임의다. 고유벡터의 부호가 정해지지 않기 때문인데, point-to-plane 비용은
    r^2 이고 야코비안도 같이 뒤집히므로 J^T J 와 J^T r 이 모두 그대로다 — 정규방정식이
    바뀌지 않아 상관없다.

    ★ 고정 개수 이웃이라 점이 성기면 이웃이 서로 다른 면을 넘나들어 법선이 무의미해진다.
      bundle_checks 의 표에 밀도별 실측이 있다.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.zeros_like(points)
    k = min(k, len(points) - 1)                     # 이웃이 모자라면 있는 만큼만

    tree = cKDTree(points)
    _, indices = tree.query(points, k=k + 1)        # 0 번은 자기 자신이라 버린다
    neighbours = points[indices[:, 1:]]             # (N, k, 3)

    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centred, centred) / k
    _, eigenvectors = np.linalg.eigh(covariance)    # 쌓인 3x3 을 한 번에 푼다
    return eigenvectors[:, :, 0]

def scan_to_world(scans, Rs, ts):
    normals = [None] * len(scans)
    points = [None] * len(scans)
    for index, scan in enumerate(scans):
        point = np.dot(scan.points, Rs[index].T) + ts[index]
        points[index] = point
        normals[index] = estimate_normals(point)
    
    return points, normals



def find_correspondences(world_points_per_scan, max_distance=0.2):
    def core(source, target):
        tree = cKDTree(target)
        distance, index = tree.query(source, k=1, distance_upper_bound=max_distance)
        keep = distance < max_distance
        return np.nonzero(keep)[0], index[keep]

    correspondences = []

    for i in range(len(world_points_per_scan) - 1):
        source_indices, target_indices = core(
            world_points_per_scan[i], 
            world_points_per_scan[i + 1])
        correspondences.append((i, i + 1, source_indices, target_indices))

    return correspondences

def build_residuals(points, normals, correspondences):
    """r = n_j . (p_i - q_j)   대응마다 하나씩, 한 벡터로 이어 붙인다.

    points 와 normals 는 **이미 현재 포즈로 world 에 올라간 것**이다 (scan_to_world 가 한다).
    여기서 포즈를 한 번 더 적용하면 안 된다 — 그러면 잔차는 두 번 변환된 점에서, 야코비안은
    한 번 변환된 점에서 평가되어 서로 다른 함수를 보게 된다 (실측: 평균 3.3 m 어긋남).
    """
    residuals = []

    for i, j, src_idx, dst_idx in correspondences:

        P = points[i][src_idx]          # source 점 (world)
        Q = points[j][dst_idx]          # 대응하는 target 점 (world)
        N = normals[j][dst_idx]         # target 면의 법선 (world)

        residuals.append(np.sum(N * (P - Q), axis=1))

    return np.concatenate(residuals)

def show_residuals(residuals):
    residuals = np.asarray(residuals)

    plt.figure()
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Point-to-Plane Residuals")
    plt.grid(True)
    plt.show()

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


def make_jacobian(scans, normals, corr, R, t):
    number_of_pose = len(R)
    Jacobians = list()
    
    for source_idx, dest_idx, src_idx, dst_idx in corr:
        X_local = scans[source_idx].points[src_idx]
        n = normals[dest_idx][dst_idx]
        # (N, 3)
        X_world = np.dot(X_local, R[source_idx].T) + t[source_idx]
        # (N, 3, 3)
        P_world_skew = skew(X_world)
        # (N, 3)
        J_rot = -np.einsum("ni,nij->nj", n, P_world_skew)
        # (N, 6)
        J_pose = np.hstack([J_rot, n])
        # (3, 6 * Frames)
        J = np.zeros((len(X_local), 6 * number_of_pose))
        J[:, 6*source_idx: 6*source_idx+6] = J_pose
        J[:, 6*dest_idx: 6*dest_idx+6] = -J_pose
        Jacobians.append(J)
    
    return Jacobians
        

def build_jacobian(scans, normals, correspondences, R, t):

    n_pose = len(R)

    J_list = []

    for i, j, src_idx, dst_idx in correspondences:

        X_local = scans[i].points[src_idx]
        n = normals[j][dst_idx]

        # (N, 3)
        X_world = np.dot(X_local, R[i].T) + t[i]

        # (N, 3, 3)
        S = skew(X_world)

        # (N, 3), J_{R} = −n^{T}[q]_{x}
        J_rot = -np.einsum("ni,nij->nj", n, S)

        # (N, 6)
        J_pose = np.hstack([J_rot, n])

        # (N, 6 * n_pose)
        J = np.zeros(
            (len(X_local), 6 * n_pose)
        )

        J[:, 6*i:6*i+6] = J_pose
        J[:, 6*j:6*j+6] = -J_pose

        J_list.append(J)

    return np.vstack(J_list)

def update_poses(R, t, delta):

    R_new = R.copy()
    t_new = t.copy()

    for i in range(1, len(R)):

        d = delta[6*(i-1):6*i]

        dtheta = d[:3]
        dt = d[3:]

        R_new[i] = so3_exp(dtheta) @ R[i]
        t_new[i] = so3_exp(dtheta) @ t[i] + dt

    return R_new, t_new

def bundle_checks():
    """원시 스캔 BA(do_bundle_adjustment 경로)의 야코비안과 정규방정식을 검증한다.

    주 단언은 '해석 야코비안 == 잔차의 중앙차분' 이다. 이것 하나가 두 종류의 버그를 다 잡는다:
      - 잔차와 야코비안을 다른 점에서 평가하면 (예: 이미 world 인 점에 포즈를 한 번 더 적용)
        두 값이 아예 안 맞는다
      - 야코비안이 블록 하나를 빠뜨리면 그 열만 0 과 어긋난다
    거기에 'H 가 특이하지 않은가' 를 따로 둔다 — 빠진 블록은 마지막 포즈를 어떤 잔차에도
    등장시키지 않아 H 의 행 6개를 통째로 0 으로 만들고, LM 감쇠는 대각에 비례하므로
    0 인 대각을 살리지 못한다. 실측으로 rank 18/24 였다.

    ★ 밀도를 낮게 잡으면 안 된다. estimate_normals 는 고정 개수 이웃을 쓰므로 점이 성기면
      이웃이 벽과 바닥을 넘나들고, 그 법선으로는 point-to-plane 비용이 줄어도 포즈가 참값에서
      멀어진다. 실측(포즈 5, 초기 오차 23 mm, 한 스텝):

        설정          점/스캔   법선 오차 p90   포즈 오차
        4ch x 60        240        48.0°      23 -> 99 mm
        6ch x 120       720        28.2°      23 -> 97 mm
        8ch x 180      1436        16.0°      23 -> 32 mm
        10ch x 300     2997        11.8°      23 -> 15 mm

      비용은 네 경우 모두 줄었다 — 솔버가 아니라 법선이 문제라는 뜻이다. 그래서 수학적
      단언(야코비안·계수·비용)과 종단 단언(포즈 오차)을 나눠 둔다.
    """
    spec = LidarSpec(channels=10, azimuth_steps=300, range_noise=0.005, dropout=0.0)
    objects = make_room_objects()
    Rs, ts, scans = make_lidar_data(objects, n_pose=5, spec=spec, seed=0)
    n_pose = len(Rs)
    R_init, t_init = perturb_poses(Rs, ts, seed=1, rot=0.01, trans=0.02)

    world_points, world_normals = scan_to_world(scans, R_init, t_init)
    correspondences = find_correspondences(world_points)

    # 법선은 점군과 함께 돈다. 포즈를 흔들 때마다 estimate_normals 를 다시 부르면 O(N^2)라
    # 못 쓴다 — 센서 프레임으로 한 번 내려두고 회전만 다시 건다. 고유벡터는 점군을 회전하면
    # 같이 회전하므로 같은 값이다.
    sensor_normals = [np.dot(world_normals[k], R_init[k]) for k in range(n_pose)]

    def residual_at(R, t):
        points = [to_world(R[k], t[k], scans[k].points) for k in range(n_pose)]
        normals = [np.dot(sensor_normals[k], R[k].T) for k in range(n_pose)]
        return build_residuals(points, normals, correspondences)

    J = build_jacobian(scans, world_normals, correspondences, R_init, t_init)
    base = residual_at(R_init, t_init)

    numeric = np.zeros((len(base), 6 * n_pose))
    h = 1e-6
    for k in range(n_pose):
        for axis in range(6):
            step = np.zeros(6)
            step[axis] = h
            R_plus, t_plus = R_init.copy(), t_init.copy()
            R_plus[k], t_plus[k] = apply_left_update(R_init[k], t_init[k], step)
            R_minus, t_minus = R_init.copy(), t_init.copy()
            R_minus[k], t_minus[k] = apply_left_update(R_init[k], t_init[k], -step)
            numeric[:, 6 * k + axis] = (residual_at(R_plus, t_plus)
                                        - residual_at(R_minus, t_minus)) / (2 * h)

    worst = np.abs(J - numeric).max() / max(1.0, np.abs(numeric).max())

    H = np.dot(J.T, J)
    free = np.arange(6, 6 * n_pose)
    H_free = H[np.ix_(free, free)]
    rank = int(np.linalg.matrix_rank(H_free))
    empty = [k for k in range(1, n_pose) if np.abs(J[:, 6 * k:6 * k + 6]).max() == 0.0]

    # 한 스텝을 실제로 밟아 본다. 대응은 고정 — 선형화를 재는 것이지 ICP 를 재는 게 아니다.
    gradient = np.dot(J.T, base)
    damped = H_free + 1e-6 * np.diag(np.diag(H_free))
    try:
        delta = np.linalg.solve(damped, -gradient[free])
        R_step, t_step = update_poses(R_init, t_init, delta)
        stepped = residual_at(R_step, t_step)
        cost_before = 0.5 * float((base ** 2).sum())
        cost_after = 0.5 * float((stepped ** 2).sum())
        before = float(pose_errors(R_init, t_init, Rs, ts)[0][1:].mean())
        after = float(pose_errors(R_step, t_step, Rs, ts)[0][1:].mean())
        solved = True
    except np.linalg.LinAlgError:
        cost_before = cost_after = before = after = float("nan")
        solved = False

    # KD-tree 이웃 탐색을 해석해로 못박는다 — 전수 비교에서 바꾼 뒤 조용히 틀리면 안 된다.
    # 평면 위 점들의 최소고유벡터는 그 평면의 법선과 정확히 같아야 한다.
    rng = np.random.default_rng(7)
    direction = np.array([0.3, -0.5, 0.8])
    direction /= np.linalg.norm(direction)
    basis = plane_basis(direction)
    patch = np.array([1.0, 2.0, 3.0]) + np.dot(rng.uniform(-0.5, 0.5, size=(400, 2)), basis)
    plane_worst = float(np.abs(np.abs(np.dot(estimate_normals(patch, k=12), direction)) - 1.0).max())

    # 대응은 max_distance 를 절대 넘지 않고, 상한을 좁히면 개수가 늘 수 없다
    cloud = rng.uniform(-1.0, 1.0, size=(300, 3))
    shifted = cloud + np.array([0.15, 0.0, 0.0])
    _, _, wide_source, wide_target = find_correspondences([cloud, shifted], max_distance=0.2)[0]
    gaps = np.linalg.norm(cloud[wide_source] - shifted[wide_target], axis=1)
    narrow = find_correspondences([cloud, shifted], max_distance=0.1)[0][2]
    bound_ok = (len(gaps) > 0 and gaps.max() < 0.2 and len(narrow) <= len(wide_source)
                and wide_source.dtype.kind == "i")

    # 이웃 선택 자체를 전수 비교 기준 구현으로 못박는다. 위의 평면 검사만으로는 '자기 자신을
    # 이웃에서 빼는가' 를 못 잡는다 — 자기 점도 평면 위라 법선이 그대로다 (뮤테이션 확인).
    # 그래서 평면이 아닌(납작하지만 휘어 있는) 구름에서 KD-tree 와 전수 비교를 맞춰 본다.
    def reference_normals(cloud, count):
        out = np.zeros_like(cloud)
        for index, point in enumerate(cloud):
            offset = cloud - point
            nearest = np.argsort(np.einsum("ni,ni->n", offset, offset))[1:count + 1]
            block = cloud[nearest] - cloud[nearest].mean(0)
            out[index] = np.linalg.eigh(np.dot(block.T, block) / count)[1][:, 0]
        return out

    sample = np.random.default_rng(11).normal(size=(200, 3)) * np.array([1.0, 1.0, 0.08])
    agreement = np.einsum("ni,ni->n", estimate_normals(sample, k=10), reference_normals(sample, 10))
    neighbour_worst = float(np.abs(np.abs(agreement) - 1.0).max())

    return [
        (f"BA 야코비안 vs 중앙차분 (최대 상대오차 {worst:.2e})", worst < 1e-5),
        (f"KD-tree 법선 == 평면의 참 법선 (최대오차 {plane_worst:.2e})", plane_worst < 1e-12),
        (f"KD-tree 이웃 == 전수 비교 이웃 (최대오차 {neighbour_worst:.2e})", neighbour_worst < 1e-9),
        (f"대응이 max_distance 안에 있다 (최대 {gaps.max():.3f} < 0.2, 좁히면 "
         f"{len(wide_source)} -> {len(narrow)})", bound_ok),
        (f"모든 자유 포즈가 잔차에 등장한다 (빈 블록 {empty})", not empty),
        (f"H_free 가 특이하지 않다 (rank {rank}/{len(free)})", rank == len(free)),
        (f"한 스텝이 비용을 줄인다 ({cost_before:.4f} -> {cost_after:.4f})",
         solved and cost_after < cost_before),
        (f"한 스텝이 포즈 오차를 줄인다 ({before * 1000:.1f} -> {after * 1000:.1f} mm)",
         solved and after < before),
    ]


def do_bundle_adjustment(args):
    spec = LidarSpec(channels=args.channels, 
                     azimuth_steps=args.azimuth_steps,
                     elevation_min_deg=args.elevation[0], 
                     elevation_max_deg=args.elevation[1],
                     max_range=args.max_range, 
                     range_noise=args.noise, 
                     dropout=args.dropout)
    objects = make_room_objects(props=not args.no_props)
    Rs, ts, scans = make_lidar_data(objects, 
                                    n_pose=args.n_pose, 
                                    spec=spec,
                                    seed=args.seed, 
                                    motion=args.motion)
    
    # 1. 초기 Pose
    R_init, t_init = perturb_poses(Rs, ts)

    # 2. World point + normal
    world_points_per_scan, normals = scan_to_world(
        scans,
        R_init,
        t_init
    )

    # 3. Correspondence
    correspondences = find_correspondences(
        world_points_per_scan
    )

    # 4. Residual
    residuals = build_residuals(
        world_points_per_scan,
        normals,
        correspondences
    )
    show_residuals(residuals)
    
    make_jacobian(
        scans,
        normals,
        correspondences,
        R_init,
        t_init)

    # 5. Jacobian
    J = build_jacobian(
        scans,
        normals,
        correspondences,
        R_init,
        t_init
    )

    # 6. Normal Equation
    # np.dot 이지 @ 가 아니다 — 이 파일 위쪽 build_normal_equation·to_world 와 같은 이유로,
    # macOS 의 numpy 2.x 는 큰 행렬곱에 거짓 "divide by zero" 경고를 낸다 (값은 유한).
    H = np.dot(J.T, J)
    g = np.dot(J.T, residuals)

    # 7. Fix pose 0
    free = np.arange(
        6,
        6 * len(R_init)
    )

    H_free = H[np.ix_(free, free)]
    g_free = g[free]

    # 8. LM
    lam = 1e-3

    H_lm = (
        H_free
        + lam * np.diag(np.diag(H_free))
    )

    delta_free = np.linalg.solve(
        H_lm,
        -g_free
    )

    # 9. Pose update
    R_new, t_new = update_poses(
        R_init,
        t_init,
        delta_free
    )
    
    ################
    world_points, normals = scan_to_world(
        scans,
        R_new,
        t_new
    )

    # 3. Correspondence
    correspondences = find_correspondences(
        world_points
    )
    

    # 4. Residual
    residuals = build_residuals(
        world_points,
        normals,
        correspondences
    )
    
    show_residuals(residuals)    
    
    show_point_cloud(world_points)

if __name__ == "__main__":
    if ARGS.selftest:
        sys.exit(selftest(ARGS))
    do_bundle_adjustment(ARGS)
