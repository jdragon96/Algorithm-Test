"""로드맵 노트북들이 공유하는 시험 문제.

노트북은 알고리즘만 담는다. 문제를 만드는 잡일은 전부 여기로 뺀다.
모든 생성기는 **참값을 함께 돌려준다** — 참값이 없으면 "수렴했다"까지만 말할 수 있고
"맞게 수렴했다"는 말할 수 없기 때문이다.

수록 문제
---------
01  make_conditioned_lsq   조건수를 지정한 선형 최소제곱
02  rosenbrock             Rosenbrock 골짜기 (최소점 (1,1))
03  curve_*                y = a e^{bt} + c 비선형 곡선 맞춤
04  add_outliers           위 곡선에 outlier 섞기
05  make_toy_ba            pose-landmark BA (arrowhead Hessian)
06  make_registration_data 알려진 (R, t) 로 만든 점군 쌍

섭동 규약
---------
회전이 끼는 문제(05, 06)는 **왼쪽(world) 섭동**을 기본으로 쓴다.

    R <- exp([dtheta]_x) R,    t <- exp([dtheta]_x) t + dt

`SLAM/LiDAR/BundleAdjustment.ipynb` 와 같은 규약이다. 오른쪽 섭동과 섞으면 Jacobian 의
회전 블록이 조용히 틀리므로, 노트북에서도 이 한 줄을 그대로 옮겨 적고 시작한다.
"""

import numpy as np

__all__ = [
    "make_conditioned_lsq",
    "rosenbrock", "rosenbrock_grad", "rosenbrock_hess",
    "curve_model", "curve_residual", "curve_jacobian", "make_curve_data", "add_outliers",
    "curve_model_redundant", "curve_residual_redundant", "curve_jacobian_redundant",
    "skew", "so3_exp", "so3_log",
    "make_toy_ba", "ba_residual_and_jacobian",
    "make_registration_data",
]


# ============================================================ 01. 조건수

def make_conditioned_lsq(m=60, n=8, cond=1e6, noise=0.0, seed=0):
    """조건수가 정확히 `cond` 인 J 와, 참값 x_true 를 아는 최소제곱 문제.

    SVD 로 거꾸로 만든다. 특이값을 1 부터 1/cond 까지 로그 등간격으로 깔면
    kappa(J) = cond 가 설계값 그대로 나온다.

    반환: J (m, n), b (m,), x_true (n,)
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(m, n)))          # (m, n) 열직교
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))          # (n, n) 직교
    s = np.logspace(0, -np.log10(cond), n)                # (n,) 1 .. 1/cond
    J = U @ np.diag(s) @ V.T

    x_true = rng.normal(size=n)
    b = J @ x_true
    if noise > 0:
        b = b + rng.normal(scale=noise, size=m)
    return J, b, x_true


# ============================================================ 02. Rosenbrock

def rosenbrock(x, a=1.0, b=100.0):
    """f(x) = (a - x0)^2 + b (x1 - x0^2)^2. 최소점은 (a, a^2), 여기서는 (1, 1)."""
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x, a=1.0, b=100.0):
    return np.array([
        -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0] ** 2),
        2.0 * b * (x[1] - x[0] ** 2),
    ])


def rosenbrock_hess(x, a=1.0, b=100.0):
    return np.array([
        [2.0 - 4.0 * b * (x[1] - 3.0 * x[0] ** 2), -4.0 * b * x[0]],
        [-4.0 * b * x[0], 2.0 * b],
    ])


# ============================================================ 03-04. 곡선 맞춤

def curve_model(theta, t):
    """y = a * exp(b t) + c.  theta = (a, b, c)

    b 가 지수 안에 있어서 강하게 비선형이다. b 초기값을 조금만 틀리게 잡아도
    Gauss-Newton 이 발산하기 때문에 LM 의 필요성을 보여주기 좋다.
    """
    a, b, c = theta
    return a * np.exp(b * t) + c


def curve_residual(theta, t, y):
    """r = model - y.  (N,)"""
    return curve_model(theta, t) - y


def curve_jacobian(theta, t):
    """dr/dtheta.  (N, 3)

        dr/da = exp(bt),  dr/db = a t exp(bt),  dr/dc = 1
    """
    a, b, _ = theta
    e = np.exp(b * t)                                     # (N,)
    return np.stack([e, a * t * e, np.ones_like(t)], axis=1)


def curve_model_redundant(theta, t):
    """y = a e^{bt} + c + d.  theta = (a, b, c, d)

    c 와 d 는 합으로만 나타나므로 **개별로는 결정되지 않는다**. 그래서 J 의 마지막 두 열이
    똑같아지고 rank 가 하나 모자란다 — 03 에서 GN 이 죽고 LM 이 사는 장면을 만들기 위한 것이다.
    실제 문제에서도 과매개화(over-parameterization)로 이런 일이 흔히 생긴다.
    """
    a, b, c, d = theta
    return a * np.exp(b * t) + c + d


def curve_residual_redundant(theta, t, y):
    return curve_model_redundant(theta, t) - y


def curve_jacobian_redundant(theta, t):
    """(N, 4). 마지막 두 열이 모두 1 이라 rank 는 최대 3 이다."""
    a, b, _, _ = theta
    e = np.exp(b * t)
    one = np.ones_like(t)
    return np.stack([e, a * t * e, one, one], axis=1)


def make_curve_data(theta_true=(2.5, -0.7, 0.5), n=60, t_max=5.0, noise=0.03, seed=0):
    """참값 theta 로 만든 (t, y). 반환: t (N,), y (N,), theta_true (3,)"""
    rng = np.random.default_rng(seed)
    theta_true = np.asarray(theta_true, dtype=float)
    t = np.linspace(0.0, t_max, n)
    y = curve_model(theta_true, t) + rng.normal(scale=noise, size=n)
    return t, y, theta_true


def add_outliers(y, fraction=0.1, magnitude=3.0, seed=1):
    """관측의 일부를 크게 망가뜨린다. 반환: 오염된 y, outlier 마스크 (N,) bool

    correspondence 오매칭을 흉내낸 것이라, 노이즈가 아니라 **다른 분포**에서 온 값이다.
    """
    rng = np.random.default_rng(seed)
    y = y.copy()
    n_bad = int(round(len(y) * fraction))
    index = rng.choice(len(y), size=n_bad, replace=False)
    y[index] += rng.choice([-1.0, 1.0], size=n_bad) * magnitude
    mask = np.zeros(len(y), dtype=bool)
    mask[index] = True
    return y, mask


# ============================================================ SO(3)

def skew(v):
    """(..., 3) -> (..., 3, 3) 반대칭 행렬. a x b = skew(a) @ b"""
    v = np.asarray(v, dtype=float)
    zero = np.zeros(v.shape[:-1])
    return np.stack([
        np.stack([zero, -v[..., 2], v[..., 1]], axis=-1),
        np.stack([v[..., 2], zero, -v[..., 0]], axis=-1),
        np.stack([-v[..., 1], v[..., 0], zero], axis=-1),
    ], axis=-2)


def so3_exp(phi):
    """Rodrigues. (3,) -> (3, 3)"""
    phi = np.asarray(phi, dtype=float)
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        return np.eye(3) + skew(phi)                      # 1차 근사로 충분하다
    axis = phi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def so3_log(R):
    """(3, 3) -> (3,)"""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-12:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return angle / (2.0 * np.sin(angle)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


# ============================================================ 05. Toy BA

def make_toy_ba(n_pose=6, n_land=40, observed=0.6, noise=0.01, seed=0):
    """pose 가 landmark 를 관측하는 작은 BA.

    관측 모델은 landmark 를 pose 좌표계로 옮긴 위치 그 자체다.

        z_ij = R_i^T (X_j - t_i) + noise

    projection 을 쓰지 않는 이유는, 05 의 주제가 **Hessian 의 희소 구조**이지
    카메라 모델이 아니기 때문이다. 구조는 실제 BA 와 똑같이 arrowhead 가 된다.

    반환: dict(Rs, ts, Xs, obs, Rs_init, ts_init, Xs_init)
      obs: (pose_index, land_index, z (3,)) 리스트
    """
    rng = np.random.default_rng(seed)

    angles = np.linspace(0.0, 1.6 * np.pi, n_pose)
    Rs = np.array([so3_exp(np.array([0.1 * np.sin(a), 0.1 * np.cos(a), a])) for a in angles])
    ts = np.array([[3.0 * np.cos(a), 3.0 * np.sin(a), 0.4 * np.sin(2 * a)] for a in angles])
    Xs = rng.uniform(-1.5, 1.5, size=(n_land, 3))

    def observe(i, j):
        z = Rs[i].T @ (Xs[j] - ts[i]) + rng.normal(scale=noise, size=3)
        return (i, j, z)

    seen = [[] for _ in range(n_land)]
    obs = []
    for i in range(n_pose):
        for j in range(n_land):
            if rng.random() > observed:                   # 모든 pose 가 모든 landmark 를 보지는 않는다
                continue
            obs.append(observe(i, j))
            seen[j].append(i)

    # 관측이 min_obs 회 미만인 landmark 는 H_ll 블록이 특이해진다. 변수로 남길 거면 관측을 채운다.
    min_obs = 2
    for j in range(n_land):
        missing = [i for i in range(n_pose) if i not in seen[j]]
        rng.shuffle(missing)
        for i in missing[:max(0, min_obs - len(seen[j]))]:
            obs.append(observe(i, j))
            seen[j].append(i)
    obs.sort(key=lambda o: (o[0], o[1]))

    # 흔든 초기값. pose 0 은 게이지라 그대로 둔다.
    Rs_init, ts_init = Rs.copy(), ts.copy()
    for i in range(1, n_pose):
        Rs_init[i] = so3_exp(rng.normal(scale=0.03, size=3)) @ Rs[i]
        ts_init[i] = ts[i] + rng.normal(scale=0.08, size=3)
    Xs_init = Xs + rng.normal(scale=0.08, size=Xs.shape)

    return dict(Rs=Rs, ts=ts, Xs=Xs, obs=obs,
                Rs_init=Rs_init, ts_init=ts_init, Xs_init=Xs_init)


def ba_residual_and_jacobian(Rs, ts, Xs, obs):
    """arrowhead 구조를 그대로 가진 H, g 를 만든다.

    잔차는  r_ij = R_i^T (X_j - t_i) - z_ij   (3,)

    **왼쪽 섭동**은 R 과 t 를 함께 돌리므로, 전개하면 t 항이 상쇄되고 [X]_x 만 남는다:
        R' = exp([dtheta]_x) R,   t' = exp([dtheta]_x) t + dt
        R'^T (X - t') ~= R^T[ (X - t) + [X]_x dtheta - dt ]
        => d r / d dtheta =  R_i^T [X_j]_x      ([X_j - t_i]_x 가 아니다!)
        => d r / d dt     = -R_i^T
        => d r / d X_j    =  R_i^T

    [X - t]_x 로 쓰는 실수가 흔하고 수식이 멀쩡해 보인다. 중앙차분과 대조하면 바로 걸린다.

    반환: Hpp (6P,6P), Hpl (6P,3L), Hll (3L,3L), bp (6P,), bl (3L,), cost
    """
    n_pose, n_land = len(Rs), len(Xs)
    P, L = 6 * n_pose, 3 * n_land
    Hpp = np.zeros((P, P))
    Hpl = np.zeros((P, L))
    Hll = np.zeros((L, L))
    bp = np.zeros(P)
    bl = np.zeros(L)
    cost = 0.0

    for i, j, z in obs:
        d = Xs[j] - ts[i]                                 # (3,)
        r = Rs[i].T @ d - z                               # (3,)
        cost += float(r @ r)

        Ji = np.hstack([Rs[i].T @ skew(Xs[j]), -Rs[i].T])  # (3, 6) pose 블록
        Jj = Rs[i].T                                      # (3, 3) landmark 블록

        pi, lj = slice(6 * i, 6 * i + 6), slice(3 * j, 3 * j + 3)
        Hpp[pi, pi] += Ji.T @ Ji
        Hpl[pi, lj] += Ji.T @ Jj
        Hll[lj, lj] += Jj.T @ Jj
        bp[pi] -= Ji.T @ r                                # b = -g
        bl[lj] -= Jj.T @ r

    return Hpp, Hpl, Hll, bp, bl, 0.5 * cost


# ============================================================ 06. 점군 정합

def make_registration_data(n_points=200, noise=0.005, seed=0):
    """알려진 (R, t) 로 옮긴 점군 쌍.

        q_i = R p_i + t + noise

    반환: P (N,3), Q (N,3), R_true (3,3), t_true (3,), R_init (3,3), t_init (3,)
    """
    rng = np.random.default_rng(seed)
    P = rng.uniform(-1.0, 1.0, size=(n_points, 3))

    R_true = so3_exp(np.array([0.25, -0.15, 0.35]))
    t_true = np.array([0.4, -0.2, 0.3])
    Q = P @ R_true.T + t_true + rng.normal(scale=noise, size=P.shape)

    R_init = so3_exp(rng.normal(scale=0.25, size=3)) @ R_true
    t_init = t_true + rng.normal(scale=0.3, size=3)
    return P, Q, R_true, t_true, R_init, t_init
