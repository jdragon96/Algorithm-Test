# 개요

- 비선형 최소제곱 $\min_{x} F(x)$ 를 푸는 알고리즘들을, **왜 그 순서로 배워야 하는지**와 함께 정리한 학습 로드맵이다.
- 기준점은 이미 구현한 [LiDAR BA](../SLAM/LiDAR/BundleAdjustment.ipynb)(Gauss-Newton + LM)와 [Optical Flow](../SLAM/Visual/OpticalFlow.ipynb)(같은 $H\delta = -g$). 뒤로는 그 밑에 깔린 기초를 메우고, 앞으로는 SLAM이 실제로 요구하는 것으로 간다.

---
# 1. 로드맵의 척추

거의 모든 비선형 최적화 알고리즘은 현재 위치 $x$ 에서 같은 두 가지를 정한다.

- `방향`: 어느 쪽으로 갈 것인가
- `보폭`: 그 방향을 얼마나 믿을 것인가

목적함수는 공통적으로 잔차의 제곱합이다.

$$ F(x) = \frac{1}{2}\sum_{i} \rho\!\left(\|r_{i}(x)\|^{2}\right) $$

- $r_{i}(x)$: $i$ 번째 관측의 residual
- $\rho$: robust loss. 당장은 $\rho(s) = s$ 로 두고 읽으면 된다

$x$ 근방에서 2차까지 전개하면 국소 모델 $m(\delta)$ 를 얻는다.

$$ m(\delta) = F(x) + g^{T}\delta + \frac{1}{2}\delta^{T}H\delta $$

- $g = \nabla F$, $H$: 2차항 행렬 (알고리즘마다 다르게 고른다)

$m$ 을 최소화하는 $\delta$ 는 $\nabla m = 0$ 에서 나오고, 이것이 로드맵 전체를 관통하는 한 줄이다.

$$
\boxed{
H\delta = -g
}
$$

즉,

> 모든 알고리즘은 $H$ 를 무엇으로 고르고 그 해를 얼마나 믿을지만 다르다

이 표가 로드맵의 목차이기도 하다.

| 알고리즘 | $H$ 를 무엇으로 두는가 | 보폭을 어떻게 정하는가 | 단계 |
|---|---|---|---|
| Gradient Descent | $I$ | line search | 2 |
| Newton | $\nabla^{2}F$ | 그대로 (전체 스텝) | 2 |
| Quasi-Newton (BFGS) | 곡률 누적 근사 $B_k$ | Wolfe line search | 2 |
| Gauss-Newton | $J^{T}J$ | 그대로 | 3 |
| Levenberg-Marquardt | $J^{T}J + \lambda D$ | $\lambda$ | 3 |
| Trust Region / Dogleg | $J^{T}J$ | 신뢰 반경 $\Delta$ | 3 |

---
# 2. Stage 1 — 선형대수 바닥

$H\delta = -g$ 는 **선형** 시스템이다. 비선형 최적화의 반복 한 번은 결국 이 선형 문제를 푸는 일이고, 여기서 정밀도를 잃으면 위에서 무엇을 하든 소용이 없다.

## 2.1. Normal Equation을 직접 풀면 안 되는 이유

최소제곱 $\min_{\delta}\|J\delta + r\|^{2}$ 의 해는 $J^{T}J\delta = -J^{T}r$ 이지만, $J^{T}J$ 를 만드는 순간 조건수가 제곱된다.

$$ \kappa(J^{T}J) = \kappa(J)^{2} $$

$\kappa(J) = 10^{6}$ 이면 $J^{T}J$ 는 배정밀도의 유효자릿수를 전부 먹는다. 그래서 수치적으로는 $J$ 를 직접 QR 분해하는 쪽이 옳다. 반대로 SLAM처럼 $J$ 가 크고 희소하면 $H$ 를 만들어 Cholesky 하는 쪽이 압도적으로 싸다 — **정확도와 비용의 맞교환**이고, 어느 쪽을 택했는지 아는 것이 요점이다.

- `Cholesky` $H = LL^{T}$: 가장 싸다. $H$ 가 양정부호일 때만
- `QR` $J = QR$: $H$ 를 만들지 않는다. 조건수에 강하다
- `SVD` $J = U\Sigma V^{T}$: 가장 비싸지만 rank 결핍을 직접 보여준다

## 2.2. 체크포인트

- $J^{T}J$ 를 만들어 푼 답과 `lstsq`(QR)로 푼 답이 $\kappa(J)$ 를 키울수록 얼마나 벌어지는가
- 명시적 `inv(H) @ -g` 가 `solve(H, -g)` 보다 왜 느리고 부정확한가

> 목표 노트북: `01_least_squares_and_conditioning.ipynb`

---
# 3. Stage 2 — Line Search 계열

$H$ 를 고르는 가장 단순한 선택들부터 본다. 이 계열은 방향을 정한 뒤 **보폭을 1차원 탐색으로** 따로 정한다.

## 3.1. Gradient Descent와 조건수

$H = I$ 로 두면 $\delta = -g$, 즉 최급강하 방향이다. 가장 단순하지만 수렴 속도가 $\kappa(H)$ 에 그대로 지배된다. Rosenbrock 함수의 좁은 골짜기에서 지그재그가 나오는 것이 이 때문이고, **이 실패를 직접 본 사람만 Newton이 왜 필요한지 안다.**

## 3.2. Armijo / Wolfe 조건

보폭 $\alpha$ 를 아무렇게나 정하면 발산한다. 최소한의 안전장치가 `Armijo 조건`(충분한 감소)이다.

$$ F(x + \alpha p) \le F(x) + c_{1}\alpha\, g^{T}p, \qquad c_{1} \approx 10^{-4} $$

여기에 보폭이 너무 작아지는 것을 막는 `curvature 조건`을 더하면 Wolfe 조건이 된다.

$$ \nabla F(x + \alpha p)^{T} p \ge c_{2}\, g^{T}p, \qquad c_{2} \approx 0.9 $$

Quasi-Newton이 곡률을 제대로 누적하려면 이 두 번째 조건이 반드시 필요하다. 생략하면 $B_k$ 가 양정부호를 잃는다.

## 3.3. Newton

$H = \nabla^{2}F$ 를 그대로 쓰면 국소적으로 2차 수렴한다. 대신 Hessian을 만들어야 하고, $x$ 가 최소점에서 멀면 $\nabla^{2}F$ 가 양정부호가 아니라 **$\delta$ 가 오르막을 가리킬 수 있다.** Stage 3의 damping이 정확히 이 문제를 푼다.

## 3.4. Quasi-Newton (BFGS, L-BFGS)

Hessian을 만들지 않고, 연속한 스텝에서 관측한 기울기 변화로 곡률을 누적한다.

$$ s_{k} = x_{k+1} - x_{k}, \qquad y_{k} = g_{k+1} - g_{k} $$

$$ B_{k+1} = B_{k} + \frac{y_{k}y_{k}^{T}}{y_{k}^{T}s_{k}} - \frac{B_{k}s_{k}s_{k}^{T}B_{k}}{s_{k}^{T}B_{k}s_{k}} $$

`L-BFGS`는 $B_k$ 를 저장하지 않고 최근 $m$ 개의 $(s,y)$ 쌍만 들고 있다가 곱셈을 재귀로 푼다. 변수가 수백만 개인 문제에서 사실상 기본값이고, SLAM보다는 deep learning 쪽에서 더 자주 만난다.

## 3.5. 체크포인트

- Rosenbrock에서 GD / Newton / BFGS의 반복 횟수 비교
- $c_2$ 를 끄면 BFGS가 어떻게 망가지는가

> 목표 노트북: `02_line_search_methods.ipynb`

---
# 4. Stage 3 — Trust Region 계열

여기가 **이미 구현한 곳**이다. 비선형 최소제곱에 특화된 $H$ 선택과, 보폭을 line search 대신 **모델을 믿는 범위**로 다루는 방식이다.

## 4.1. Gauss-Newton: 2차항을 버릴 수 있는 이유

$F = \frac{1}{2}\|r\|^{2}$ 의 진짜 Hessian은 다음과 같다.

$$ \nabla^{2}F = J^{T}J + \sum_{i} r_{i}\nabla^{2}r_{i} $$

두 번째 항은 residual $r_i$ 에 비례한다. 해 근처에서 $r_i$ 가 작으면 무시해도 되고, 그러면 **1차 미분만으로 2차 정보를 얻는다**. 이것이 Gauss-Newton의 전부다.

$$
\boxed{
H_{GN} = J^{T}J, \qquad g = J^{T}r
}
$$

거꾸로, residual이 큰 문제(outlier가 섞였거나 모델이 틀렸을 때)에서 GN이 잘 안 되는 것도 같은 이유다. 버린 항이 작지 않기 때문이다.

## 4.2. Levenberg-Marquardt: $\lambda$ 의 두 얼굴

$$ (J^{T}J + \lambda D)\,\delta = -J^{T}r, \qquad D = \operatorname{diag}(J^{T}J) $$

- $\lambda \to 0$: Gauss-Newton. 빠르지만 모델을 전부 믿는다
- $\lambda \to \infty$: $\delta \to -g/\lambda$, 즉 아주 작은 gradient descent. 느리지만 안전하다

그리고 $\lambda$ 는 **신뢰 반경 $\Delta$ 와 일대일 대응**한다. damping을 거는 것과 "이 반경 안에서만 모델을 믿겠다"는 것이 같은 말이라는 점이 이 단계의 핵심 통찰이다.

## 4.3. $\lambda$ 를 자동으로 조절하기 — 다음 과제

지금 BA 노트북은 `lam = 1e-3` 고정이다. 제대로 된 LM은 매 반복마다 모델이 실제로 맞았는지를 `gain ratio` 로 재고 $\lambda$ 를 조절한다.

$$
\boxed{
\gamma = \frac{F(x) - F(x + \delta)}{m(0) - m(\delta)}
}
$$

- $\gamma$: gain ratio. 실제 감소량 ÷ 모델이 약속한 감소량 (robust loss $\rho$ 와 헷갈리지 않게 기호를 분리했다)

분모는 전개하면 $\delta$ 와 $g$ 만으로 떨어져 따로 계산할 필요가 없다.

$$ m(0) - m(\delta) = \frac{1}{2}\delta^{T}\!\left(\lambda D\delta - g\right) $$

- $\gamma > 0$: 스텝을 받아들이고 $\lambda \leftarrow \lambda/3$ (모델을 더 믿는다)
- $\gamma \le 0$: 스텝을 버리고 $\lambda \leftarrow 2\lambda$ 로 다시 푼다

**스텝을 거부할 수 있다는 것**이 고정 $\lambda$ 와의 결정적 차이다. 지금 노트북은 나빠지는 스텝도 그대로 받는다.

## 4.4. Dogleg

Trust region을 $\lambda$ 대신 반경 $\Delta$ 로 직접 다룬다. Cauchy point(gradient 방향 최소)와 Gauss-Newton 점을 잇는 선분에서 $\|\delta\| = \Delta$ 인 지점을 고른다. LM이 매 $\lambda$ 마다 선형 시스템을 다시 푸는 것과 달리 **한 번만 풀면 되므로**, 큰 문제에서 유리하다.

## 4.5. 체크포인트

- gain ratio를 붙인 LM과 고정 $\lambda$ LM의 수렴 곡선 비교 (BA 노트북에 그대로 이식 가능)
- $J$ 를 rank 결핍으로 만들면 GN은 죽고 LM은 사는가
- LM의 비용이 **한 번도 증가하지 않는가** (step rejection이 보장하는 성질)

한 가지는 미리 못 박아 둔다. **LM은 전역최소를 찾아주지 않는다.** damping이 보장하는 것은
"터지지 않는다"와 "나빠지지 않는다"까지고, 어느 국소최소로 갈지는 초기값이 정한다. 실제로
나쁜 초기값에서는 스텝을 거부하는 LM이 먼저 정체하고, 거칠게 움직이는 GN이 더 좋은 골짜기로
넘어가 버리는 경우도 있다 — `03` 노트북에 그 반례를 그대로 남겨 두었다.

> 목표 노트북: `03_gauss_newton_lm_trust_region.ipynb`

---
# 5. Stage 4 — Robust Estimation

실제 데이터에는 correspondence 오매칭이 섞인다. 제곱 손실은 잔차에 **제곱으로** 가중하므로 outlier 하나가 전체 해를 끌고 간다. Stage 1의 $\rho$ 를 이제 진짜로 쓴다.

## 5.1. M-estimator와 영향 함수

$$ F(x) = \frac{1}{2}\sum_{i}\rho\!\left(r_{i}^{2}\right) $$

| $\rho$ | 성질 |
|---|---|
| $L_2$ | $\rho(s) = s$. outlier에 무방비 |
| Huber | 임계 $k$ 까지는 $L_2$, 그 밖은 $L_1$. 볼록해서 국소최소가 하나 |
| Cauchy | 큰 잔차의 영향을 계속 줄인다. 비볼록 |
| Tukey | 임계 밖을 완전히 0으로. 가장 공격적, 초기값에 민감 |

## 5.2. IRLS — 기존 Gauss-Newton에 끼워 넣기

$\nabla F = \sum_i \rho'(r_i^{2})\, r_{i}J_{i}^{T}$ 이므로, **가중치를 붙인 normal equation**과 형태가 같다.

$$ J^{T}WJ\,\delta = -J^{T}Wr, \qquad W = \operatorname{diag}\!\left(\rho'(r_{i}^{2})\right) $$

Huber라면 $\rho'(r^{2}) = \min(1,\ k/|r|)$ 이다. 즉 **매 반복 잔차로 가중치를 다시 계산해 꽂아주기만 하면** 기존 GN/LM 코드가 그대로 robust해진다. 구현 비용이 거의 없는 것이 이 방법의 강점이다.

## 5.3. RANSAC과의 분업

같은 문제를 푸는 것 같지만 역할이 다르다. RANSAC은 **초기값이 없을 때** 모델을 세우고, M-estimator는 **괜찮은 초기값에서** 남은 outlier를 눌러 정밀도를 올린다. 보통 RANSAC → LM+Huber 순으로 이어 붙인다.

> 목표 노트북: `04_robust_loss_and_irls.ipynb` — BA에 outlier correspondence를 일부러 섞고 $L_2$ vs Huber 비교

---
# 6. Stage 5 — 희소성과 구조

SLAM에서 실제 병목은 여기다. 지금 BA 노트북의 $H$ 는 $6N \times 6N$ dense이고, frame이 늘면 $O(N^{3})$ 으로 죽는다. 실제 $H$ 는 거의 비어 있다.

## 6.1. BA Hessian의 구조

pose와 landmark를 나눠 정렬하면 `arrowhead` 모양이 된다.

```txt
        pose (6N)        landmark (3M)
      +-------------+-------------------+
pose  |   H_pp      |      H_pl         |   H_pp : 블록 대각 + 인접 pose
      |  (sparse)   |    (sparse)       |   H_pl : 관측이 있는 쌍만
      +-------------+-------------------+
land  |   H_pl^T    |      H_ll         |   H_ll : 완전 블록 대각 (3x3)
      |             |  (block diagonal) |         landmark끼리는 연결 없음
      +-------------+-------------------+

H_ll 이 블록 대각인 것은 "두 landmark가 직접 만나는 관측이 없다"는 사실의 결과다.
그래서 H_ll^-1 이 3x3 역행렬 M 개로 분해되고, Schur complement가 싸진다.
```

## 6.2. Schur Complement

$\delta_{l}$ 을 먼저 소거하면 pose만의 훨씬 작은 시스템이 남는다.

$$
\boxed{
\left(H_{pp} - H_{pl}H_{ll}^{-1}H_{pl}^{T}\right)\delta_{p} = b_{p} - H_{pl}H_{ll}^{-1}b_{l}
}
$$

- $b = -g$ 를 pose/landmark 블록으로 나눈 것

$H_{ll}^{-1}$ 이 $3\times3$ 블록들의 역행렬이라 사실상 공짜이므로, $3M$ 개의 변수를 이 한 번의 대입으로 지운다. 푼 뒤 $\delta_{l} = H_{ll}^{-1}(b_{l} - H_{pl}^{T}\delta_{p})$ 로 되돌린다. 이것이 g2o·Ceres·GTSAM이 전부 하는 일이다.

## 6.3. 그 다음

- `Ordering`: Cholesky의 fill-in을 줄이는 변수 순서 (COLAMD/AMD). 순서만 바꿔 몇 배가 빨라진다
- `Marginalization`: 오래된 frame을 지우되 정보는 남긴다 — sliding window의 기반. 지우면 dense해진다는 대가가 있다
- `Incremental`: 매번 전부 다시 풀지 않고 바뀐 부분만 (iSAM2, Bayes tree)

> 목표 노트북: `05_sparse_ba_schur.ipynb` — dense와 Schur의 $N$ 에 따른 실행시간 곡선

---
# 7. Stage 6 — 다양체 위의 최적화

회전은 벡터공간이 아니다. $R + \delta R$ 은 회전행렬이 아니므로, 지금까지의 $x \leftarrow x + \delta$ 가 성립하지 않는다. BA 노트북이 이미 쓰고 있는 부분이기도 하다.

## 7.1. Retraction

접평면에서 얻은 $\delta$ 를 다양체로 되돌리는 연산으로 덧셈을 대체한다.

$$ R \leftarrow R \exp\!\left([\delta\theta]_{\times}\right), \qquad t \leftarrow t + \delta t $$

## 7.2. 좌/우 섭동 규약을 못 박아라

$$ \exp([\delta\theta]_{\times})R \quad (\text{left}) \qquad\text{vs}\qquad R\exp([\delta\theta]_{\times}) \quad (\text{right}) $$

둘 중 무엇을 쓰든 상관없지만 **섞으면 모든 Jacobian의 회전 블록이 조용히 틀린다.** 결과 수식이 멀쩡해 보여서 더 위험하다. 노트북 맨 위에 한 줄로 적어두고 끝까지 그것만 쓴다.

## 7.3. 체크포인트

- 해석적 Jacobian과 수치 미분의 차이가 $10^{-6}$ 수준인가 (이 한 줄이 규약 실수를 거의 다 잡는다)
- left/right를 일부러 섞으면 수렴 **속도**가 어떻게 달라지는가 — 터지지 않는다는 것이 함정이다

> 목표 노트북: `06_manifold_optimization.ipynb`

---
# 8. Stage 7 — 경계 밖

여기까지가 SLAM의 주류 경로다. 이 다음은 필요할 때 찾아가면 된다.

- `제약 최적화`: Lagrange/KKT, penalty, Augmented Lagrangian — 등식·부등식 제약이 붙을 때
- `Convex relaxation`: rotation averaging의 SDP 완화. 초기값 없이 전역해에 접근하는 드문 경우
- `확률적 최적화`: SGD, Adam. 데이터가 너무 커서 전체 $J$ 를 못 만들 때. Hessian을 포기하는 대신 샘플을 늘린다

---
# 9. 순서 요약

```txt
Stage 1  선형대수 바닥        cond(J^T J) = cond(J)^2, Cholesky/QR/SVD
   |                          "반복 한 번 = 선형 시스템 한 번"
Stage 2  Line search          GD -> Armijo/Wolfe -> Newton -> BFGS
   |                          방향과 보폭을 나눠 생각하기
Stage 3  Trust region     ★   GN -> LM -> gain ratio -> Dogleg
   |                          ★ 현재 위치. 다음 과제는 gain ratio
Stage 4  Robust               rho, IRLS, RANSAC과의 분업
   |                          실데이터로 가는 최소 조건
Stage 5  Sparsity             Schur, ordering, marginalization
   |                          SLAM이 커질 때의 유일한 탈출구
Stage 6  Manifold             retraction, left/right 규약
   |                          회전이 끼는 순간 필수
Stage 7  그 밖                제약 / convex / stochastic
```

각 단계는 **앞 단계의 실패를 고치는 방식으로** 이어진다. GD의 지그재그 → Newton, Newton의 비양정부호 → LM의 damping, GN의 outlier 취약성 → robust loss, dense $H$ 의 $O(N^3)$ → Schur. 순서를 건너뛰면 "왜 이게 필요한가"가 사라진다.

---
# 10. 구현 계획

노트북 규약은 [BundleAdjustment](../SLAM/LiDAR/BundleAdjustment.ipynb)·[OpticalFlow](../SLAM/Visual/OpticalFlow.ipynb) 와 같다 — 유도 → 참값을 아는 합성 데이터 → 박스 수식당 코드 셀 하나 → 메인 루프 → 검증.

전부 작성해 실행까지 마쳤다. 아래 숫자는 노트북을 그대로 돌려 나온 값이다.

| 노트북 | 무엇을 증명하는가 | 실측 | 선행 |
|---|---|---|---|
| `01_least_squares_and_conditioning` | 오차가 normal equation은 $\kappa^2$, QR은 $\kappa$ 로 자란다 | 로그-로그 기울기 **1.99 / 0.91** | — |
| `02_line_search_methods` | Rosenbrock에서 GD ≫ BFGS > Newton | **19436 / 36 / 22** 회 | 01 |
| `03_gauss_newton_lm_trust_region` | $J$ 가 rank 결핍이면 GN은 죽고 LM은 산다 | GN `LinAlgError` / LM 수렴 (RMS **0.93 → 0.0071**) | 02 |
| `04_robust_loss_and_irls` | outlier 10%에서 $L_2$ 는 깨지고 robust loss는 버틴다 | $\lVert\theta-\theta^*\rVert$ **0.91 / 0.17 / 0.044 / 0.026** ($L_2$/Huber/Cauchy/Tukey) | 03 |
| `05_sparse_ba_schur` | Schur는 dense와 **같은 답**을 더 싸게 준다 | 차이 **2.8e-16**, landmark 640에서 **7.7배**, 기울기 2.07 vs 1.23 | 03 |
| `06_manifold_optimization` | 좌/우 섭동을 섞으면 Jacobian이 틀린다 | 일치 **1.9e-10** / 섞음 **0.80** | 03 |

`06` 에서 하나 더 얻었다. **규약을 섞어도 발산하지 않는다.** $J^{T}J$ 가 양정부호인 한 방향은
여전히 내리막이라 같은 최소점에 도착하고, 다만 Newton 스텝이 아니게 되어 **2차 수렴이 1차로
떨어진다** (2회 → 12회). 작은 문제에서는 반복 몇 번 더 도는 것으로 끝나 알아채지 못하고,
큰 BA에서는 solver 탓으로 착각하게 된다. 중앙차분 대조가 유일한 방어선인 이유다.

공유 모듈은 `problems.py` 다. 여섯 문제의 생성기와 참값, SO(3) 헬퍼가 들어 있다.

검증은 전부 **참값 대비 숫자**로 한다. 합성 데이터라 참값을 알고 있으므로, "수렴했다"가 아니라 "맞게 수렴했다"까지 말할 수 있어야 한다.

---
# 11. 참고 자료

- Nocedal & Wright, *Numerical Optimization* — Stage 2·3의 표준 교과서. 3장(line search), 4장(trust region), 10장(least squares)
- Madsen, Nielsen & Tingleff, *Methods for Non-Linear Least Squares Problems* — 60쪽. LM의 $\lambda$ 제어와 gain ratio가 가장 간명하다
- Triggs et al., *Bundle Adjustment — A Modern Synthesis* — Stage 5의 원전
- Barfoot, *State Estimation for Robotics* — Stage 6. 좌/우 섭동 규약을 끝까지 일관되게 쓴다
- Ceres Solver / GTSAM 문서 — 위 내용이 실제 API에서 어떤 이름으로 나타나는지
