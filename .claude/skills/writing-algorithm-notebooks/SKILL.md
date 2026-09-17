---
name: writing-algorithm-notebooks
description: Use when the user wants an algorithm worked out in a Jupyter notebook (.ipynb) — "이 노트북에 X 정리해줘", "옵티컬 플로우 노트북 만들어줘", "BA 구현해보자", "이 알고리즘 직접 돌려보고 싶다" — or when a request for 정리/설명 names a .ipynb file, an existing notebook, or asks to implement-and-verify rather than just explain. Produces the 유도 → 시뮬레이터 → 박스 수식 1:1 코드 셀 → 메인 루프 → 검증 플롯 pipeline, where every boxed equation in the markdown has exactly one code cell and the notebook proves convergence against ground truth. Also use when reviewing or extending a notebook of this kind. For a pure markdown study note with no runnable code, use writing-algorithm-notes instead.
---

# Writing Algorithm Notebooks

## Overview

A study note explains an algorithm. A notebook **proves you understood it** — it runs, and it
shows you were right. That extra burden is the whole reason to reach for `.ipynb` instead of
markdown, and it drives every structural decision here.

The reader is 나중의 나, six months out, who needs to re-derive the formula *and* trust the
implementation. So the notebook is organized as a single claim with evidence attached:

> 유도한 수식이 맞다면, 흔들어 놓은 초기값이 참값으로 수렴해야 한다.

Everything in the notebook either states that claim (markdown), executes it (code), or checks
it (plots). A cell that does none of the three is dead weight.

**Core principle: every `$$\boxed{}$$` in the markdown gets exactly one code cell, and that
cell's function signature mirrors the equation's inputs.** This correspondence is what makes
the notebook re-readable. If you can't point from a boxed equation to the function that
implements it, the notebook has drifted into being a script.

## The markdown half

The derivation cells follow **`writing-algorithm-notes`** — 개요, numbered `## 1.1.` sections,
symbol glosses, `$$\boxed{}$$` for what carries into code, `>` blockquote for the one-line
intuition, 표기 규약. Read that skill for the prose rules; don't restate them here.

Two things change in a notebook:

**The boxed equations become a contract.** In a markdown note, boxing means "you'd implement
this." In a notebook you *do* implement it, so choose boxes with the code in mind — one box
per function you're about to write. Three or four is still the right count.

**Close the derivation with a pipeline box.** The final box names the stages in order, and the
code cells below follow that exact order and numbering:

```markdown
$$ \boxed{ r \rightarrow J \rightarrow H=J^TJ,\; g=J^Tr \rightarrow H_{LM} \rightarrow \delta\xi \rightarrow T^{new} } $$
```

This one line is the notebook's table of contents. The reader scrolls, sees `# 4. Make
Jacobian`, and knows exactly where they are in the math.

## Cell order

Markdown and code alternate in a fixed rhythm. The numbering below is the notebook's, and it
should be visible in the cell headers:

```txt
[markdown]  # 개요 + # 1. 유도          ← writing-algorithm-notes 규약, 박스 수식으로 끝
[code]      setup                      ← autoreload, import, 출력/폰트 설정
[markdown]  ## 0. 샘플 데이터 생성하기
[code]      시뮬레이터 + 참값 + 흔든 초기값
[markdown]  # 1. 데이터 시각화
[code]      입력을 눈으로 확인
[markdown]  # 2. <알고리즘 이름>
[code]      # 1. <박스 수식 1>          ← 한 셀 = 한 함수 = 한 박스 수식
[code]      # 2. <박스 수식 2>
[code]      # 3. <박스 수식 3>
[code]      메인 루프                    ← 번호 주석이 위 셀들과 1:1
[markdown]  # 3. 검증
[code]      수렴 곡선 / 잔차 분포
[code]      before / after 시각화
```

Why this order and not "이론 전부 → 코드 전부": the data cell sits between the derivation and
the implementation because **you cannot debug an algorithm you can't see the input to.** By the
time you write the Jacobian, you've already looked at the point cloud and know what shape the
data has.

### Setup cell

```python
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
plt.rcParams["figure.dpi"] = 110
```

`autoreload` first, before any project import — it only tracks modules imported after it loads.
Without it, editing the sibling `.py` and re-running the cell silently keeps the stale module,
which produces confusing `NameError`s that look like the file is broken.

`suppress=True` matters more than it looks: rotation matrices printed in scientific notation
are unreadable, and you will print a lot of them.

If the notebook has Korean labels in plots, add a font fallback here — matplotlib's default has
no Hangul glyphs and renders 네모. See `references/skeleton.md`.

### Helpers live in a sibling `.py`, not in the notebook

Anything reusable — the simulator, SE(3) helpers, data structures — goes in a module next to
the notebook and is imported. The notebook keeps only what *is* the algorithm.

The reason is that the notebook's job is to be read top to bottom as an argument. Two hundred
lines of ray-casting in the middle destroys that, and it's not what you came to understand.
With `autoreload` the module is still fully editable mid-session, so you lose nothing.

## The data cell earns its own section

```python
Rs, ts, scans = make_lidar_data(objects, n_pose=9, spec=spec, seed=0)   # 참값
R_init, t_init = perturb_poses(Rs, ts)                                  # 흔든 초기 추정
```

Two non-obvious requirements, and skipping either makes the notebook unable to prove anything:

**Ground truth.** Synthetic data means you know the answer. Without it, "잔차가 줄었다" only
shows the optimizer found *a* minimum, not the right one. Real data is for later; a study
notebook that can't grade itself teaches you nothing.

**A deliberately wrong initial guess.** If you start at the ground truth, the algorithm has
nothing to fix and converges instantly whether or not your Jacobian is correct — the most
dangerous possible false positive. Perturb it:

```python
def perturb_poses(Rs, ts, seed=1, rot=0.05, trans=0.15):
    """참값을 흔들어 초기 추정을 만든다. 포즈 0 은 게이지라 그대로 둔다."""
```

Size the perturbation so the algorithm has real work but stays in its basin of convergence.
Too large and it diverges for reasons unrelated to your math, which you'll then debug for an
hour.

**Seed everything.** `seed=0` on the data, a fixed `default_rng(0)` for plot colors. A notebook
whose figure changes on every run can't be compared against yesterday's.

## One cell per boxed equation

Head every cell with the numbered comment that matches the pipeline box:

```python
# 4. Make Jacobian
def make_jacobian(scans, normals, corr, R, t):
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
```

Three habits do the heavy lifting:

**Shape comments on every intermediate.** `# (N, 3, 3)` above the line that produces it. In
numerical code the bug is almost always a shape or an axis, and the comment is what lets you
find it by reading rather than by inserting prints. It also makes the `einsum` subscripts
checkable against the line above.

**Functions are pure — arguments in, array out, no globals.** Notebook cells get re-run out of
order constantly. A function that reads a global `R` gives different answers depending on
execution history, and that class of bug is miserable to track down because the code looks
fine. Passing `R, t` explicitly costs one line and makes every cell independently re-runnable.

**Vectorize, and say what the subscripts mean.** `np.einsum("ni,nij->nj", n, P_skew)` over a
Python loop, because the whole point of working out the Jacobian in matrix form was to avoid
looping. Keep the shape comment adjacent so the subscripts stay readable.

### The main loop mirrors the math

```python
for it in range(iteration):
    # 1. Scan space to World space - (Number of Frame, N, 3)
    points, normals = scan_to_world(scans, R, t)
    # 2. Find correspondence
    corrs = find_correspondences(points, 0.2)
    # 3. Find Residual
    residuals = build_residuals(points, normals, corrs)
    residual_history.append(residuals)
    # 4. Make Jacobians between Frames
    J = make_jacobian(scans, normals, corrs, R, t)
    # 5. Normal Equation
    H = np.dot(J.T, J)
    g = np.dot(J.T, residuals)
    # 6. Fixed Pose 0
    free = np.arange(6, 6 * number_of_frames)
    H_free, g_free = H[np.ix_(free, free)], g[free]
    # 7. LM
    H_lm = H_free + lam * np.diag(np.diag(H_free))
    delta_free = np.linalg.solve(H_lm, -g_free)
    R, t = update_poses(R, t, delta_free)
```

The loop should read as the pipeline box with parentheses. Each numbered comment names a stage
from the derivation; a reader comparing the two sees the correspondence immediately.

Keep it flat and inline — no `def run_ba()` wrapper. The loop body *is* the algorithm, and
burying it in a function hides the one thing the notebook exists to show. Accumulate history
(`residual_history`) as you go so the verification cells have something to plot.

Solve, never invert: `np.linalg.solve(H_lm, -g_free)`, not `inv(H_lm) @ -g_free`. Worth a
one-line note in the markdown when you box the update equation, since the boxed math says
$\delta\xi = -H^{-1}g$ and the code deliberately doesn't.

## Verification is the point, not the epilogue

A notebook that ends at the main loop has shown that the code runs, which is not the claim.
Attach evidence in two forms:

**The metric shrinking.** Plot `residual_history` — distribution per iteration, or RMS against
iteration count. A histogram that narrows toward zero is the picture of convergence.

**Before / after against ground truth.** The same visualization function called twice:

```python
show_point_cloud(scans, R_init, t_init)    # 흔든 초기값
show_point_cloud(scans, R, t)              # 최적화 후
```

Reuse the visualizer from `# 1. 데이터 시각화` rather than writing a second one — a before/after
pair drawn by two different functions with different axis limits proves nothing.

When ground truth is available, also print the number, because eyes are generous:

```python
print("pose error:", np.abs(t - ts).max())
```

If the result is bad, **say so in a markdown cell and leave the evidence in place.** A notebook
recording "3회 이후 발산 — correspondence가 끊긴다" is more useful six months later than one
quietly tuned until the picture looked acceptable. The failed run is data about the algorithm.

## 언어

Korean prose and comments, English technical terms left in English — residual, Jacobian,
correspondence, Frame, perturbation. Same convention as `writing-algorithm-notes`.

Comments say **why**, since the code already says what:

- Good: `# 포즈 0 은 게이지라 고정한다`, `# 축 비율을 1:1:1 로 맞춰야 방이 방처럼 보인다`
- Dead: `# H를 계산한다`, `# 반복문`

Shape comments are the exception — `# (N, 3, 3)` is a "what", and it earns its place because
shape is the thing you actually need while reading numerical code.

## Common mistakes

- **Code cells with no boxed equation behind them.** The cell is doing something the derivation
  never justified. Either box the equation or drop the cell.
- **Starting from ground truth.** Converges regardless of whether the Jacobian is right.
- **No ground truth at all.** The notebook can't grade itself.
- **Functions reading globals.** Breaks on out-of-order cell execution, in a way that looks
  like a math bug.
- **Wrapping the main loop in a function.** Hides the algorithm.
- **Ending at the loop.** Running ≠ working.
- **Helpers pasted into the notebook.** Two hundred lines of scaffolding between the derivation
  and the algorithm.
- **Unseeded randomness.** Yesterday's figure is no longer comparable.
- **`%autoreload` after the project import.** Silently keeps the stale module.

## 기존 노트북에 이어쓸 때

Read the whole notebook first and inherit its conventions over this skill's defaults — its
section numbering, symbol choices, plotting helpers, comment density. Reuse its visualization
function instead of adding a parallel one, continue the numbering rather than restarting, and
match the surrounding cells' level of detail.

If the notebook is empty or a stub, build the full skeleton in `references/skeleton.md`.

## Reference

`references/skeleton.md` — a cell-by-cell template with the setup cell (including the Korean
font fallback), the data/perturbation pattern, and the verification cells, plus notes on
building the `.ipynb` JSON programmatically. Read it when starting a notebook from scratch.
