# Notebook skeleton

A cell-by-cell template. Copy the structure, replace the algorithm.

---

## Cell 1 — [markdown] 유도

Follows `writing-algorithm-notes`. 개요 → `# 1.` → `## 1.1.` … Ends with the pipeline box that
the code cells below will mirror:

```markdown
# 개요

- <알고리즘 한 줄>
- <어떤 문제를 푸는가>

# 1. <큰 주제>

- <독자가 이미 아는 방법>
	- <한 줄 특징>
- <설명할 방법>
	- <한 줄 특징>

## 1.1. <가장 단순한 형태>
## 1.2. <핵심 residual / 목적함수>     ← $$\boxed{}$$
## 1.3. <유도: 미분 / Jacobian>        ← $$\boxed{}$$
## 1.4. <최적화와 update>              ← $$\boxed{}$$ + 파이프라인 박스
```

---

## Cell 2 — [code] setup

```python
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
plt.rcParams["figure.dpi"] = 110

# 한글 폰트가 있으면 쓴다. 없으면 그림의 한글 라벨이 네모로 나온다.
from matplotlib import font_manager
_available = {f.name for f in font_manager.fontManager.ttflist}
for _name in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic",
              "Malgun Gothic", "Noto Sans CJK KR", "Arial Unicode MS"):
    if _name in _available:
        plt.rcParams["font.family"] = _name
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["mathtext.fontset"] = "dejavusans"   # 수식은 계속 DejaVu 로
        break
```

`%autoreload` goes before any project import — it only tracks modules loaded after it.
`axes.unicode_minus = False` is needed because the Hangul fonts lack the Unicode minus glyph,
so axis tick labels like `-15` break without it.

---

## Cell 3 — [markdown] `## 0. 샘플 데이터 생성하기`

One or two lines. What the simulator produces and what the ground truth is.

---

## Cell 4 — [code] 데이터

```python
import importlib
import <helper_module>
importlib.reload(<helper_module>)     # 커널을 재시작하지 않아도 최신 버전을 반영
from <helper_module> import *

def perturb_<params>(truth, seed=1, scale=0.05):
    """참값을 흔들어 초기 추정을 만든다. 게이지에 해당하는 원소는 그대로 둔다."""
    rng = np.random.default_rng(seed)
    ...

truth = make_<data>(..., seed=0)      # 참값
init  = perturb_<params>(truth)       # 흔든 초기 추정
```

The explicit `importlib.reload` is belt-and-braces alongside `%autoreload 2` — it makes the
cell correct even in a kernel where autoreload was loaded late.

---

## Cell 5 — [markdown] `# 1. 데이터 시각화`

---

## Cell 6 — [code] 시각화 함수

Write it **once**, parameterized, so the verification cells can call it again:

```python
def show_<data>(data, params=None, ...):
    """..."""
    rng = np.random.default_rng(0)     # 색을 고정해서 실행마다 같은 그림이 나오게 한다
    ...
    # 축 비율을 실제 거리대로 맞춘다
    ax.set_box_aspect((1, 1, 1))
```

Fixed axis limits or `aspectmode="data"` matter here: a before/after pair auto-scaled to
different ranges can make a worse result look better.

---

## Cell 7 — [markdown] `# 2. <알고리즘 이름>`

Optionally list the numbered stages, matching the pipeline box.

---

## Cells 8..N — [code] 박스 수식당 한 셀

```python
# 1. <stage name>
def <stage_1>(...):
    # (N, 3)
    ...
```

```python
# 2. <stage name>
def <stage_2>(...):
    ...
```

Pure functions, explicit arguments, shape comments on intermediates.

---

## Cell N+1 — [code] 메인 루프

Flat, not wrapped in a function. Numbered comments matching the cells above:

```python
iteration = 5
state = init
history = []

for it in range(iteration):
    # 1. <stage 1>
    ...
    # 2. <stage 2>
    ...
    history.append(metric)
```

---

## Cell N+2 — [markdown] `# 3. 검증`

---

## Cells N+3.. — [code] 검증

```python
# 수렴 곡선
plt.plot([np.sqrt(np.mean(h ** 2)) for h in history], "o-")
plt.xlabel("iteration"); plt.ylabel("RMS residual"); plt.grid(True)
plt.show()
```

```python
# before / after — 같은 함수, 같은 축
show_<data>(data, init)
show_<data>(data, state)
print("error vs truth:", np.abs(state - truth).max())
```

---

# Building the .ipynb programmatically

Notebook JSON is plain enough to write directly, which is usually faster and more predictable
than driving a kernel. Minimum valid structure:

```python
import json

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": next_id(),
            "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

def md(src):
    return {"cell_type": "markdown", "id": next_id(),
            "metadata": {}, "source": src.splitlines(keepends=True)}

nb = {
    "cells": [...],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
```

Notes that save debugging time:

- `nbformat_minor` 5 requires a unique `id` on every cell (8 hex chars is conventional).
- `source` is a **list of lines with `\n` kept** — a single joined string technically loads but
  renders as one line in some tools.
- `ensure_ascii=False` keeps Korean readable in the file and in git diffs.
- Backslashes in LaTeX: write the source in a normal string and it's fine, but if you build it
  in Python with `\n` escapes, use raw strings so `\boxed` doesn't become a backspace.

**Always validate after writing**, then execute to confirm the code actually runs:

```bash
python -c "import json; json.load(open('X.ipynb')); print('JSON OK')"
jupyter nbconvert --to notebook --execute --inplace X.ipynb   # 실행까지 확인
```

`nbconvert --execute` is what separates "the notebook parses" from "the algorithm converges" —
run it before claiming the notebook works, and read the resulting outputs.
