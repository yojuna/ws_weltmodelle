# Offset autopsy — why Euclidean `φ` wins short but not offset

**Date opened:** 2026-08-28  
**Trigger:** Multi-seed Euclidean replicate (`04_euclid_multiseed.md` / `lewm_phi_euclid_multiseed_summary.md`):
- short: E2 **36.7%** ≫ E1 **16.7%** (real signal)
- offset: E2 **6.7%** ≈ E1 **4.7%** ≤ E4 **7.3%** (no useful win)

**Goal:** Name the failure mode with measurements, then pick **one** transfer fix.

**Frozen weights:** `stablewm/checkpoints/pusht/lewm_phi_v2/reach.pt`  
**Trunk:** `hf_pusht` frozen  
**Compare always:** L2-`z` · trained `φ` · random `φ` (same init seed)

---

## Hypotheses (pick with data, don’t assume)

| ID | Hypothesis | What would confirm it |
|----|------------|------------------------|
| H1 | `φ` ranks **real** progress well, but **imagined** `ẑ` is OOD so CEM scores garbage | High corr(d,k) on real path; gap `d(ẑ,z*)−d(z_true,z*)` large / ranking flips on rollouts |
| H2 | Cost **landscape collapsed** for CEM (low std across candidates) | `diag_cost_scale`-style stats: φ std ≪ L2 on same candidates |
| H3 | `φ` does **not** rank real offset progress (only short-lag) | Low corr(d, remaining) on offset-length real segments |
| H4 | Absolute success ceiling is low (dynamics/search), cost choice is second-order | All costs similar; top-vs-bottom action gap tiny in pose |

---

## How we run it (two phases)

### Phase A — Offline bank autopsy (fast, primary)

No full CEM eval loop. Collect one kinematic bank (seed 0, ~200 eps), sample **offset pairs** (same `pair_mode=offset` as eval).

For each pair `(t → t+Δ)` with `Δ≈25`:

1. **Real-path progress**  
   Encode `z_{t+k}` for `k=0…Δ`. For each cost, compute `d(z_{t+k}, z_g)`.  
   Log: Pearson/Spearman vs remaining steps; fraction of steps where cost decreases.

2. **Imagination gap**  
   Roll out the **true bank actions** for `H` steps from `t` through the frozen predictor → `ẑ_{t+H}`.  
   Compare `d(ẑ_{t+H}, z_g)` vs `d(z_{t+H}, z_g)` and `‖ẑ−z‖₂` / `‖φ(ẑ)−φ(z)‖₂`.

3. **Candidate ranking (synthetic CEM)**  
   Sample `S` random action sequences; `get_cost` under each head; correlate cost with a cheap proxy (e.g. predicted `‖ẑ_H − z_g‖₂` as reference, plus optional short env rollout of best/worst on a subset).

**Outputs:**  
`le-wm/eval_results/pusht/offset_autopsy/`  
- `summary.json` + `summary.md`  
- per-pair CSV / plots: `real_progress.png`, `imagination_gap.png`, `cost_spread.png`

**Runner:** `le-wm/scripts/offset_autopsy.py` (to implement)

**Budget:** ~15–30 min on GPU (bank + encodes), not another multi-seed wall-clock day.

### Phase B — Online CEM slices (only if A is ambiguous)

Small `eval_live` with `--plan-debug` on **n=8–12** offset episodes × {E1, E2, E4}, seed 0:

- Elite cost curves, near-miss plots (existing `PlanDebugger`)
- Optional: dump final elite action and compare end pose

Use this to confirm H2/H4 under the real solver, not to re-litigate success %.

---

## Decision rule after Phase A

| Finding | Next experiment |
|---------|-----------------|
| H1 strong (real OK, imagine bad) | Train `φ` on **imagined** hindsight / predictor rollouts (transfer fix #1) |
| H2 strong (collapsed spread) | Cost normalize / temperature / hybrid `L2 + α φ` |
| H3 strong (real ranking fails at long lag) | Different target (longer-k weight, or Sep) — readout capacity issue |
| H4 strong (all costs ~useless) | Shorter horizon + more replan / policy — not another `φ` loss |

**One fix only** after the autopsy; then multi-seed offset re-eval (E1/E2/E4).

---

## Explicit non-goals

- No new IQL run in this autopsy  
- No Sep encoder until A points at H3/capacity  
- No 3-seed full offset campaign until a fix is chosen  
- Do not overwrite `lewm_phi_euclid_multiseed/` artifacts

---

## Checklist

- [x] Implement `scripts/offset_autopsy.py` (Phase A)
- [x] Run Phase A → write `docs/lewm_phi_offset_autopsy_summary.md`
- [x] Map result → H1–H4
- [ ] Choose single fix; open a short follow-up note
- [ ] Phase B only if needed
