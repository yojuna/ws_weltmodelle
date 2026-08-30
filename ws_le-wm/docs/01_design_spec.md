# Technical Design Spec — JEPA + Thin Reachability `φ` (v1)

**Status:** v1 design locked to `00_decisions.md`. Success criteria in §11 are the original Phase A gate; they are **superseded for planning** by [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md). Do not treat E2 ≯ E1 as a cue to iterate the cost head.  
**Env:** PushT (`swm/PushT-v1`)  
**Codebase:** `le-wm/` on stable-worldmodel + stable-pretraining

---

## 1. Problem and claim

### Problem

LeWM learns a stable end-to-end JEPA from pixels (`L_pred + λ_SIGReg`) and plans with CEM using **L2 distance in embedding space** to a **goal image** encoded every episode (`jepa.JEPA.criterion` / `get_cost`). That cost is not guaranteed to approximate reachability (cost-to-go), and the goal-image interface is awkward for task-agnostic deployment.

### Claim (v1)

A **thin projection `φ`** trained with **self-supervised hindsight temporal regression**, with **no gradients into the JEPA trunk**, yields a planning cost that:

1. Improves PushT planning vs L2-in-`z` under the same CEM setup, and  
2. Supports **few-shot latent anchors (C1)** without a goal-image stream in the control loop, and  
3. Does so **without** task success rewards in the learning objective.

### Non-goals (v1)

Long-horizon hierarchy, Maze walls stress test, IQL, online lifelong fine-tuning as the main result, real-robot affordances.

---

## 2. System overview

```text
                    ┌─────────────────────────────────────┐
  pixels o_t  ──►   │  E (ViT) → projector → z_t          │  JEPA trunk
                    │  P(z_hist, a_hist) → ẑ_{t+1}        │  (LeWM)
                    │  SIGReg(z) anti-collapse            │
                    └──────────────┬──────────────────────┘
                                   │ stop-grad for reach
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  φ: z → u     (thin MLP)            │
                    │  d(u, u*)     (distance / quasi)    │
                    └──────────────┬──────────────────────┘
                                   │
              practice: z* from hindsight / buffer
              deploy:   z* from few-shot anchor images
                                   ▼
                    CEM / MPC minimizes d(φ(ẑ_H), φ(z*))
```

**Intellectual split**

| Module | Role | Trained by |
|--------|------|------------|
| `E`, `P`, SIGReg | Rich dynamics model | `L_pred + λ_S L_SIGReg` |
| `φ`, `d` | Reachability geometry for planning | `L_reach` only (`λ≈0` into trunk) |
| CEM | Action selection | Inference only |
| C1 anchors | Task configuration | Encode at deploy; not a learned task head |

---

## 3. Modules (concrete)

### 3.1 JEPA trunk (existing)

Reuse as-is from:

- `jepa.JEPA` — `encode`, `predict`, `rollout`, `get_cost`
- `module.SIGReg`, `ARPredictor`, `Embedder`, `MLP` projector
- Config: `config/train/model/lewm.yaml`, `config/train/lewm.yaml`

Defaults (PushT): `embed_dim=192`, `history_size=3`, `img_size=224`, SIGReg weight ≈ `0.09`.

**v1 training of trunk**

- Prefer **load pretrained LeWM PushT checkpoint** and **freeze `E` (and optionally `P`)** while training `φ` for the first milestone (fastest falsification of “does `φ` help?”).
- Alternate path: continue joint trunk training on offline HDF5 with existing `lejepa_forward` in `train.py`, still **no** reachability gradients into trunk.

> Note: Decision D2 says joint updates of `E,P,φ`. That means *allowed* to train jointly for the full recipe. For implementation sequencing, **Phase A may freeze the trunk** to isolate `φ`; Phase B enables joint trunk+`φ` only if we train from scratch or fine-tune trunk without `L_reach` grads into `E`. Reaching loss still stops at `φ` (D6).

Clarified rule:

- **D6 always:** `L_reach` does not train `E`.
- **D2:** `E`/`P` may still update via `L_pred + SIGReg` in the same run (joint optimization of two losses on different modules), or trunk may be frozen. Both respect D6.
- **Freeze-`E` as anti-drift:** only if re-encoding is not enough (D2 fallback).

### 3.2 Reachability projection `φ`

```text
φ: R^{D} → R^{d_φ}
```

Suggested defaults:

| Hyperparameter | Start value | Notes |
|----------------|-------------|--------|
| `d_φ` | 64 or 128 | Smaller than 192; keep thin |
| Arch | 2-layer MLP, GELU, optional LayerNorm | No BN required initially |
| Input | `z` from projector output (same as planning emb) | Match `info["emb"]` |

Stop-gradient: `u = φ(z.detach())` during `L_reach` (and during cost if trunk frozen).

### 3.3 Distance `d`

**Prototype (D3):** differentiable distance on `u` trained to match temporal separation `k`.

Options (pick one for v1; document choice in code config):

1. **Euclidean:** `d(u,u*) = ‖u − u*‖_2`  
2. **Squared Euclidean:** `‖u − u*‖_2^2`  
3. **Quasimetric form (preferred inductive bias, still regression target):**  
   asymmetric parameterization (e.g. interval quasimetric / softplus asymmetries) so `d(a,b) ≠ d(b,a)` is representable, but **trained with regression on `k`**, not full IQL yet.

**Recommendation:** implement (3) if cheap; else (1). Do not block on IQL.

### 3.4 Goal / anchor encoder path

Same `E` + projector as states. Always:

```text
z* = encode(pixels*).emb
u* = φ(z*.detach())   # for costs / reach loss
```

---

## 4. Losses

### 4.1 World model (unchanged LeWM)

From `train.py` `lejepa_forward`:

```text
L_pred   = MSE(pred_emb, tgt_emb)
L_SIGReg = SIGReg(emb)           # over time×batch
L_WM     = L_pred + λ_S L_SIGReg
```

Only applied when trunk parameters are trainable.

### 4.2 Reachability regression (new)

**Hindsight pair construction** (offline HDF5 trajectories, and later optional online buffer):

For a trajectory segment with embeddings `z_0 … z_T` (re-encoded each batch from pixels):

1. Sample indices `(t, k)` with `k ∈ {1, …, k_max}`, `t+k ≤ T`.
2. Targets: `y = k` (or `y = k / k_max` normalized).
3. Predict: `δ = d(φ(z_t), φ(z_{t+k}))`.
4. Loss: `L_reach = Huber(δ, y)` or MSE.

**Mixture (practice distribution)**

| Source | Weight (start) | Definition |
|--------|----------------|--|
| Hindsight future | 0.7 | `z_{t+k}` from same trajectory |
| Random buffer state | 0.3 | `z` from another index / trajectory as `z*`, target = `k_max` or mask / ignore if unknown | 

For v1 **offline-only**, prefer **100% hindsight** (known `k`) to avoid noisy targets. Add random-buffer goals when online practice exists.

**Unknown-distance pairs:** do not train regression on pairs without a temporal path in data (no hallucinated `k`).

### 4.3 Total objective (v1)

```text
L = L_WM  [if trunk trainable]  +  λ_R L_reach(φ)
```

with `∂L_reach / ∂E = 0`.

Suggested `λ_R = 1.0` initially (scale `y` and `d` so magnitudes are O(1)).

### 4.4 Explicitly excluded from `L`

- Task success `S` / IoU / sparse reward  
- IQL expectile backups  
- VICReg / extra anti-collapse on `u` (optional later if `φ` collapses)  
- Pixel reconstruction  

---

## 5. Planning

### 5.1 Rollout

Reuse `JEPA.rollout` (history `H`, CEM samples `S`, horizon `T`).

### 5.2 Cost (replace L2-in-z)

Current (`jepa.criterion`):

```text
cost = MSE(predicted_emb[..., -1], goal_emb[..., -1])
```

New:

```text
cost = d( φ(predicted_emb[..., -1].detach()), φ(goal_emb[..., -1].detach()) )
```

Aggregate to `(B, S)` as today for CEM.

Config flag: `plan_cost: l2_z | phi_d` for ablations.

### 5.3 Deploy configurator (C1)

At episode start:

1. Load `N` anchor RGB frames (N=1 default; try N∈{1,3,5}).
2. `z* = mean_i encode(o_i).emb` (or keep set and use `min_i d(φ(ẑ), φ(z*_i))`).
3. Cache `z*` / `u*` for the episode (re-encode if `E` changes mid-eval—normally frozen at eval).
4. **Do not** feed a streaming `goal` image tensor each step.

**Anchor selection for PushT eval (fair protocol)**

- Take the **same target frame** currently used as `goal_pixels` in `eval_logging.pairs` / dataset goal-offset protocol, but pass it **once** as C1 (not as continuous goal conditioning beyond that).  
- For “few-shot,” optionally add nearby successful frames if available—still configure-time only.

Legacy goal-image eval remains available as a **baseline comparison**, not the v1 method.

### 5.4 CEM hyperparameters

Start from `config/eval/solver/cem.yaml` + `config/eval/pusht.yaml` (`horizon`, `receding_horizon`, `action_block`). Keep identical across cost ablations.

---

## 6. Data and practice modes

### 6.1 v1 data (offline)

- Existing PushT HDF5 under `$STABLEWM_HOME` / `LOCAL_DATASET_DIR` (`pusht_expert_train`, etc.).
- Keys already used: `pixels`, `action`, `state`, `proprio` (state for **eval metrics only**, not for `L`).

### 6.2 Latent-goal “practice” in v1

**Offline hindsight practice** is enough for the first claim:

- No env interaction required to train `φ`.
- “Invented goals” = future states in recorded trajectories (Family C, offline instantiation).

### 6.3 Optional later: online practice (not required for v1 claim)

Plan→act in `swm/PushT-v1`, store transitions, hindsight-train `φ`. Still **no `S` in loss**. Eval may log success.

---

## 7. Training protocols

### Protocol T0 — Baseline sanity

- Load pretrained LeWM PushT.
- Eval with existing L2 + goal-image pipeline.
- Record success rate / pose error via `eval_logging` (reference numbers).

### Protocol T1 — Train `φ` only (primary v1 path)

1. Freeze trunk (`E`, `P`, action encoder, projectors).
2. Train `φ` (+ distance params) with `L_reach` on hindsight pairs.
3. Re-encode pixels every batch with frozen `E`.
4. Eval with `plan_cost=phi_d` + C1 anchors.

### Protocol T2 — Joint WM + `φ` (optional)

1. Trainable trunk with `L_WM` only.
2. Trainable `φ` with `L_reach` (stop-grad from `φ` into `E`).
3. Same optimizer run or two param groups / LRs.
4. Compare to T1; expect similar or better if trunk benefits from more pred training—not from reachability.

### Protocol T3 — IQL upgrade (deferred)

Replace `L_reach` regression with quasimetric IQL (Destrade-style). Only after T1 ablations.

---

## 8. Evaluation protocols

### Metrics (PushT)

Reuse `eval_logging` definitions:

- Success: pose tolerance vs `goal_state` (`pusht_success`)
- Best / final pose and angle error
- Planning time / replan counts

### Experiments

| ID | Trunk cost | Task interface | Expectation |
|----|------------|----------------|-------------|
| E0 | L2 `z` | Goal image (legacy) | Reproduce baseline |
| E1 | L2 `z` | C1 few-shot | Likely drop vs E0 |
| E2 | `d_φ` | C1 few-shot | **Main result** |
| E3 | `d_φ` | Goal image | Upper bound / diagnostics |
| E4 | random `φ` | C1 | Should be weak |

Primary comparison for the paper claim: **E0 vs E2** (and E1 to show C1 alone is not enough without `φ`).

### Splits

- Train `φ` on train split of HDF5.
- Eval on held-out pairs (existing goal-offset sampling). Same seeds as current eval runs when possible.

---

## 9. Implementation surfaces (where code changes)

| Concern | Primary files |
|---------|----------------|
| `φ` module | new `le-wm/reachability.py` (or `module.py`) |
| Cost hook | `jepa.py` `criterion` / `get_cost` |
| Train loop | `train.py` or new `train_phi.py` |
| Config | `config/train/phi_*.yaml`, `config/eval/pusht_phi.yaml` |
| C1 eval path | `eval_logging/runner.py`, `eval_live.py`, pairs → pass anchors once |
| Ablation flag | Hydra `plan_cost`, `anchor_mode: stream_goal \| c1` |

Keep stable-worldmodel CEM solver unchanged; only the cost callable changes.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `φ` learns scaled L2(`z`) and adds nothing | Ablation E4; correlate `d_φ` with true steps vs ‖Δz‖ |
| Hindsight `k` ≠ optimal cost-to-go | Accept for prototype; IQL later; still useful if better than raw L2 |
| C1 weaker than streaming goal | Report E1 vs E2; few-shot is the product constraint |
| Encoder drift (if trunk trains) | Re-encode goals every batch; freeze trunk in T1 |
| Collapse of `u` | Monitor std of `u`; light variance penalty only if observed |

---

## 11. Success criteria for v1

**Go:** E2 success rate ≥ E1 and meaningfully closes gap toward E0, with clear gain of `d_φ` over L2 under C1 (E2 > E1).  
**Strong go:** E2 ≈ E0 (few-shot + `φ` matches goal-image L2 baseline).  
**Pivot:** If E2 ≯ E1, either `φ` capacity / `k` sampling is wrong, or regression signal is insufficient → consider T3 IQL before hierarchy or `S`-based learning.

---

## 12. Naming

Working name options (pick one when coding):

- `lewm-phi` / `LeWM-φ`
- `Reach-JEPA` (v1 offline hindsight)
- `C1-Reach`

Use **`lewm-phi`** in configs and checkpoints for neutrality.
