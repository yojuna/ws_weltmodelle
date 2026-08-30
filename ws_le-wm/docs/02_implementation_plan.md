# Implementation Plan — `lewm-phi` v1 (PushT)

**Depends on:** `00_decisions.md`, `01_design_spec.md`  
**Stack:** `le-wm/` + stable-worldmodel planning + existing PushT eval logging

**Status (2026-08-30):** Phases 0–4 of this document are **done** (Phase A campaigns). The v1 “go / pivot” is recorded in `07` / `08`: short-horizon φ > L2 is supported; offset / φ-alone is not. Phase B diagnostics are in [`11_phase_b_report.md`](11_phase_b_report.md); C0 gate in [`13_phase_c0_report.md`](13_phase_c0_report.md). Remaining engineering follows [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md) (CA0 first). What is in the tree: [`10_implementation_status.md`](10_implementation_status.md). Do not start another cost-head phase from this file. Do not start C1.

---

## Guiding rules

1. Smallest diff that enables E0–E4 ablations in the design spec.
2. Do not refactor stable-worldmodel; hook cost + train `φ` locally.
3. Prefer a **frozen pretrained trunk + train `φ`** milestone before joint WM training.
4. No `S` in losses; no IQL; no C3; no Maze in this plan.

---

## Phase 0 — Baseline freeze (½–1 day)

**Goal:** Reproducible PushT reference numbers before any `φ` work.

### Tasks

- [ ] Confirm pretrained PushT LeWM checkpoint loads (`policy=…` under `$STABLEWM_HOME`).
- [ ] Run existing short-horizon eval (goal-offset / `eval_logging`) and save artifacts under `eval_results/pusht/baseline_e0/`.
- [ ] Record: success rate, mean pose/angle error, mean planning time.
- [ ] Snapshot Hydra configs + git commit hash in that folder’s `metrics.json` notes.

### Exit criteria

E0 numbers exist and match prior “known good” runs within noise.

---

## Phase 1 — Reachability module + cost hook (1–2 days)

**Goal:** Code path for `plan_cost=phi_d` without yet claiming performance.

### 1.1 New module

Add `le-wm/reachability.py` (suggested):

```text
ReachProjection  # MLP z → u
Distance         # euclidean | sq_euclidean | quasimetric_proto
ReachabilityHead # wraps both; forward(z, z*) → d
```

Unit tests (lightweight):

- Shape checks `(B,D) → (B,d_φ)`
- `d(a,a) ≈ 0`
- Gradients w.r.t. `φ` params exist; with `z.detach()` no grad to fake encoder params

### 1.2 Wire into `JEPA`

In `jepa.py`:

- Optional `self.reach = ReachabilityHead | None`
- `criterion`: if `reach` enabled, cost = `reach(pred_z, goal_z)`; else legacy MSE
- `get_cost`: unchanged control flow (encode goal pixels → emb → rollout → criterion)

Keep backward compatible: no reach head ⇒ identical to today’s LeWM.

### 1.3 Config

- `config/train/model/lewm_phi.yaml` — JEPA + reach head hydra targets
- Eval: `config/eval/pusht_phi.yaml` with `plan_cost: phi_d` and legacy toggle

### Exit criteria

Dummy/random `φ` runs CEM without crash; E4-style run produces finite costs.

---

## Phase 2 — Offline hindsight dataset + train `φ` (2–3 days)

**Goal:** Protocol T1 — freeze trunk, train `φ` on HDF5 hindsight pairs.

### 2.1 Pair sampler

Implement dataset or collate transform:

- Input: trajectory pixels window
- Sample `(t, k)` with `k ~ Uniform(1, k_max)` (start `k_max=25` to align with PushT goal offset)
- Yield pixels_t, pixels_tk, target_k

Re-encode inside training step (not cached latents).

### 2.2 Training entrypoint

Prefer **new** `train_phi.py` (cleaner than overloading `train.py`):

- Load LeWM ckpt → freeze all trunk modules
- Optimizer only on `reach` params
- Loss: Huber/MSE(`d(φ(z_t), φ(z_{t+k}))`, `k` or normalized `k`)
- Log: `reach_loss`, mean predicted `d`, correlation(d, k) on val

Hydra: `config/train/phi_pusht.yaml`

### 2.3 Checkpointing

Save:

- `*_phi_object.ckpt` or weight dict with `{trunk ref, reach state_dict, cfg}`
- Training curves (wandb optional; local CSV fine)

### Exit criteria

Val correlation `(d, k)` clearly positive (e.g. > 0.5) and loss stable. If not, debug sampling / scaling before eval.

---

## Phase 3 — C1 deploy path in eval (1–2 days)

**Goal:** Few-shot anchors without streaming goal images.

### 3.1 Eval plumbing

Extend eval runner / live eval:

- `anchor_mode: stream_goal | c1`
- `c1`: set `goal_emb` / cached `z*` once from `goal_pixels` (N=1 first)
- Ensure env loop does **not** require refreshing goal pixels each replan beyond the cached embedding

### 3.2 Compatibility with pairs

`eval_logging.pairs` already has `goal_pixels`. For C1:

- Use that frame once as the anchor
- Still use `goal_state` only for **success metrics**

### Exit criteria

E1 (L2 + C1) and E2 (`φ` + C1) both runnable end-to-end.

---

## Phase 4 — Ablation campaign (1–2 days)

**Goal:** Numbers for the v1 claim.

| Exp | Cost | Interface | Script / config |
|-----|------|-----------|-----------------|
| E0 | L2 z | stream goal | existing |
| E1 | L2 z | C1 | `pusht_phi` + `plan_cost=l2_z` + `anchor_mode=c1` |
| E2 | φ | C1 | `plan_cost=phi_d` + `c1` |
| E3 | φ | stream goal | diagnostic |
| E4 | random φ | C1 | untrained / reinit reach |

### Reporting

Write `eval_results/pusht/lewm_phi_v1/summary.md` with table + notes.

### Exit criteria

Clear statement: **go / weak go / pivot** per design spec §11.

---

## Phase 5 — Only if needed (conditional)

Do **not** schedule by default. Trigger from observed failure:

| Trigger | Action |
|---------|--------|
| `φ` loss thrashing under joint trunk train | Freeze-`E` windows + re-encode buffer |
| E2 ≯ E1 but `(d,k)` correlation good | Check CEM scale / cost normalization; try quasimetric form |
| E2 ≯ E1 and weak `(d,k)` | Fix sampling / `k_max` / architecture |
| Geometry OK but planning still fails structure | **Then** consider IQL (D3 deferred), not hierarchy first |
| Need Maze walls story | New env milestone; shared API pass |

Optional joint trunk (`L_WM` + `L_reach` stopgrad): only after T1 works, as Protocol T2.

---

## Suggested file touch list

```text
docs/
  README.md                 # done
  00_decisions.md           # done
  01_design_spec.md         # done
  02_implementation_plan.md # this file

le-wm/
  reachability.py           # NEW
  jepa.py                   # criterion / optional reach head
  train_phi.py              # NEW
  config/train/phi_pusht.yaml          # NEW
  config/train/model/lewm_phi.yaml     # NEW
  config/eval/pusht_phi.yaml           # NEW
  eval_logging/runner.py    # C1 anchor mode
  eval_live.py              # if used for interactive runs
  scripts/test_reachability.py         # NEW small tests
```

Avoid editing `stablewm/` vendored package unless a bug blocks cost injection.

---

## Milestone timeline (calendar sketch)

| Milestone | Outcome |
|-----------|---------|
| M0 | E0 baseline archived |
| M1 | Random `φ` planning path works |
| M2 | Trained `φ`, val `(d,k)` healthy |
| M3 | C1 eval works |
| M4 | E0–E4 table + go/pivot decision |

Rough wall time if focused: **~1–1.5 weeks** calendar for M4, assuming GPU access and existing PushT ckpt/data.

---

## Definition of done (v1 engineering)

- [ ] Docs match code flags (`plan_cost`, `anchor_mode`)
- [ ] E0–E4 runnable from documented commands
- [ ] Trunk never receives `L_reach` gradients (assert in test)
- [ ] No success/`S` terms in train_phi loss
- [ ] Results + configs saved under `eval_results/pusht/lewm_phi_v1/`

---

## First coding step (when implementation starts)

1. Add `reachability.py` + unit test.  
2. Patch `jepa.criterion` with a flag.  
3. Smoke-eval with random `φ` (E4).  
4. Only then build `train_phi.py`.

Do not start with online env loops or IQL.
