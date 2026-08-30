# lewm-phi Phase B — diagnostics, results, and findings

**Date:** 2026-08-29  
**Repo:** `ws_weltmodelle` · branch `feat/lewm-phi` (parent + `ws_le-wm/le-wm` submodule)  
**Normative plan:** [`09_phase_b_plan.md`](09_phase_b_plan.md)  
**Chronicle:** [`experiment_log.md`](experiment_log.md)  
**Code vs unrun (pre-GPU):** [`10_implementation_status.md`](10_implementation_status.md)  
**This document:** standalone record of what was measured, how to read it, and what it implies. If a probe JSON disagrees with a rounded table here, the JSON wins. If this file disagrees with `00_decisions.md`, this file **does not** silently flip D6 — gates are recorded here and in the experiment log only.

**Runtime:** CUDA 12.6 container `weltmodelle-lewm` ([`../../docs/docker_usage.md`](../../docs/docker_usage.md)). Frozen checkpoints `hf_pusht` / `hf_reacher` under `$STABLEWM_HOME`.

---

## 1. One-page readout

**Why Phase B.** Phase A showed a thin reachability head `φ` is **not** the bottleneck on hard PushT. Short hops: Euclidean `φ` **36.7±7.6%** vs L2 **16.7±7.6%** (random `φ` **25.0±8.7%**). Offset: every head ~2–10%. Phase A mixed **harder pose geometry** with **25-step predictor drift** and never asked whether frozen `z` linearly exposes sim state. Phase B **stopped iterating cost heads** and measured three axes separately.

**What was run.** CPU tests; latent dumps (48×26, seed 0); Ridge vs MLP probes with **live** one-step `P` intervention; drift curves at CEM h=5 vs action-shuffle; L2 CEM horizon sweep `T ∈ {2,3,5,8}` on PushT offset n=50 and Reacher `live_reset` n=20.

**B1 (representation).** PushT frozen `z` **linearly** encodes the T: `block_x` R² **0.90**, `block_y` **0.78**, `block_angle` **0.55**. Nudging `z` along `block_x` and taking one predictor step moves the probe the right way **16/16**. Mean linear R² **0.70** (still **~0.58** after dropping fake velocity channels). MLP is **not** better. **D6 keep / extract.** Do not start Sep.

**B2 (drift vs horizon).** At the horizon CEM actually uses, PushT `‖ẑ−z‖₂` is **5.26** (predicted-only mean **9.83**, grows to **15.5** by clip end). Phase A’s ~6.3 was a **25-step** open-loop number, the wrong horizon. The planner test: shortening `T` does **not** lift offset (4–6% flat). Drift exists; it is **not the lever**.

**Binding axis.** **Geometry / search (Axis G)** on the claim env. Frozen `z` knows where the T is; L2/`φ` distance plus 5-step CEM cannot compose a far contact-rich push. Hierarchy is a **composition** candidate, not a “imagine less” patch.

**B3 (docs only).** One row from `09` §5: keep D6; next module = hierarchy or policy-prior; **not** Sep, **not** another cost head. That B4 build was not started and is superseded by C0 → C-alt (`14`).

---

## 2. Setup and questions

### 2.1 Inherited from Phase A

Frozen LeWM JEPA trunk (`E` + `P` + SIGReg). Planning is CEM. Task interface is C1 (encode goal pixels once). No success `S` in the loss. Reachability stop-grad into `E` (**D6**). Claim env **PushT**; **Reacher** is diagnostic only.

`short_horizon` vs `offset` in [`eval_logging/pairs.py`](../le-wm/eval_logging/pairs.py) share `goal_offset=25`. They differ by **pose band**, not imagination length:

| Mode | Temporal Δ | Pose band | Angle |
|------|------------|-----------|-------|
| `short_horizon` | 25 steps | [12, 25] | ≤ 0.25 |
| `offset` | 25 steps | [20, 55] | ≤ 0.6 |

Default CEM (`config/eval/pusht.yaml`): `horizon=5`, `receding_horizon=5`, `action_block=5`.

### 2.2 Questions (from `09`)

| Axis | Question |
|------|----------|
| **R** | Which ground-truth factors are linearly vs nonlinearly in frozen `z`? Does a linear nudge drive `P`? |
| **D** | How large is `‖ẑ−z‖` at **h=5**? Does shuffled action change Δz? Does shortening CEM `T` lift offset success? |
| **G** | Already measured in Phase A (offset band). Keep pair_mode and `horizon` as **separate** knobs. |

**D6 flip** only if factors are present (high MLP R²) but linear R² is poor **and/or** linear directions do not drive `P`. Not because on-path Spearman(`k`) is high (Phase A ~0.99 is tautological).

**Hierarchy fire** if drift@5 is large vs L2 spread, **or** shorter `T` lifts offset, **or** drift@5 is small while hard geometry still fails (composition, not OOD).

### 2.3 What we explicitly did not run

No new `φ` loss, IQL, imagined-φ, hybrid `L2+α d_φ` as a result, Sep, FF-JEPA `G`, residual policy, new AdaLN.

---

## 3. Methods

Code: [`phase_b.py`](../le-wm/phase_b.py) (`HISTORY=3`, `CEM_HORIZON=5`, `DUMP_VERSION=1`). Dump index of CEM h=5 is `HISTORY + 5 − 1 = 7`. `summarize_drift` reports `mean_all_frames` (includes teacher-forced zeros), `mean_predicted_only`, `at_h5_index`. **Use the last two** for gates.

| Step | Command / script | Output |
|------|------------------|--------|
| Tests | `scripts/test_phase_b.py`, `test_reachability.py` | pass |
| Dump | `latent_dump.py --env {pusht,reacher} --seed 0 --device cuda` | `eval_results/{env}/phase_b_dump/seed0/dump.npz` + `dump.meta.json` |
| B1 | `latent_probe.py --dump … --intervene-live --device cuda` | `probe_summary.json`, `probe_r2.png` |
| B2 drift | `predictor_drift.py --dump …` | `drift_summary.json`, `drift_curve.png` |
| B2 `T` | `eval_live.py --plan-cost l2_z --horizon T --receding` matches T | `phase_b_horizon/…/metrics.json` |

**Dump construction.** Encode each frame with frozen `E`. Open-loop imagine with true padded actions and with **future-action shuffle** (history kept). Persist `z`, `z_hat`, `z_hat_shuf`, state, `remaining_k`, remaining pose-to-clip-end, episode id.

- PushT: kinematic bank, 64 episodes, 48 segments of length 26, `ckpt=hf_pusht`.
- Reacher: `collector=weak` (random actions), 40 episodes, same segment geometry, `ckpt=hf_reacher`.

**Probes.** Episode-holdout Ridge (`alpha=1`) vs MLP (128–64, early stopping). Val ≈ 25% of episodes (PushT 24/8, Reacher 19/6). Intervention target: `block_x` or first factor. **Live** test: one `P` step, **zero** actions, `z` vs `z+ε d` (`ε=0.5`, 16 trials); hit = probe on predicted next increases.

**Horizon sweep.** E1 L2 only, seed 0. PushT: `online_offset`, `pair_mode=offset`, n=50, kinematic collect 200. Reacher: `live_reset`, n=20. `T ∈ {2,3,5,8}`. Setting `--horizon T` with receding default 0 sets **receding = T** (shorter T = more frequent replan with shorter look-ahead).

---

## 4. B1 results — what frozen `z` contains

### 4.1 PushT (claim env)

Artifacts: `eval_results/pusht/phase_b_dump/seed0/probe_summary.json`.

**Headline:** mean state linear R² **0.702**, mean MLP **0.247**, live `block_x` hit_rate **1.0**. Script heuristic: `keep_extract`.

| Factor | Linear R² | MLP R² | Notes |
|--------|-----------|--------|--------|
| agent_x | −0.109 | 0.398 | Not a clean linear axis; MLP modest |
| agent_y | 0.791 | 0.377 | Linear wins |
| **block_x** | **0.903** | 0.602 | Primary object pose |
| block_y | 0.780 | −0.141 | Linear wins; MLP overfit |
| block_angle | 0.549 | 0.494 | Present, noisier |
| agent_vx | 1.000 | 0.000 | **Invalid** — kinematic bank has ~zero velocity |
| agent_vy | 1.000 | 0.000 | Same |
| remaining_k | −8.79 | −1.49 | Sanity path probe; **not** D6 |
| remaining_pose | −9.92 | −0.96 | Scalar “distance to *this clip’s* end” from `z_t` **alone** |

Dropping vx/vy, mean linear on the five pose factors is **~0.58**. Still above the script’s 0.4 keep threshold.

**How to read the bad path probes.** `remaining_k` and `remaining_pose` are functions of *(frame, this segment’s goal)*, not of a single embedding. The same visual state can sit at different `k` in different clips. Negative R² there does **not** mean “progress is absent from `z`.” Do not confuse this with Phase A’s on-path Spearman(`k`)≈0.99 (ranking along one real trajectory).

**Intervention.**

| Test | Result |
|------|--------|
| Offline imagined-step sign hit | 0.42 (≈ chance) — weak proxy |
| Live `P(z+ε d)` vs `P(z)`, zero actions, `block_x` | **1.0** (16/16) |

Live test says: the linear `block_x` direction is **causally live** in one-step `P`. It does **not** say multi-step rollouts track real physics.

### 4.2 Reacher (diagnostic env)

Artifacts: `eval_results/reacher/phase_b_dump/seed0/probe_summary.json`.

State is concatenated `qpos/qvel/finger/target`; only the first two names are `qpos_0`, `qpos_1`. Mean linear **−0.34**, mean MLP **−1.20**. Script heuristic `information_may_be_absent` is **misled by the mean**.

| Factor | Linear R² | MLP R² |
|--------|-----------|--------|
| qpos_0 | **0.565** | **0.834** |
| qpos_1 | −0.551 | −0.047 |
| factor_2…7 | mixed, many ≪ 0 | worse |

Live intervention on `qpos_0`: hit_rate **1.0**. The joint we named is readable and drives `P`. Unlabeled channels (likely velocities / target) are not a D6 flip on the claim env.

MLP > linear on `qpos_0` (gap +0.27) is a **mild** nonlinear remainder, not “state only nonlinear + intervention miss.”

### 4.3 D6 gate

| Criterion (`09` §6) | Observation | Flip? |
|---------------------|-------------|-------|
| High MLP, poor linear | PushT: linear **>** MLP on pose | No |
| Linear directions do not drive `P` | Live hit 1.0 on `block_x` | No |
| On-path Spearman(`k`) | Demoted; not used | No |

**Recorded decision: keep D6 (extract).** Reachability stays a thin readout. Sep / encoder value-shaping is not justified. `00_decisions.md` left unchanged pending agreement; the measurement is keep.

---

## 5. B2 results — drift, liveness, CEM `T`

### 5.1 Open-loop drift vs horizon

Artifacts: `dump.meta.json`, `drift_summary.json`, `drift_curve.png` (not `drift_vs_h.png`).

Teacher-forced prefix: 3 frames at ~0 error. CEM marker: frame index 7.

| Env | mean all frames | predicted-only | **at h=5** | end (~23 pred steps) | shuffle−true end | shuffle−true mean (pred) |
|-----|-----------------|----------------|------------|----------------------|------------------|---------------------------|
| PushT | 8.70 | **9.83** | **5.26** | 15.46 | **0.002** | 0.001 |
| Reacher | 11.89 | **13.45** | **11.86** | 15.41 | 0.427 | 0.584 |

**Vs Phase A autopsy.** Mean `‖ẑ−z‖≈6.3` on a 25-step true-action path **included** teacher-forced zeros, so it **underestimated** predicted-only error, and it was **not** CEM’s horizon. At h=5, PushT drift (**5.26**) is the same order as autopsy L2 candidate **std ≈ 6.3**. Imagination is already noisy where CEM scores. Unrolling to the old 25-step window goes to ~15.

**Liveness.** PushT kinematic dump: shuffled future actions ≈ true. Either `P` barely uses action on this data, or kinematic finite-difference actions are too homogeneous for a shuffle test. **Do not** add AdaLN (already in `ConditionalBlock`) from this dump. Reacher random-action dump is weakly live (end gap 0.43).

### 5.2 CEM horizon sweep (E1 L2)

`--horizon T` and receding=`T`. Seed 0.

**PushT offset n=50** (Phase A seed-0 E1 was **8%**, 4/50):

| T | Success | n_ok | mean min state dist | mean plan s/ep |
|---|---------|------|---------------------|----------------|
| 2 | **6%** | 3/50 | 180.1 | 6.89 |
| 3 | **6%** | 3/50 | 134.8 | 7.18 |
| 5 | **4%** | 2/50 | 137.8 | 6.18 |
| 8 | **6%** | 3/50 | 131.9 | 4.75 |

T=5 at 4% vs 8% is two successes on n=50 — within binomial noise. The curve is **flat**: no T unlocks offset.

**Reacher `live_reset` n=20:**

| T | Success | n_ok | mean min dist | mean plan s/ep |
|---|---------|------|---------------|----------------|
| 2 | 35% | 7/20 | 0.143 | 9.85 |
| 3 | 35% | 7/20 | 0.149 | 12.7 |
| 5 | 30% | 6/20 | 0.139 | 15.3 |
| 8 | **15%** | 3/20 | 0.130 | 4.28 |

On Reacher, longer look-ahead is **worse** (15% at T=8 vs 30–35% at short T). Axis D is more plausible here than on PushT. It does not rewrite the PushT claim.

### 5.3 Hierarchy / rollout trigger

| Trigger | PushT outcome | Fire? |
|---------|---------------|-------|
| Drift@5 large vs L2 spread | ~5.3 vs std ~6.3 — noisy, same order | Ambiguous alone |
| Shorter `T` lifts offset | **No** — 6, 6, 4, 6% | **No** |
| Drift@5 small + Axis G still binds | Drift not small; offset still ~5% | Composition yes, OOD-as-lever **no** |

**Recorded:** do **not** treat “shorten imagination” as the fix. If hierarchy is built, it is **subgoal composition / search**, because hard pose goals stay hard at every `T`.

---

## 6. What we found (synthesis)

### 6.1 The Phase B pass sentence

> On hard PushT, **geometry / search binds**. Frozen `z` linearly exposes block pose and one-step `P` listens to that direction. CEM-horizon shortening does not unlock offset. Keep D6. Do not iterate the cost head.

### 6.2 How the three axes resolved

```text
Axis R (entanglement)     → NOT binding on PushT (linear pose + live P)
Axis D (drift @ CEM h=5)  → real, but NOT the lever (T sweep flat)
Axis G (hard pose goals)  → BINDING (offset 4–6% at every T, matching Phase A band)
```

### 6.3 How this sits on the v1 claim

Claim: frozen JEPA + thin `φ` + C1 beats L2 without trunk value-shaping.

| Piece | Status after B1/B2 |
|-------|-------------------|
| Trunk stores control-relevant state | **Supported** (block xy linear) |
| Thin `φ` is why offset fails | **Falsified** (Phase A + flat T-sweep on L2) |
| Must reshape `E` with value (flip D6 / Sep) | **Not supported** |
| Offset fails because CEM’s 5-step ẑ is OOD | **Insufficient** — shortening T does not help |
| Offset fails as far-pose / contact search | **Supported** |

The latent is a **state embedding**. L2(`z`, `z*`) and `φ` trained on temporal `k` ask a **metric on that embedding** to stand in for multi-step contact-rich reaching. That works for nearby hops (Phase A short `φ` win, partly CEM-conditioning). It does not compose a far T.

### 6.4 Findings we must not overclaim

1. **Velocity in `z`** — kinematic dump has none; vx/vy R²=1 is a constant.
2. **`P` action-dead on physics** — shuffle test is on kinematic FD actions; Reacher random dump is weakly live.
3. **Reacher mean R²** — does not mean the arm latent is empty; `qpos_0` is there.
4. **remaining_pose / remaining_k** — misspecified as single-`z` probes; not “cost-to-go absent.”
5. **High task success** — not a Phase B goal (`09` §11).
6. **Offline intervention 0.42** — ignore relative to live 1.0.

### 6.5 B3 — one architectural next step (not built)

From `09` §5 table:

| If… | Then |
|-----|------|
| Linear R² high + intervention works + shorter T does not help | **Keep D6. Search/geometry → hierarchy or policy-prior, not Sep.** |

That is the matched row. **B4** (implement `G`, a light proposal policy, or receding short-hop composition) is a later session. Still out: new `φ`, IQL-on-frozen-`φ`, hybrid cost as a paper result, Sep+SIGReg on value, TD-MPC2 (only if B1 had said information absent).

---

## 7. Artifact index

Checkpoints: `$STABLEWM_HOME/checkpoints/hf_{pusht,reacher}/` (gitignored). Eval under `le-wm/eval_results/`.

| Item | Path |
|------|------|
| PushT dump | `eval_results/pusht/phase_b_dump/seed0/dump.npz` + `dump.meta.json` |
| PushT probes | `…/probe_summary.json`, `probe_r2.png` |
| PushT drift | `…/drift_summary.json`, `drift_curve.png` |
| Reacher dump / probe / drift | `eval_results/reacher/phase_b_dump/seed0/` |
| PushT T-sweep | `eval_results/pusht/phase_b_horizon/offset_seed0_h{2,3,5,8}/` |
| Reacher T-sweep | `eval_results/reacher/phase_b_horizon/live_seed0_h{2,3,5,8}/` |
| Compact log | [`experiment_log.md`](experiment_log.md) |
| Plan / triggers | [`09_phase_b_plan.md`](09_phase_b_plan.md) |

Code entrypoints: `scripts/latent_dump.py`, `latent_probe.py --intervene-live`, `predictor_drift.py`, `eval_live.py --horizon`, `scripts/run_horizon_sweep.sh`.

---

## 8. Reproduction

From `docker/`, container cwd is `le-wm`. `latent_probe` needs `--intervene-live` for the D6 causal gate.

```bash
./run.sh up
./run.sh python scripts/test_phase_b.py
./run.sh python scripts/latent_dump.py --env pusht --seed 0 --device cuda
./run.sh python scripts/latent_probe.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz --intervene-live --device cuda
./run.sh python scripts/predictor_drift.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz
./run.sh bash scripts/run_horizon_sweep.sh
./run.sh stop
```

---

## 9. Status

Phase B **diagnose** is evidence-complete for seed 0 on the instruments in `09`. Phase B **pass** (name the binding axis) was recorded as **geometry/search on PushT; D6 keep.** Phase B **build** (B4) was not started.

**Superseded for next steps (2026-08-30):** C0 ([`13_phase_c0_report.md`](13_phase_c0_report.md)) resolved to Outcome B (model fidelity on reachable oracle pairs). Hard-offset geometry remains a later problem; the present bottleneck is open-loop rollout fidelity. Planning: [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md). Do not start B4 / C1 from this file.
