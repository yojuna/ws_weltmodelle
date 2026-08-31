# Experiment log — lewm-phi sequence

Running chronicle of tests. Normative protocols: `03`–`07`.  
**Full synthesis:** [`07_status_synthesis.md`](07_status_synthesis.md).

---

## 2026-08-28 — Timeline

| Step | What | Outcome | Doc / artifacts |
|------|------|---------|-----------------|
| T1 Euclidean v1–v3 | Hindsight-`k` on frozen LeWM | Soft short win (v2); offset no-go (v3) | `lewm_phi_v{1,2,3}_summary.md` |
| T3 IQL | Quasimetric IQL on thin `φ` | No-go vs L2 (short+offset) | `03_…`, `lewm_phi_iql_v1_summary.md` |
| Multi-seed Euclidean | Seeds 0–2, short+offset, E1/E2_v2/E4 | Short **PASS** (36.7±7.6% vs 16.7±7.6%); offset weak (6.7% vs 4.7%, random competitive) | `04_…`, `lewm_phi_euclid_multiseed_summary.md` |
| Offset autopsy A | Real ranking / imagination / CEM spread | **H1** + **H2**; not H3/H4 | `05_…`, `lewm_phi_offset_autopsy_summary.md` |
| Imagined-φ (H1) | Train `φ` on predictor futures | Val corr **0.747** but **offset FAIL** (2.0% vs v2 6.7%); short OK (31.7% ≥ E1) | `06_…`, `lewm_phi_imagined_v1_summary.md` |
| Status synthesis | Consolidate all gates | Phase A: short win real; offset/`φ`-alone capped; H1 falsified | `07_status_synthesis.md` |

---

## Locked readings

### Autopsy
- Real-path Spearman ~0.99 → `φ` **can** read progress on real `z`.
- ‖ẑ−z‖₂ ≈ 6.3 → CEM lives in **OOD** latents (H1).
- φ candidate std ≪ L2 → flat CEM landscape (H2).

### After H1 eval
- Exposing `φ` to ẑ via temporal-`k` regression **did not** fix offset (made it worse).
- Train corr ≠ planning success.

### Claim status (Phase A / D6)
- **Supported:** Euclidean `φ` > L2 on **short_horizon** (multi-seed).
- **Not supported:** reliable offset win; IQL-on-`φ`; imagined-`φ` transfer.
- **Open:** whether hybrid L2+`φ` (H2) adds anything; whether system change (replan/policy/Sep) is required for hard goals.

---

## Next-step fork (see `07_status_synthesis.md`)

Historical A vs B table is **superseded for planning** by [`09_phase_b_plan.md`](09_phase_b_plan.md). H2 hybrid is optional bookkeeping only.

---

## 2026-08-29 — Phase B engineering (no GPU numbers yet)

| Step | What | Outcome |
|------|------|---------|
| Spec | Diagnose geometry vs CEM-horizon drift vs linear state; stop cost-head churn | [`09_phase_b_plan.md`](09_phase_b_plan.md) |
| Code | Dump / probe / drift / `--horizon` CLI; episode-split `train_phi`; C1 cache key; Reacher state concat | [`10_implementation_status.md`](10_implementation_status.md) |
| Runtime | CUDA 12.6 image with baked `/opt/venv`; named container `weltmodelle-lewm` (start via `run.sh`, no reboot autostart) | `docker/` |

---

## 2026-08-29 — Phase B B1/B2 GPU runs

**Full writeup:** [`11_phase_b_report.md`](11_phase_b_report.md). Compact tables below.

Artifacts: `le-wm/eval_results/{pusht,reacher}/phase_b_dump/seed0/` and `…/phase_b_horizon/`. Frozen `hf_pusht` / `hf_reacher`. Probes used `--intervene-live`. Gates recorded here and in `11`; D6 keep later recorded in `00` (cleanup, not a flip).

### B1 — state probes + live intervention

Dump: 48 segments × 26 frames, `z_dim=192`. PushT kinematic bank 64 eps; Reacher random/`weak` 40 eps.

**PushT** (`probe_summary.json`) — mean state linear R² **0.702**, mean MLP **0.247**. Live `P` intervention on `block_x`: **hit_rate=1.0** (16/16). Per-factor linear / MLP:

| Factor | Linear R² | MLP R² |
|--------|-----------|--------|
| agent_x | −0.11 | 0.40 |
| agent_y | 0.79 | 0.38 |
| block_x | **0.90** | 0.60 |
| block_y | 0.78 | −0.14 |
| block_angle | 0.55 | 0.49 |
| agent_vx | 1.00* | 0.00 |
| agent_vy | 1.00* | 0.00 |
| remaining_k (sanity) | −8.8 | −1.5 |

\*Kinematic collector has **zero** velocity (scaler std 0); Ridge “perfectly” fits a constant. Drop vx/vy → mean linear still **~0.58**. `remaining_k` is not a D6 input.

**Reacher** (diagnostic env) — mean linear **−0.34**, mean MLP **−1.20** (unnamed `factor_*` dominate the mean). `qpos_0` linear **0.56** / MLP **0.83**; live intervention hit_rate **1.0**. Do not let the Reacher *mean* override PushT.

**D6 (09 §6):** **keep / extract.** Control-relevant PushT factors (esp. block xy) are linearly readable and linearly drive `P`. Not a flip: MLP is not ≫ linear on state, intervention does not miss. `latent_probe` heuristic `keep_extract` agrees. Do **not** start Sep.

### B2 — drift @ CEM h=5 + action liveness

h=5 is dump index `HISTORY + 5 − 1 = 7`. Use `mean_predicted_only` and `at_h5_index` (not `mean_all_frames`).

| Env | ‖ẑ−z‖ predicted-only | at h=5 | end (h≈23) | shuffle−true end gap |
|-----|----------------------|--------|------------|----------------------|
| PushT | 9.83 | **5.26** | 15.46 | **0.002** (dead) |
| Reacher | 13.45 | **11.86** | 15.41 | 0.43 (weakly live) |

Phase A 25-step open-loop ~6.3 **understated** predicted-only drift and was the wrong horizon. At the horizon CEM actually uses, PushT drift is already **~5.3** (same order as autopsy L2 candidate std ~6.3) and grows if you keep unrolling. PushT shuffle ≈ true on this **kinematic** dump: `P` is barely action-sensitive here. AdaLN is already in the predictor; do not add conditioning from this alone (kinematic actions may be too smooth).

### B2 — CEM `T` sweep (E1 L2 only, seed 0)

`--horizon T` also sets receding=`T`. PushT offset n=50; Reacher `live_reset` n=20.

| T | PushT offset | Reacher live_reset |
|---|--------------|--------------------|
| 2 | **6%** (3/50) | 35% (7/20) |
| 3 | **6%** (3/50) | 35% (7/20) |
| 5 | **4%** (2/50) | 30% (6/20) |
| 8 | **6%** (3/50) | **15%** (3/20) |

PushT T=5 at 4% vs Phase A seed-0 E1 **8%** is two successes on n=50 (noise-wide). The PushT curve is flat.

**Hierarchy trigger (09 §6):** shortening `T` does **not** lift PushT offset (flat 4–6%). Offset stays in the Phase A ~2–10% band → **Axis G (hard pose geometry) binds**, not “imagine fewer steps.” Drift@5 is already non-trivial, but the `T` sweep says that is **not** the lever. Fire hierarchy only as a **composition / search** candidate (subgoals or policy-prior), not as a rollout-shortening patch. Do not fire Sep.

### B3 — one row from the 09 table (docs only; not built)

> Linear state R² high + intervention works + shorter `T` does not help offset → **keep D6**. Search/geometry: hierarchy or policy-prior, **not** Sep, **not** another cost head.

Phase B pass sentence: on hard PushT, **geometry binds**; frozen `z` linearly exposes block pose and `P` listens; CEM-horizon shortening does not unlock offset.

---

## 2026-08-30 — Phase C0 confirmation gate

**Full writeup:** [`13_phase_c0_report.md`](13_phase_c0_report.md). Spec: [`12_phase_c_plan.md`](12_phase_c_plan.md). C1 / C-alt not built at run time.

Packing: CEM tokens are `np.tile(env_action, 5)` (SWM `action_block=5`), not zero-pad. Kinematic dump **with that pack** is still action-dead (seed 1 gap 0.004) — B2’s 0.002 was the bank.

### C0.2 — diverse-action liveness (ran first)

PushT `collector=random`, `eval_results/pusht/phase_b_dump_diverse/seed0/`.

| Bank | predicted-only ‖ẑ−z‖ | at h=5 | shuffle−true predicted gap |
|------|----------------------|--------|----------------------------|
| Random `env.step` | 5.26 | 2.94 | **1.773 PASS** |
| Kinematic (B2) | 9.83 | 5.26 | **0.002** |

Threshold was ≥0.15 or ≥10% of true drift. `P` is action-live. **C-alt not triggered.**

### C0.3 — oracle vs CEM on kinematic offset (n=50, seed 0, budget 50)

Same pair recipe as B2 T=5 (CEM L2 **4%**).

| Actor | Success |
|-------|---------|
| CEM L2 T=5 (B2) | 4% (2/50) |
| GoalPush | 8% (4/50) |
| WeakPolicy | 2% (1/50) |

**Search-binds not shown** (privileged ≠≫ CEM). C0.3b GoalPush imagination: mean ‖ẑ_end−z\*‖ 17.3 vs true-end 15.6; 8% of imaginations moved toward the goal; 1/4 env successes failed the fidelity cut. Env/controller also fails — not a clean model-fidelity redirect.

### C0.1 — seeds 1–2 + Reacher names

| Seed | block_x | block_y | live hit |
|------|---------|---------|----------|
| 0 (B1) | 0.90 | 0.78 | 1.0 |
| 1 | 0.93 | 0.86 | 1.0 |
| 2 | 0.67 | 0.96 | 1.0 |

**Pass** (seed 2 `block_x` slightly under 0.7; hit 1.0, `block_y` 0.96). **D6 keep.** Reacher renamed: `qpos_0` 0.56, `finger_y` 0.82; mean still −0.34. **Drop Reacher from the PushT legibility claim.**

### C0.4 — effective rank (informational)

`z` 192-d rank **22.5**; `u=φ(z)` 64-d rank **10.1** (ratio 0.45). Not rank-1. Does not block anything.

### Gate (pre-redo)

**Do not build C1 until a real oracle is run.** C0.3-as-run was underpowered (GoalPush/Weak on kinematic interpolated poses). Deaf-`P` is retired. D7 stays a proposal in this log, not a `00` lock. Redo: [`12a_c03_redo.md`](12a_c03_redo.md).

---

## 2026-08-30 — C0.3-redo live-bank oracle (seed 0)

**Spec:** [`12a_c03_redo.md`](12a_c03_redo.md). C1 / C-alt not built at run time.

Bank: `eval_results/pusht/c0_oracle_livebank/seed0/` — GoalPush then Weak, `(t, t+25)` windows in `short_horizon` (pose ∈ [12, 25], angle ≤ 0.25), `tile_block`. Raw stride-25 hops miss the band (median pose ~170); scan every start, keep gap 25. n=50, mean pose 22.4, source `goal+weak`.

| Arm | Env success / metric |
|--|--|
| Oracle-replay (budget 25) | **94%** (47/50) tautology pass (≥90%) |
| CEM-L2 T=5 budget 50 | **50%** (25/50); oracle−CEM = 44 pts |
| Oracle-imagine | toward-goal **2%**; mean ‖ẑ_end−z\*‖ 8.23 vs start 2.61 |

**Outcome B — MODEL FIDELITY.** Replay works; `P` does not track `z*` on those actions (2% ≪ 60%). Redirect to C-alt; keep D6; do not build C1. Search is also weak (44-pt gap) but A is blocked by imagine. **Scope:** reachable on-policy futures, not kinematic hard offset. Seeds 1–2 not run.

**Next (not started):** [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md) — CA0 closed-loop imagine discriminator first. D6 keep recorded in `00`. D7 stays proposed.

---

## 2026-08-30 — CA0 closed-loop imagine (seed 0)

**Spec:** [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md) CA0. Trajectories: `eval_results/pusht/ca0_closed_loop/seed0/ca0.npz`. Fork is `summary.json`, not a figure. **C1 / CA-train not started.**

Same live-bank as C0 (`c0_oracle_livebank/seed0/`, n=50). Re-encode every `m ∈ {1,3,5,12,25}`. `m=25` matches C0 open-loop: mean ‖ẑ_end−z*‖ **8.228**, toward-goal **2%**, d_start **2.611**.

| m | mean ‖ẑ_end−z*‖ | toward-goal |
|---|-----------------|-------------|
| 1 | 1.426 | 84% |
| 3 | 2.017 | 66% |
| 5 | 2.351 | 62% |
| 12 | 5.037 | 8% |
| 25 | 8.228 | 2% |

Pre-registered fork: **CA0-INFIDELITY**. `m=1` fails the teacher-force guard (toward 0.84 < 0.90 or d_end 1.43 > 1.0). `m=5` would have met ACCUMULATION cuts (toward ≥ 0.60 and d_end ≤ 3) if that guard had passed — do not retune after seeing the curve.

Viz toolkit v0 (spec [`15_viz_toolkit_spec_v0.md`](15_viz_toolkit_spec_v0.md)) landed after this dump: Figs 1–7 + gallery `index.html`.

**CA1** contact/free mean drift (predicted frames only): kinematic dump ratio **1.18** (11.50 vs 9.78); diverse dump **0.83**; CA0 `m=25` vs true z **1.33** (6.64 vs 4.98). Not a contact-only spike. Live-bank wall band exists (wall 7.29 vs free 4.98); kinematic dump never hits the wall mask.

**CA2** (Fig-7 on kinematic dump): participation ratio **22.5** / 192-d; 90% variance at k=29; dead-dim fraction **0.58**. Hard elbow + dead tail — do not scale the token.

**CA3** (`block_x` ε-sweep, one P step): slope **0.255** free / **0.252** contact, linearity R² >0.999 both. Geometry is linearly steerable; does not rescue the CA0 rollout fault.

CEM capture (ep 0 first replan): selected cost **5.69**, oracle packed cost **21.8**, regret **+16** (oracle expensive in imagination → model-side, consistent with Outcome B).

**Do not start C1 or CA-train from this result.** Pictures are not a gate.

Viz toolkit **v3** report corrected: BLUF uses bank-mean end-dist **8.23** (pair 0=18.8 is labeled example); A5 citable scalar is full-space bank angle **42.5°**; A3 pose-probe energy ~1% on the live-bank; B2 labeled wrong-objective (more CEM iterations would recede from the oracle); same-state re-encode floor **0**. Artifact: `eval_results/pusht/viz_report/seed0/diagnostic_report.html`.

---

## 2026-08-31 — Part A encoder-floor bracket (seed 0)

**Spec:** [`16_fidelity_retrain_plan.md`](16_fidelity_retrain_plan.md) Part A only. Cuts frozen in `le-wm/thresholds.yaml` (`frac ≤ 0.25` ACCUMULATION, `≥ 0.5` INFIDELITY) **before** the GPU run. **Part B / CA-train / C1 not started.**

Bank: CA0 dump `ca0_closed_loop/seed0` + live-bank `c0_oracle_livebank/seed0` (n=50). Not kinematic `phase_b_dump`. Artifact: `eval_results/pusht/encoder_floor/seed0/encoder_floor.json` (copied next to CA0 for viz).

### A-calibrate — pass

| Check | Result |
|-------|--------|
| Same-state re-encode (32 frames, two forwards + batch-position) | **0.0** |
| Dump z vs fresh encode | rel **0.0** |
| Dump `d_end` m=1 vs `summary.json` | **1.426** matches |
| Fresh `P` m=1 vs dump ẑ | rel **0.0** (50 pairs) |
| Units | raw `encode()['emb'][:,0]`, L2, no SIGReg |

On this bank, **goal pixels are the last path frame**: mean ‖z_true[-1] − z*‖ = **0**. So CA0's 1.43 is mean last-frame ‖ẑ_end − z_true[-1]‖ (last-step one-step, mean). `frac` still uses the **bank-median over all t ≥ 3**, not that last-frame mean.

### A-decide — INFIDELITY (do not retune cuts)

| Quantity | Value | Role |
|----------|-------|------|
| Perfect floor | 0.0 | lower bound |
| One-step median (bracket) | **1.205** | number under test |
| Adjacent true-z median (null) | **1.225** | identity predictor |
| Random-pair spread median | 18.74 | latent scale |
| Fork mean `d_end` m=1 | 1.426 | CA0's 1.43; not used in `frac` |
| **frac** (medians) | **0.983** | frozen cut ≥ 0.5 → INFIDELITY |

Per-step `frac` median 1.02 (p10 0.51, p90 2.66): on a typical step, `P` is as bad as “predict no movement,” sometimes worse. Descriptive compounding (not a gate): one-step/adjacent ≈ 0.98; naive T=5 linear ≈ 4.9, sqrt ≈ 2.2.

**Guard:** `INFIDELITY`. `gate_part_b` is true on the spec. **This log does not start Part B.** Next is a scoped B plan (matched 1-step baseline + multi-step), not C1.

---

## 2026-08-31 — A-confirm (thorough, pre-retrain)

**Spec:** [`16_fidelity_retrain_plan.md`](16_fidelity_retrain_plan.md) A-confirm. Cuts frozen in `thresholds.yaml` `infidelity_confirm` before the run. Artifact: `eval_results/pusht/infidelity_confirm/summary.json`. **Part B / C1 not started.**

**Overall: CONFIRMED_INFIDELITY.** No seed fluke, not encoder jitter, not bank-specific. Writeup: [`16a_infidelity_investigation.md`](16a_infidelity_investigation.md).

| Arm | frac | guard |
|-----|------|-------|
| Live-bank seed 0 | **0.988** | INFIDELITY |
| Live-bank seed 1 | **1.092** | INFIDELITY |
| Live-bank seed 2 | **0.990** | INFIDELITY |
| Random-action (80 segs, 1840 steps) | **1.404** | INFIDELITY |

Mechanism (not new gates): P **does move** (predicted-move / adjacent **1.40**) and **hears actions** (shuffle gap / adjacent **0.58**; zero-action gap **0.22**). Pose Spearman **0.67** (tracks agent xy; block xy step median is **0** on these short windows). Direction is still wrong (~46°). Free and contact both infidelity. Large-step tercile still **0.66** (≥ 0.5). Random-action is *worse* than identity (frac 1.40).

**Before B (folded into [`16`](16_fidelity_retrain_plan.md) B.4, not launched):** (1) a **block-moving** eval bank so “fidelity fixed” is not certified on pusher-only hops; (2) **small-step tercile** as a named success criterion (pre: 2.17 / 0.96 / 0.66) so an average-only `frac` drop cannot hide the regime that accumulates.

---

## 2026-08-31 — B.eval-block + B.eval-tercile (pre-retrain, no Part B)

**Spec:** [`16`](16_fidelity_retrain_plan.md) B.eval-block / B.eval-tercile. Cuts frozen in `thresholds.yaml` (`median_step_block_xy_min: 2.0`; live-bank Δz edges `0.83155` / `1.74008`) *before* collection. Artifact: `eval_results/pusht/block_motion_eval/seed0/summary.json`. **Part B / C1 not started.**

**Overall: BLOCK_INFIDELITY.** GoalPush 80 eps yielded 237 qualifying windows; took 50. Mean per-pair block-step median **7.34 px** (cut 2.0). Bank-median `frac` **0.887** (≥ 0.5). Small-step tercile **1.70** (worse than bank median; frozen edges, not re-fit). Angle ~51°. Block xy step median **5.26** (live-bank was 0). Spearman vs block **0.43** (live-bank 0.10). Predicted-move / adjacent **1.04** (not frozen).

Live-bank dump-only slice (same frozen edges): parked-step `frac` 1.03 (n=1046) vs moving-step `frac` 0.75 (n=104, still infidelity, underpowered).

| Bank | frac | small / mid / large | block-xy step median | angle |
|------|------|---------------------|----------------------|-------|
| Live-bank seed 0 | 0.99 | 2.17 / 0.96 / 0.66 | **0** | ~46° |
| Block-moving n=50 | **0.89** | **1.70** / 1.09 / 0.72 | **5.26** | ~51° |

Claim strengthens to pusher **and** block. Small-step tercile remains the named B target. Do not retune 2.0 or the tercile edges.

---

## 2026-08-31 — Smear structure (dump-only, pre-retrain)

**Spec:** [`16a`](16a_infidelity_investigation.md) §5.3. Cuts frozen in `thresholds.yaml` `smear_structure` before reading shares. Script: `scripts/smear_structure.py`. Artifact: `eval_results/pusht/smear_structure/seed0/{live,block,live_seed1,live_seed2}.json`. Real 192-d `hf_pusht` dumps (unit tests use a 16-d fixture only). **Part B / C1 not started.**

First 192-d ridge of Δpose from Δz went negative even for **true** Δz (overfit). Instrument was restricted to occupancy-live90 with a positive control; physics is only read when that control passes.

| Bank | where | kNN cos (null) | Δpose R² from r | physics |
|------|-------|----------------|-----------------|---------|
| Live 0 / 1 / 2 | **MIXED** | 0.34 / 0.36 / 0.36 (null ~0.10) | −0.08 / 0.02 / −0.11 | **PERP_NO_POSE** (control 0.17 / 0.12 / 0.25) |
| Block-moving | **MOTION_CONFUSION** | **0.44** (0.07) | −0.10 | **PERP_NO_POSE** (control 0.26) |

All four arms **SYSTEMATIC**. Dead occupancy holds 31–35% of live-bank ‖r‖² and **14%** on the block bank (true Δz itself ~4–5% dead). Block-bank motion90 share **0.66**. Pose lives in the parallel component (R² 0.24–0.42), not in `r`.

Retrain target: concentrate heading inside the motion span; drop perpendicular fraction and cone angle. Not “stop leaking into dead dims.”

