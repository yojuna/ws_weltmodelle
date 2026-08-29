# lewm-phi Phase A — design, experiments, and results

**Date:** 2026-08-29 (covers work through 2026-08-28 EOD)  
**Repo:** `ws_weltmodelle` · branch `feat/lewm-phi` (parent + `ws_le-wm/le-wm` submodule)  
**Normative sources:** `00_decisions.md`, `01_design_spec.md`, campaign summaries, `07_status_synthesis.md`  
**This document:** standalone narrative of what was designed, what was run, and what the numbers mean. If a numbered protocol disagrees with this prose, the protocol + its summary win for that campaign; if a brainstorm note in `drafts.md` disagrees, `00_decisions.md` wins.

---

## 1. One-page readout

**Setup.** Frozen pretrained LeWM (JEPA) on PushT, plus a thin projection `φ` trained by self-supervised hindsight temporal regression. Planning uses CEM. The task is configured by a few-shot latent anchor (C1), not a streaming goal image. No task-success reward in the loss; no reachability gradients into the trunk (D6).

**Claim under test.** This stack beats L2 distance in embedding space (`L2-in-z`) as a CEM cost, without reshaping the world model for control.

**What held.** On **short-horizon** PushT (n=20, seeds 0–2), Euclidean `φ` (v2 weights) succeeds **36.7±7.6%** vs L2 **16.7±7.6%**. The gap replicates on every seed. Random `φ` sits in between (~25–27%). That is a real planning signal, not a single-run fluke.

**What did not hold.** On the **firm offset** protocol (n=50, t→t+25, seeds 0–2), no thin cost head is useful. Euclid v2 is **6.7±1.2%** vs L2 **4.7±3.1%**; random `φ` often ties or wins; absolute success for *every* head is ~2–8%. Two attempted upgrades failed their gates:

| Upgrade | Train diagnostic | Offset planning |
|---------|------------------|-----------------|
| Quasimetric IQL on frozen `φ` (T3) | val `L_VF` ≈ 0.035 | 4% (seed 0) ≤ L2 |
| Train `φ` on imagined predictor futures (H1) | val corr(d,k) ≈ **0.75** | **2.0±0.0%** — worse than v2 |

**Diagnosis.** On real encoded paths, L2, trained `φ`, and random `φ` all rank remaining steps almost perfectly (Spearman ≈ 0.99). Failure is in CEM’s imagined latents: mean ‖ẑ−z‖₂ ≈ 6.3 (out of distribution), and `φ` costs are nearly flat across candidates (std ≈ 0.48 vs L2 ≈ 6.3). Seeing ẑ during temporal-`k` training did not fix offset.

**Status.** Phase A of “thin `φ` under D6” is evidence-complete. The claim is **partially supported** (short) and **not supported** (offset / `φ` alone for hard goals). Next step is a fork, not another `φ` loss: (A) eval-only hybrid `L2 + α d_φ`, or (B) change the planning system.

---

## 2. Project context

### 2.1 Workspace

`ws_weltmodelle` is a local workspace for JEPA / world-model research.

| Path | Role |
|------|------|
| `ws_le-wm/le-wm` | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) submodule; feature work on `feat/lewm-phi` |
| `ws_le-wm/docs` | Design specs, protocols, campaign summaries (this file lives here) |
| `ws_le-wm/stablewm` | Local `$STABLEWM_HOME` cache (checkpoints gitignored) |
| `ws_dino_wm` | Placeholder for DINO-WM; not started |
| `wiki` | Reference papers |

Upstream LeWM (Maes, Le Lidec, Scieur, LeCun, Balestriero) is a stable end-to-end JEPA from pixels: next-embedding prediction plus a SIGReg Gaussian regularizer. It plans with CEM using **L2 in embedding space** to a **goal image** encoded every episode. That cost is not guaranteed to be reachability (cost-to-go), and a streaming goal image is an awkward interface for task-agnostic deployment.

### 2.2 Intellectual bet

Keep the prediction-rich world model as the primary object. Put planning alignment in a **thin readout**, not by retraining the encoder for control. Configure the task at deploy time with a handful of example observations encoded once (`z*`). Falsify the smallest system first; add complexity only after a measured failure.

This is **not** DINO-WM, not FF-JEPA hierarchy, not TD-MPC reward latents, not a behavior-cloned policy replacing CEM, and not a real-robot affordance loop. Those are explicitly out of v1.

---

## 3. Design

Locked in `00_decisions.md` (2026-08-28). Technical detail in `01_design_spec.md`. Implementation sequencing in `02_implementation_plan.md`.

### 3.1 Decision table

| ID | Topic | Choice | Still deferred |
|----|--------|--------|----------------|
| **D1** | Deploy configurator | **C1** — encode 1–N example observations once → `z*` | C3 privileged→latent adapter |
| **D2** | Training schedule | Joint `E,P,φ` *allowed*; Phase A **froze the trunk** to isolate `φ` | Unfreeze only if needed |
| **D3** | Reach loss | **Euclidean hindsight-`k`** (best planning signal = v2). IQL T3 failed. Imagined-φ H1 failed offset | H2 hybrid cost at eval; Sep / system change if offset stays dead |
| **D4** | Env | **PushT** only | Shared multi-env API + Maze |
| **D5** | Task success `S` in learning | **None** | Eval may still log sim success |
| **D6** | Trunk shaping from reachability | **`φ` only (`λ ≈ 0`)** — stop-grad into `E` | Small `λ` only if `φ` cannot read progress; Sep is an explicit flip |

Stability rule (always on): store **goal pixels** (or dataset indices), not stale latents. Re-encode with the current `E` whenever computing `L_reach` or setting deploy anchors.

### 3.2 Architecture

```text
                    ┌─────────────────────────────────────┐
  pixels o_t  ──►   │  E (ViT) → projector → z_t          │  JEPA trunk
                    │  P(z_hist, a_hist) → ẑ_{t+1}        │  (LeWM, frozen in Phase A)
                    │  SIGReg(z) anti-collapse            │
                    └──────────────┬──────────────────────┘
                                   │ stop-grad for reach
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  φ: z → u     (thin MLP)            │
                    │  d(u, u*)     (Euclidean in v1/v2)  │
                    └──────────────┬──────────────────────┘
                                   │
              practice: z* from hindsight / buffer
              deploy:   z* from few-shot anchor images (C1)
                                   ▼
                    CEM / MPC minimizes d(φ(ẑ_H), φ(z*))
```

| Module | Role | Trained by |
|--------|------|------------|
| `E`, `P`, SIGReg | Dynamics model | Pretrained LeWM (`L_pred + λ_S L_SIGReg`). Frozen in Phase A |
| `φ`, `d` | Reachability geometry for planning | `L_reach` only |
| CEM | Action selection | Inference only |
| C1 anchors | Task configuration | Encode at deploy; not a learned task head |

Defaults reused from PushT LeWM: `embed_dim=192`, `history_size=3`, `img_size=224`. Suggested `φ`: 2-layer MLP, GELU, `d_φ` ∈ {64, 128}, input = projector embedding (same as `info["emb"]`).

**D6 vs D2.** Reachability loss never trains `E` (`u = φ(z.detach())`). The trunk *may* still update via `L_pred + SIGReg` in a joint run; Phase A did not do that. Freeze-`E` as anti-drift is a fallback if re-encoding is not enough.

### 3.3 Losses

World model (unchanged LeWM, not updated in Phase A):

```text
L_pred   = MSE(pred_emb, tgt_emb)
L_SIGReg = SIGReg(emb)
L_WM     = L_pred + λ_S L_SIGReg
```

Reachability (new). From a trajectory with re-encoded embeddings `z_0 … z_T`:

1. Sample `(t, k)` with `k ∈ {1, …, k_max}`, `t+k ≤ T`. Start `k_max=25` to match PushT goal offset.
2. Target `y = k` (or `k / k_max`).
3. Predict `δ = d(φ(z_t), φ(z_{t+k}))`.
4. `L_reach = Huber(δ, y)` (or MSE).

Unknown-distance pairs are not trained (no hallucinated `k`). v1 offline used **100% hindsight**. Random-buffer “far” goals were left for later online practice.

Total: `L = L_WM [if trunk trainable] + λ_R L_reach(φ)` with `∂L_reach / ∂E = 0`.

**Explicitly not in `L`:** task success / IoU / sparse reward; IQL expectile backups (until T3); extra VICReg on `u`; pixel reconstruction.

### 3.4 Planning and C1

Legacy LeWM cost: `MSE(predicted_emb[..., -1], goal_emb[..., -1])`.

New cost: `d(φ(predicted_emb[..., -1].detach()), φ(goal_emb[..., -1].detach()))`, aggregated to `(B, S)` for CEM. Flag: `plan_cost: l2_z | phi_d`.

**C1 at episode start:**

1. Load N anchor RGB frames (N=1 in all reported evals).
2. `z* = encode(o).emb` (or mean over N).
3. Cache `z*` / `u*` for the episode.
4. Do **not** stream a goal image each replan.

For fair PushT eval, the anchor is the **same target frame** the existing goal-offset protocol would have used as `goal_pixels`, passed once. `goal_state` is used only for **success metrics**.

CEM hyperparameters stay identical across cost ablations (`config/eval/solver/cem.yaml` + `config/eval/pusht.yaml`).

### 3.5 Data

**Do not** train `φ` on the HuggingFace PushT expert HDF5. `train_phi.py` builds a live `TrajectoryBank` in `swm/PushT-v1` with the same collectors as eval (`weak`, `kinematic`, or `goal`) and samples hindsight pairs from those rollouts. Eval still loads the pretrained HF **checkpoint** (`hf_pusht` weights only). See `data_source.md`.

### 3.6 Ablations required to defend the claim

| ID | Cost | Task interface | Role |
|----|------|----------------|------|
| **E0** | L2 `z` | Streaming goal image (legacy) | Reproduce upstream-style baseline |
| **E1** | L2 `z` | C1 few-shot | Isolates C1 without `φ` |
| **E2** | `d_φ` trained | C1 | **Main result** |
| **E3** | `d_φ` | Streaming goal | Diagnostic upper bound (not the campaign focus) |
| **E4** | random / untrained `φ` | C1 | Trained head must beat a frozen random projection |

Primary comparison for the claim: **E2 vs E1** (same C1 interface). E0 is a reference, not the product constraint. E4 guards against “`φ` is just a scaled L2.”

**Go criteria (design spec §11):** E2 success ≥ E1 and a meaningful gain of `d_φ` over L2 under C1. **Strong go:** E2 ≈ E0. **Pivot:** if E2 ≯ E1.

### 3.7 What was implemented

Mapped to `02_implementation_plan.md`. Phase A froze the trunk (Protocol T1) and did not run joint WM+`φ` (T2).

| Surface | Location |
|---------|----------|
| `φ` + distance | `le-wm/reachability.py` (`ReachabilityHead`; Euclidean and later IQE-sum) |
| Cost hook | `le-wm/jepa.py` `criterion` / `get_cost`; C1 goal-emb cache |
| Train | `train_phi.py`, `train_phi_imagined.py`, `train_phi_iql.py` |
| Pair sampling | `phi_data.py`, `phi_imagined_data.py`, `phi_iql_data.py` |
| Eval | `eval_live.py`, `eval_logging/` |
| Campaigns | `scripts/run_euclid_multiseed.sh`, `run_imagined_multiseed.sh`, aggregators |
| Autopsy | `scripts/offset_autopsy.py`, `scripts/diag_cost_scale.py` |
| IQL/IQE | `iqe.py`, `iql_loss.py` |

Eval caveats that still apply when reading absolute rates (from an earlier live-eval review, not a campaign result): this protocol is **not** the paper’s expert-HDF5 PushT eval (published ~96% SR is not a valid comparison target); success is pose match to sampled `goal_state`, not green-T coverage; live action scalers differ from training HDF5 stats. Compare **E1 / E2 / E4 on the same protocol**, not against published LeWM tables.

---

## 4. Evaluation protocols used in Phase A

Two pairing modes share the same live kinematic bank and C1 cache:

| Protocol | Pairing | n (typical) | Role |
|----------|---------|-------------|------|
| **short_horizon** | Nearby start/goal in the bank | 20 | Soft gate; first place a cost can show signal |
| **offset** | Paper-style `t → t+Δ` with `Δ ≈ 25` | 50 | **Firm gate** for the claim |

Collectors: `weak` (noisy policy, used heavily for Euclidean `φ` training) vs `kinematic` (start→goal motion, used for eval banks and some trains). Seeds **0, 1, 2** for the replicate campaigns. Device `cuda`. Videos off.

E0 archive (`e0_baseline_note.md`): a 10-episode L2 + goal-image run at 0% success / mean min state distance ~68.6. It is **not** matched to the later C1 tables (different n, pairing, and interface). Treat it as a freeze of the pre-`φ` pipeline, not as the L2 number to beat.

---

## 5. Experiments and results

All dates 2026-08-28. Chronicle: `experiment_log.md`. Each subsection points at the campaign summary that owns the raw table.

### 5.1 T1 Euclidean v1 — first ablation (pivot)

**Summary:** `lewm_phi_v1_summary.md`  
**Train:** live weak bank, 60 eps / 6k steps, 5 epochs. Best val corr(d,k) = 0.569. Checkpoint later treated as **leaky** (val resampled from the same bank, not episode-holdout).  
**Eval:** short_horizon, n=8, seed 0, C1 on.

| ID | Cost | Success % | mean min pose ↓ |
|----|------|-----------|-----------------|
| E1 | L2 `z` | 12.5 (1/8) | 93.7 (state) |
| E2 | trained `φ` | 12.5 (1/8) | 107.2 |
| E4 | random `φ` | 12.5 (1/8) | 96.3 |

**Verdict: pivot.** Moderate temporal signal, no planning gain. n=8 is too small to trust; next probes were more data, held-out val, and a matched n≥20 eval — not IQL yet.

### 5.2 T1 Euclidean v2 — first short-horizon win (weak go)

**Summary:** `lewm_phi_v2_summary.md`  
**Train:** `lewm_phi_v2/` · **24k** weak steps (241 eps) · 15 epochs · 217/24 episode holdout · best val corr **0.536** @ epoch 13.  
Also recorded: leaky v1 (`lewm_phi/`) and a 6k-step **fixed** split run (`lewm_phi_fixed/`, val corr 0.487) that still did not beat L2.

**Eval:** short_horizon n=20, seed 0, kinematic bank 320 eps, C1 on.

| ID | Weights | Success % | mean min pose ↓ | mean min state ↓ |
|----|---------|-----------|-----------------|------------------|
| E1 | — | 25.0 (5/20) | 27.34 | 107.1 |
| E2_fixed | `lewm_phi_fixed` | 20.0 (4/20) | 28.52 | 104.3 |
| **E2_v2** | `lewm_phi_v2` | **35.0 (7/20)** | **23.86** | **85.3** |
| E4 | random | 20.0 (4/20) | 32.43 | 144.6 |

**Verdict: weak go** for v2 weights only. Larger live bank + longer train + held-out checkpointing produced a cost that beats L2 on this matched short protocol. Small-n; next required a firmer offset / larger-n gate.

v2 is the Euclidean checkpoint used for all later multi-seed, autopsy, and imagined-φ comparison evals.

### 5.3 T1 Euclidean v3 — offset n=50 (pivot on the firm gate)

**Summary:** `lewm_phi_v3_summary.md`

| Run | Data | Epochs | Best val corr(d,k) | Note |
|-----|------|--------|---------------------|------|
| `lewm_phi_v3_kin` | kinematic 256 eps / 20.5k steps | 20 | **0.731** @ ep 1 | Train corr → 0.91; **overfits**; early-stop |
| `lewm_phi_v3_weak` | weak 481 eps / **48k** steps | 20 | 0.535 @ ep 19 | Same corr band as v2 despite 2× data |

**Eval:** `pair_mode=offset`, n=50, seed 0, kinematic bank 200 eps.

| ID | Weights | Success % | mean min pose ↓ | mean min state ↓ |
|----|---------|-----------|-----------------|------------------|
| **E1** | — | **8.0 (4/50)** | **45.4** | 139.4 |
| E2_v2 | v2 | 6.0 (3/50) | 45.0 | **130.6** |
| E2_v3_kin | v3_kin | 8.0 (4/50) | 50.3 | 185.6 |
| E2_v3_weak | v3_weak | 6.0 (3/50) | 45.8 | 144.5 |
| E4 | random | 4.0 (2/50) | 45.9 | 166.4 |

**Verdict: pivot on offset / large-n.** No trained `φ` improves success over L2. High kinematic `(d,k)` corr does not transfer to better CEM. Scaling weak/kinematic regression further was judged a dead end (signal saturates ~0.54 on weak data).

This is the trigger that opened Protocol T3 (IQL) and, after IQL failed, the multi-seed Euclidean replicate + autopsy path.

### 5.4 Protocol T3 — quasimetric IQL (no-go)

**Plan:** `03_quasimetric_iql_t3.md` · **Summary:** `lewm_phi_iql_v1_summary.md`

Destrade et al. (*Value-Guided Action Planning with JEPA World Models*) win with **VF_quasi** and a **separate** value encoder (Sep) trained by `L_VF`. Under locked D6 we adapted Appendix 7.3: frozen LeWM for rollouts; thin `φ` + **IQE-sum** (Wang & Isola) trained by Destrade Eq. (1) expectile; `V = -d`; CEM cost = `d`; stop-grad into `E`.

| Field | Value |
|-------|--------|
| Checkpoint | `lewm_phi_iql_v1/` |
| Distance | IQE-sum (`k=8, l=8`, `φ` dim 64) |
| Loss | `γ=0.93`, `τ=0.60` |
| Bank | kinematic 256 eps / 20.5k; holdout 230/26 |
| Best | epoch 15, val `L_VF=0.0353` |

Training was healthy: loss decreased, mean IQE distance stayed ~11–12 (not collapsed). `frac_s==g` ≈ 0 on sampled batches, so the reward term is almost always −1 (pure discounted consistency).

**Short n=20, seed 0**

| ID | Success % | mean min pose ↓ |
|----|-----------|-----------------|
| E1 L2 | **25.0 (5/20)** | **27.34** |
| E2 IQL | 10.0 (2/20) | 27.73 |
| E4 random IQE | 30.0 (6/20) | 24.83 |

**Offset n=50, seed 0**

| ID | Success % |
|----|-----------|
| E1 L2 | **8.0 (4/50)** |
| E2 IQL | 4.0 (2/50) |
| E4 random IQE | 6.0 (3/50) |

**Verdict: no-go.** Paper-faithful VF_quasi *on frozen-LeWM `φ`* does not beat L2 under D6. Likely mismatch: Sep trains the state encoder with `L_VF`; we only train a thin map on detached `z`. Random IQE beating trained IQL on short-horizon is a further warning on landscape / scale.

IQL was **not** multi-seeded after the seed-0 gate failed.

### 5.5 Euclidean multi-seed replicate (short PASS, offset weak)

**Protocol:** `04_euclid_multiseed.md` · **Summary:** `lewm_phi_euclid_multiseed_summary.md`  
**Weights:** frozen `lewm_phi_v2/reach.pt` (no retrain). Seeds 0, 1, 2. Conditions E1 / E2_v2 / E4. Short n=20 (kin bank 320 eps); offset n=50 (kin bank 200 eps).

#### Short-horizon

| Cond | mean±std success % | seed 0 | seed 1 | seed 2 | mean min pose |
|------|--------------------|--------|--------|--------|---------------|
| E1 L2 | **16.7±7.6** | 25% (5/20) | 15% (3/20) | 10% (2/20) | 28.09 |
| E2 v2 | **36.7±7.6** | 35% (7/20) | 30% (6/20) | 45% (9/20) | 24.35 |
| E4 random | 25.0±8.7 | 35% (7/20) | 20% (4/20) | 20% (4/20) | 27.16 |

E2 > E1 on **every** seed. E2 ≥ E4 on the mean; seed 0 ties random at 35%.

#### Offset

| Cond | mean±std success % | seed 0 | seed 1 | seed 2 | mean min pose |
|------|--------------------|--------|--------|--------|---------------|
| E1 L2 | **4.7±3.1** | 8% (4/50) | 4% (2/50) | 2% (1/50) | 47.25 |
| E2 v2 | **6.7±1.2** | 6% (3/50) | 8% (4/50) | 6% (3/50) | 44.78 |
| E4 random | **7.3±2.3** | 10% (5/50) | 6% (3/50) | 6% (3/50) | 47.99 |

Go/no-go as written in the protocol: short E2>E1 **PASS**; short E2≥E4 **PASS**; offset E2≥E1 **PASS but weak** (and random is still highest). Gaps ≲10pp with n=20/50 are fragile; the offset “win” is not a useful planning result.

**Reading.** The v2 short-horizon signal is real. Euclidean-on-frozen-`z` does not transfer to the firm offset gate. That is a protocol/transfer problem, not “`φ` never works.”

### 5.6 Offset autopsy Phase A (H1 + H2)

**Protocol:** `05_offset_autopsy.md` · **Summary:** `lewm_phi_offset_autopsy_summary.md`  
Seed 0, 24 offset pairs (`Δ=25`), v2 weights. Offline bank autopsy (no full CEM campaign).

| Hypothesis | Meaning | Result |
|------------|---------|--------|
| **H1** | `φ` ranks real progress; imagined `ẑ` is OOD so CEM scores garbage | **Confirmed** |
| **H2** | Cost landscape collapsed for CEM (low std across candidates) | **Confirmed** |
| **H3** | `φ` does not rank real offset progress (only short-lag) | **Rejected** |
| **H4** | Absolute ceiling is dynamics/search; cost choice is second-order | **Rejected** (costs are not equivalent on spread; real ranking is excellent) |

**Real-path ranking** (cost vs remaining steps):

| Cost | Spearman (mean±std) | frac decreasing |
|------|---------------------|-----------------|
| L2 `z` | 0.996±0.005 | 0.907 |
| `φ` | 0.991±0.014 | 0.847 |
| random `φ` | 0.995±0.007 | 0.868 |

**Imagination gap.** Mean ‖ẑ−z‖₂ along true-action predictor rollouts: **6.311**. (Reported “relative end-gap” ratios are inflated because they divide by `d(z_g,z_g)≈0`; prefer the L2 gap and scatter plots.)

**Candidate cost spread** (synthetic CEM):

| Cost | mean | std | cv |
|------|------|-----|----|
| L2 `z` | 123.47 | **6.31** | 0.061 |
| `φ` | 8.36 | **0.48** | 0.066 |
| random | 6.59 | 0.21 | 0.033 |

Same coefficient of variation, **~13× smaller** absolute dynamic range for `φ` than L2. Random is even flatter.

**Decision rule used:** H1 strong → try one transfer fix (train `φ` on imagined hindsight). H2 left as the backup if that failed. Phase B (online CEM slices / plan-debug) was not required to pick a hypothesis and was not run.

### 5.7 H1 imagined-φ (train OK, offset FAIL)

**Protocol:** `06_imagined_phi.md` · **Summary:** `lewm_phi_imagined_v1_summary.md`

Single fix: for hindsight `(t,k)`, roll out **true bank actions** through the frozen predictor to get `ẑ_{t+k}`, then Huber-regress `‖φ(z_t)−φ(ẑ_{t+k})‖₂` onto `k`. Mix **25%** real `z_{t+k}` so short-horizon skill is not thrown away. Same Euclidean head as v2. New checkpoint dir `lewm_phi_imagined_v1/` (v2 not overwritten).

Training: best held-out corr(d,k) ≈ **0.747** — the strongest train/val signal in Phase A.

**Eval gate:** seeds 0–2, short n=20 and offset n=50, conditions E1 / E2_v2 / **E2_imagined** / E4.

#### Short

| Cond | mean±std % | seed 0 | seed 1 | seed 2 |
|------|------------|--------|--------|--------|
| E1 L2 | 16.7±7.6 | 25 | 15 | 10 |
| E2 v2 | **36.7±7.6** | 35 | 30 | 45 |
| E2 imagined | 31.7±10.4 | 20 | 35 | 40 |
| E4 random | 26.7±2.9 | 25 | 25 | 30 |

Short imagined ≥ E1 (**PASS**); still below v2.

#### Offset

| Cond | mean±std % | seed 0 | seed 1 | seed 2 |
|------|------------|--------|--------|--------|
| E1 L2 | 4.7±3.1 | 8 | 4 | 2 |
| E2 v2 | **6.7±1.2** | 6 | 8 | 6 |
| E2 imagined | **2.0±0.0** | 2 | 2 | 2 |
| E4 random | 4.0±2.0 | 2 | 4 | 6 |

Go required mean offset(imagined) > v2 **and** ≥ E1. **FAIL** on both. Imagined-φ is strictly worse than v2 and L2, with zero seed variance at 2%.

Random offset means differ across campaigns (Euclid multi-seed 7.3±2.3 vs imagined campaign 4.0±2.0). Same `--seed` **reuses the same pairs**; the swing is random-`φ` re-init, not redrawn banks. Within a campaign, E1/E2/E4 are matched.

Lite autopsy on imagined weights was optional and **not run**. Planning next in the protocol was H2 (hybrid), not another train.

### 5.8 Master table (Phase A close)

Copied for convenience from `07_status_synthesis.md`. Multi-seed = mean±std over seeds 0,1,2 unless noted. Frozen `hf_pusht`, C1, kinematic eval banks.

**Short-horizon (n=20)**

| Condition | Seeds | Success % |
|-----------|-------|-----------|
| E1 L2 `z` | 0–2 | **16.7±7.6** |
| E2 Euclidean v2 | 0–2 | **36.7±7.6** |
| E2 Imagined-φ | 0–2 | 31.7±10.4 |
| E4 random `φ` | 0–2 | 25.0±8.7 Euclid; 26.7±2.9 imagined |
| E2 IQL | 0 only | 10.0 |

**Offset (n=50)**

| Condition | Seeds | Success % |
|-----------|-------|-----------|
| E1 L2 `z` | 0–2 | **4.7±3.1** |
| E2 Euclidean v2 | 0–2 | **6.7±1.2** |
| E2 Imagined-φ | 0–2 | **2.0±0.0** |
| E4 random `φ` | 0–2 | ~4–7 (campaign-dependent) |
| E2 IQL | 0 only | 4.0 |

**Training diagnostics (not planning)**

| Run | Best held-out signal |
|-----|----------------------|
| Euclid v2 | corr(d,k) ≈ 0.54 |
| IQL T3 | val `L_VF` ≈ 0.035 |
| Imagined-φ | corr(d,k) ≈ **0.75** |

Higher train/val corr does **not** imply better offset CEM.

---

## 6. What is falsified vs open

| Statement | Status |
|-----------|--------|
| Thin Euclidean `φ` can beat L2 on **short** hops (multi-seed) | **Supported as a phenomenon.** Matched random `φ` is 25.0±8.7; part of the gap vs L2 is CEM conditioning. |
| Same `φ` reliably beats L2 on **offset** | **Not supported** (tiny / noisy edge; random competitive) |
| Quasimetric IQL on frozen-`φ` (T3 under D6) | **Falsified** for this gate |
| Train `φ` on imagined futures (H1) fixes offset | **Falsified** (made it worse) |
| Phase A “`φ` alone is enough for hard goals” | **Weak / leaning false** |
| JEPA trunk is useless | **Not claimed** — the short win needs the stack; offset is low for *all* costs |
| Hybrid `L2 + α d_φ` (H2) adds anything | **Open** (eval-only, not run) |
| Receding horizon / policy residual / Sep would raise offset success | **Open** (system change, not run) |

---

## 7. Timeline (git)

Parent `ws_weltmodelle` and submodule `le-wm` both on `feat/lewm-phi`. Essentially all Phase A commits are **2026-08-28**.

| Parent | Submodule | What |
|--------|-----------|------|
| `7db8f52` | (init) | Workspace + le-wm submodule |
| `bcaa33f` | `26f3447`… | E0 baseline note; local eval instrumentation |
| … | `5186be1` `8f3bbcb` `7657070` `ef0b40e` | Reachability cost hook, eval flags, `train_phi`, live banks (no HF HDF5) |
| `4970648` | | v1 ablation table + pivot |
| `683c9ae` | `e2293dd` | train_phi CSV/plots |
| `8874100` | `4877609` | GPU + HWC→CHW fix |
| `2b76cfc`–`5126bc3` | `c8bef1d`–`487001a` | T3: IQE, IQL loss, sampler, `train_phi_iql`; eval no-go; D3 update |
| `4469528` `7f8d8bf` | `9aebc72` | Euclid multi-seed protocol + results |
| `2ba5e5c` `7259c78` | `eead006` `71ffd46` | Offset autopsy script + H1/H2 writeup |
| `409ffaf`–`d52a69b` | `7e0f305` `ccd05aa` | Imagined-φ trainer + multi-seed eval (offset fail) |
| **`c8668c0`** (HEAD) | | Status synthesis, experiment log, README, D3 refresh |

HEAD message: *Consolidate Phase A evidence and document next-step fork.*

**Working tree (2026-08-29), not a new experiment:** submodule has uncommitted C1 goal-cache hardening, episode-holdout pair splits in `phi_data` / `train_phi`, `collect-episodes` plumbing, extra tests, and untracked `scripts/diag_cost_scale.py`. Parent has untracked `lewm_phi_v2_summary.md` and `lewm_phi_v3_summary.md` (already referenced by other docs). None of this is fork A or B.

---

## 8. Artifact index

Checkpoints under `$STABLEWM_HOME` / `ws_le-wm/stablewm/checkpoints/pusht/` (gitignored). Eval under `le-wm/eval_results/pusht/`.

| Campaign | Summary | Weights | Eval root |
|----------|---------|---------|-----------|
| v1 Euclidean | `lewm_phi_v1_summary.md` | `lewm_phi/` (leaky) | `lewm_phi_v1/` |
| v2 Euclidean | `lewm_phi_v2_summary.md` | **`lewm_phi_v2/`** | `lewm_phi_v2_eval/` |
| v3 Euclidean | `lewm_phi_v3_summary.md` | `lewm_phi_v3_kin/`, `lewm_phi_v3_weak/` | `lewm_phi_v3_eval/` |
| IQL T3 | `lewm_phi_iql_v1_summary.md` | `lewm_phi_iql_v1/` | `lewm_phi_iql_v1/` |
| Multi-seed Euclid | `lewm_phi_euclid_multiseed_summary.md` | v2 | `lewm_phi_euclid_multiseed/` |
| Offset autopsy | `lewm_phi_offset_autopsy_summary.md` | v2 | `offset_autopsy/` |
| Imagined-φ | `lewm_phi_imagined_v1_summary.md` | `lewm_phi_imagined_v1/` | `lewm_phi_imagined_v1_eval/` |

Also: `lewm_phi_fixed/` (6k-step held-out train that did not beat L2).

Code entrypoints for reproduction are in the campaign summaries (typical pattern: `python train_phi.py …` then `python eval_live.py --plan-cost phi_d --phi-weights …`).

---

## 9. Next-step fork (superseded)

Historical A/B from `07` is **not** the Phase B plan. See [`09_phase_b_plan.md`](09_phase_b_plan.md): diagnose geometric hardness vs predictor drift at CEM horizon vs linear state decodability; Option A (hybrid cost) is optional bookkeeping only; do not pre-select hierarchy or Sep.

`short_horizon` vs `offset` in this report are **pose-band** filters on the same 25-step window, not imagination length. Autopsy ‖ẑ−z‖ ≈ 6.3 is **25-step** open-loop drift; CEM already uses `horizon=5`.

D6 remains in force until the B1 **state-factor** gate in `09`. Env claim remains PushT; Reacher is a Phase B diagnostic env.

---

## 10. How to read the rest of the docs

| Want | Read |
|------|------|
| What is locked | `00_decisions.md` |
| Full architecture / losses / E0–E4 definition | `01_design_spec.md` |
| Original engineering plan | `02_implementation_plan.md` |
| IQL equations and D6 adaptation | `03_quasimetric_iql_t3.md` |
| Compact results | `07_status_synthesis.md` |
| Phase B diagnose-first plan | `09_phase_b_plan.md` |
| Dated one-liners | `experiment_log.md` |
| Raw campaign tables | `lewm_phi_*_summary.md` |
| Historical brainstorm | `drafts.md` (not normative) |
