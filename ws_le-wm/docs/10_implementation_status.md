# Implementation status — what is in the tree

**Date:** 2026-08-31  
**Repo:** `ws_weltmodelle` `feat/lewm-phi` · submodule `ws_le-wm/le-wm` same branch  
**Normative plan:** [`16_fidelity_retrain_plan.md`](16_fidelity_retrain_plan.md) (CONFIRMED_INFIDELITY + BLOCK_INFIDELITY; smear-structure done; Part B not started). C-alt localize spec: [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md).  
**Runtime:** [`../../docs/docker_usage.md`](../../docs/docker_usage.md)  
**Numbers:** [`11_phase_b_report.md`](11_phase_b_report.md) (Phase B) · [`13_phase_c0_report.md`](13_phase_c0_report.md) (C0) · [`experiment_log.md`](experiment_log.md) (compact) · [`16a_infidelity_investigation.md`](16a_infidelity_investigation.md) (fork + smear). D6 **keep**; [`00_decisions.md`](00_decisions.md) records that keep (not a flip).

This file records **what is in the tree**. Encoder-floor, A-confirm, block-moving eval, smear-structure, and the dump-driven viz toolkit are in; **CA-train / Part B and C1 are not**. Do not start C1.

---

## 1. Phase A (done, previously committed)

Frozen LeWM + thin `φ` on PushT, live eval, no author HDF5. Campaigns and artifacts are in `07` / `08` and the `lewm_phi_*_summary.md` files.

| Surface | Path |
|---------|------|
| Reach head | `le-wm/reachability.py`, `jepa.py` `criterion` / `get_cost` |
| Train φ | `train_phi.py`, `train_phi_imagined.py`, `train_phi_iql.py` |
| Live eval | `eval_live.py`, `eval_logging/` |
| Offset autopsy | `scripts/offset_autopsy.py` |

**Verdict (unchanged):** Euclidean `φ` v2 beats L2 on short hops; offset ~2–10% for every head; IQL-on-φ and imagined-φ failed their gates. Stop cost-head churn.

---

## 2. Phase B diagnostics — code landed, GPU runs done

Shared helpers in `le-wm/phase_b.py`: encode frames, open-loop imagine with true vs shuffled actions, per-step ‖ẑ−z‖₂, dump format `dump.npz` + `dump.meta.json` (`DUMP_VERSION=1`). History = 3 teacher-forced frames; CEM horizon marker = 5.

| Script | Role | Writes |
|--------|------|--------|
| `scripts/latent_dump.py` | Collect kinematic (PushT) or random-action (Reacher) bank; persist `z`, `z_hat`, `z_hat_shuf`, state factors, remaining `k`, remaining pose error | `eval_results/{env}/phase_b_dump/seed{N}/` |
| `scripts/latent_probe.py` | Episode-holdout Ridge vs MLP R² per factor + `remaining_k` (sanity only); `--intervene-live` along `block_x` / first factor | `probe_summary.json`, `probe_r2.png` |
| `scripts/predictor_drift.py` | Plot true vs shuffled drift; mark teacher-forced end and CEM h=5 | `drift_curve.png` + `drift_summary.json` |
| `scripts/run_horizon_sweep.sh` | E1 L2 only: PushT offset n=50 and Reacher `live_reset` n=20, `T ∈ {2,3,5,8}` | `eval_results/{env}/phase_b_horizon/` |
| `scripts/test_phase_b.py` | CPU tests for dump/shuffle/horizon plumbing | — |

`eval_live.py` CLI: `--horizon`, `--receding-horizon`, `--collect-episodes`.

**Not implemented (still gated):** Sep, FF-JEPA `G`, residual policy, new AdaLN, new cost heads, C1 actor.

### 2.2 Eval / data plumbing

- `eval_logging/pairs.py`: `EpisodeTraj.__len__` falls back to pixels; Reacher collector concatenates `qpos`/`qvel`/`finger_pos`/`target_pos` into `state`.
- `eval_logging/runner.py` + `jepa.py`: C1 goal cache uses an explicit pair key (`_forced_goal_cache_key`).
- `phi_data.py` + `train_phi.py`: hindsight pairs indexed as `(ep, t, k)`; train/val split by episode.
- `scripts/test_reachability.py`: covers the episode split.
- `scripts/diag_cost_scale.py`: CEM cost dynamic-range check (H2 bookkeeping; not a campaign).

### 2.3 Docker

See [`../../docs/docker_usage.md`](../../docs/docker_usage.md). Python is baked at `/opt/venv`. Named container `weltmodelle-lewm`. Do not recreate `ws_le-wm/le-wm/.venv`.

---

## 3. GPU campaign (2026-08-29) — done

Seed 0 dumps, probes (`--intervene-live`), drift, L2 `T` sweep. Headline: PushT **D6 keep**; offset **flat 4–6%** across T=2,3,5,8; Reacher T=8 **15%**. Full writeup: [`11_phase_b_report.md`](11_phase_b_report.md).

B4 (build the B3 module) was **not** started and is **superseded** by C0 → C-alt. Do not start Sep / `G` / residual policy from this file.

---

## 4. Phase C0 (2026-08-30) — code + GPU gate done

CLI: `latent_dump.py --action-mode diverse` / `--collector`; `pack_action_token` (tile, not zero-pad); `eval_live.py --actor {cem,cem_l2,goal_push,weak,oracle_replay} --oracle-bank`; `scripts/oracle_bank.py`; `scripts/oracle_imagine.py`; `latent_probe.py --effective-rank`. Numbers: [`13_phase_c0_report.md`](13_phase_c0_report.md). C0.3-redo → **Outcome B** (redirect C-alt). **Do not start C1.**

---

## 5. Phase C-alt dumps + viz toolkit (2026-08-30)

Spec: [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md). Viz **v3** (what landed): [`15_viz_toolkit_spec.md`](15_viz_toolkit_spec.md). Viz **v0** (gallery, superseded as the writeup): [`15_viz_toolkit_spec_v0.md`](15_viz_toolkit_spec_v0.md). **CA0 fork is JSON, not a figure.** Do not start CA-train or C1.

| Surface | Path |
|---------|------|
| Closed-loop imagine | `le-wm/phase_b.py` `imagine_closed_loop`; `scripts/closed_loop_imagine.py` (reads `thresholds.yaml`) |
| Contact heuristic | `phase_b.contact_events` (PushT radius 45 / wall 40); Reacher stub |
| CA1 scalars | `scripts/drift_by_event.py` |
| Viz core | `le-wm/viz.py` (`FigureResult`), `scripts/viz.py`, `scripts/test_viz.py` |
| Thresholds | `le-wm/thresholds.yaml` (CA0 / encoder-floor / B.eval cuts; do not retune) |
| CEM capture | `eval_live.py --capture-cem`; `eval_logging/cem_capture.py` (line-search + per-iter elites) |
| CA3 sweep | `scripts/intervene_sweep.py` |
| Diagnostic report | `scripts/report.py` → `eval_results/pusht/viz_report/seed0/diagnostic_report.html` (`viz_report.py` is a shim) |

**CA0 (seed 0, live-bank n=50):** `m=25` matches C0 (‖ẑ_end−z*‖ 8.228, toward 2%). Fork **CA0-INFIDELITY** (`m=1` teacher-force guard). Re-derived 2026-08-31 on median one-step `frac` → **CONFIRMED_INFIDELITY**, then **BLOCK_INFIDELITY**. Numbers: [`experiment_log.md`](experiment_log.md). Spec: [`16_fidelity_retrain_plan.md`](16_fidelity_retrain_plan.md).

**Viz v3:** A1/A2/B1 + report landed. Pairings in A1 (overlay + full-space curve + in-plane fraction), A3 (`probe_decompose`), D1 (TwoNN). B1 is cost-vs-dist + selected→oracle line-search + candidate PCA (no height map). B4 and C1-by-hardness deferred.

## 6. Encoder-floor / B.eval (2026-08-31)

Cuts frozen in `le-wm/thresholds.yaml` (`encoder_floor`, `infidelity_confirm`, `b_eval_block`, `b_eval_tercile`, `smear_structure`). Do not retune. **Part B not started.**

| Surface | Path |
|---------|------|
| Encoder floor | `scripts/encoder_floor.py`, `scripts/test_encoder_floor.py` |
| A-confirm | `scripts/infidelity_confirm.py`, `scripts/test_infidelity_confirm.py` |
| Block-moving pairs | `eval_logging/oracle_bank.py` `window_block_moving_pairs` |
| B.eval-block / tercile | `scripts/block_motion_eval.py` |
| Smear structure | `scripts/smear_structure.py`, `scripts/test_smear_structure.py` |

Artifacts (gitignored dumps): `eval_results/pusht/encoder_floor/seed0/`, `infidelity_confirm/`, `block_motion_eval/seed0/`, `smear_structure/seed0/`.
