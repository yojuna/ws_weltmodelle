# Implementation status — what is in the tree

**Date:** 2026-08-30  
**Repo:** `ws_weltmodelle` `feat/lewm-phi` · submodule `ws_le-wm/le-wm` same branch  
**Normative plan:** [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md) (CA0 first)  
**Runtime:** [`../../docs/docker_usage.md`](../../docs/docker_usage.md)  
**Numbers:** [`11_phase_b_report.md`](11_phase_b_report.md) (Phase B) · [`13_phase_c0_report.md`](13_phase_c0_report.md) (C0) · [`experiment_log.md`](experiment_log.md) (compact). D6 **keep**; [`00_decisions.md`](00_decisions.md) records that keep (not a flip).

This file records **what is in the tree**. C-alt (CA0 and later) is **not** started. Do not start C1.

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

## 5. Phase C-alt — not started

Spec: [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md). Viz: [`15_viz_toolkit_spec.md`](15_viz_toolkit_spec.md). Next experiment is **CA0** (`scripts/closed_loop_imagine.py` — not in the tree yet). Do not implement CA-train, C1, or the viz module from this file.
