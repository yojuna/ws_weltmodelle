# Implementation status — Phase A code + Phase B diagnostics

**Date:** 2026-08-29  
**Repo:** `ws_weltmodelle` `feat/lewm-phi` · submodule `ws_le-wm/le-wm` same branch  
**Normative plan:** [`09_phase_b_plan.md`](09_phase_b_plan.md)  
**Runtime:** [`../../docker/README.md`](../../docker/README.md)

This file records **what is in the tree**, not new experimental numbers. B1/B2 dumps, probes, and the CEM `T` sweep have **not been run** yet. Do not treat this as a D6 or hierarchy flip.

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

## 2. Landed this round (code; GPU runs pending)

### 2.1 Phase B diagnostics (`09` B1/B2)

Shared helpers in `le-wm/phase_b.py`: encode frames, open-loop imagine with true vs shuffled actions, per-step ‖ẑ−z‖₂, dump format `dump.npz` + `dump.meta.json` (`DUMP_VERSION=1`). History = 3 teacher-forced frames; CEM horizon marker = 5.

| Script | Role | Writes |
|--------|------|--------|
| `scripts/latent_dump.py` | Collect kinematic (PushT) or random-action (Reacher) bank; persist `z`, `z_hat`, `z_hat_shuf`, state factors, remaining `k`, remaining pose error | `eval_results/{env}/phase_b_dump/seed{N}/` |
| `scripts/latent_probe.py` | Episode-holdout Ridge vs MLP R² per factor + `remaining_k` (sanity only); offline (and optional live) intervention along `block_x` / first factor | `probe_summary.json`, `probe_r2.png` |
| `scripts/predictor_drift.py` | Plot true vs shuffled drift; mark teacher-forced end and CEM h=5 | `drift_vs_h.png` + JSON |
| `scripts/run_horizon_sweep.sh` | E1 L2 only: PushT offset n=50 and Reacher `live_reset` n=20, `T ∈ {2,3,5,8}` | `eval_results/{env}/phase_b_horizon/` |
| `scripts/test_phase_b.py` | CPU tests for dump/shuffle/horizon plumbing | — |

`eval_live.py` CLI: `--horizon`, `--receding-horizon`, `--collect-episodes` (kinematic bank size when short_horizon is pair-starved).

**Not implemented (gated by B3):** Sep, FF-JEPA `G`, residual policy, new AdaLN, new cost heads.

### 2.2 Eval / data plumbing (needed for dumps and honest φ val)

- `eval_logging/pairs.py`: `EpisodeTraj.__len__` falls back to pixels; Reacher collector concatenates `qpos`/`qvel`/`finger_pos`/`target_pos` into `state`.
- `eval_logging/runner.py` + `jepa.py`: C1 goal cache uses an explicit pair key (`_forced_goal_cache_key`) instead of a fragile pixel checksum (CEM only expands tensors).
- `phi_data.py` + `train_phi.py`: hindsight pairs indexed as `(ep, t, k)`; **train/val split by episode**, not a second RNG draw from the same bank.
- `scripts/test_reachability.py`: covers the episode split.
- `scripts/diag_cost_scale.py`: CEM cost dynamic-range check (H2 bookkeeping; not a campaign).

### 2.3 Docker (workspace `docker/`)

Python is **baked into the image** at `/opt/venv` (torch cu126, `stable-worldmodel[train,env]`, scikit-learn, matplotlib). Entrypoint is env-only (`PATH`, `STABLEWM_HOME`, MuJoCo/X11). Named container `weltmodelle-lewm` with `restart: unless-stopped`. Workspace bind `..:/workspace` is the source of truth for code, `$STABLEWM_HOME`, and `eval_results`. No named Docker volumes. Prefer `./run.sh stop`/`start` over `compose down`.

Host `ws_le-wm/le-wm/.venv` is not used. Do not recreate it.

---

## 3. Still to run (after the image is up)

Commands in `09` §12, from `/workspace/ws_le-wm/le-wm` inside the container:

```bash
python scripts/test_phase_b.py
python scripts/test_reachability.py
python scripts/latent_dump.py --env pusht --seed 0 --device cuda
python scripts/latent_dump.py --env reacher --seed 0 --device cuda
python scripts/latent_probe.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz
python scripts/latent_probe.py --dump eval_results/reacher/phase_b_dump/seed0/dump.npz
python scripts/predictor_drift.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz
bash scripts/run_horizon_sweep.sh
```

Then record B1/B2 numbers in [`experiment_log.md`](experiment_log.md) and apply the D6 / hierarchy triggers in `09` §6. Until those numbers exist, D6 stays **held**.
