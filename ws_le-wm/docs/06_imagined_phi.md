# Imagined-φ transfer fix (H1)

**Date opened:** 2026-08-28  
**Trigger:** Offset autopsy (`05_offset_autopsy.md` / `lewm_phi_offset_autopsy_summary.md`) — primary **H1**, secondary **H2**.  
**Parent log:** [`experiment_log.md`](experiment_log.md)

## Claim under test

Training `φ` only on **real** encoded pairs leaves CEM scoring **imagined** `ẑ` OOD.  
If we regress temporal `k` using **predictor futures**, offset planning should improve vs `lewm_phi_v2`.

## Method (single fix — H1 only)

Frozen `hf_pusht`. Euclidean `ReachabilityHead` (same as v2).

For hindsight `(t, k)` with `t ≥ HISTORY−1`:

1. Encode true history `z_{t−H+1:t}` from pixels.  
2. Roll out **true bank actions** for `k` steps → `ẑ_{t+k}`.  
3. `z_t = encode(pixels_t)` (real; matches CEM’s real start).  
4. Loss: `Huber( ‖φ(z_t) − φ(ẑ_{t+k})‖₂ , k )` with stop-grad into trunk.  
5. **Mix:** with probability `real_frac` (default **0.25**), use real `z_{t+k}` instead of `ẑ` (stability / keep short-horizon skill).

Defaults: weak or kinematic bank (match v2 scale: prefer **weak ~24k steps** or kinematic 256), `k_max=25`, epochs 15, episode holdout val.

**Checkpoint dir (new):** `stablewm/checkpoints/pusht/lewm_phi_imagined_v1/`  
**Do not overwrite** `lewm_phi_v2`.

## Eval gate (after train)

Multi-seed **0,1,2**, same as `04_euclid_multiseed.md`:

| Protocol | Conditions |
|----------|------------|
| short n=20 | E1 L2 · E2_v2 · **E2_imagined** · E4 random |
| offset n=50 | same |

**Go:** mean offset success(E2_imagined) > mean(E2_v2) **and** ≥ mean(E1); ideally ≥ E4.  
Short should not collapse below v2.

Also re-run a **lite autopsy** (n_pairs=12) on imagined weights: expect lower imagination end-gap / better φ cost std vs v2.

## Explicit non-goals

- No IQL / IQE in this round  
- No hybrid `L2+αφ` yet (H2) — only if imagined-φ fails gate  
- No Sep / unfreeze trunk  

## Checklist

- [x] `phi_imagined_data.py` + `train_phi_imagined.py`
- [x] Train `lewm_phi_imagined_v1` + metrics/curves/console (best val corr **0.747**)
- [x] Multi-seed short+offset eval incl. E2_imagined — **offset gate FAIL**
- [ ] Lite autopsy on new weights (optional; planning next = H2)
- [x] `lewm_phi_imagined_v1_summary.md` + update `experiment_log.md`
