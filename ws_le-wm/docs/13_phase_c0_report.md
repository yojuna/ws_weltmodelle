# Phase C0 report — Confirmation gate

**Date:** 2026-08-30  
**Spec:** [`12_phase_c_plan.md`](12_phase_c_plan.md)  
**Code:** `le-wm/phase_b.py` packing + dump CLI; `eval_live.py --actor`; `scripts/oracle_imagine.py`; `latent_probe.py --effective-rank`  
**`00_decisions.md`:** D6 keep recorded (not a flip). D7 stays proposed. C1 not built; C-alt is the next plan ([`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md)), not built in this report.

---

## Gate (one paragraph)

C0.2 **passes**: on a **random `env.step`** PushT bank with CEM-matched `tile_block` action tokens, shuffle−true predicted-only drift gap is **1.77** (kinematic bank was **~0.002**, including after the same pack). `P` is not action-deaf. C0.1 **passes**: block pose linear R² and live `block_x` hit replicate on seeds 1–2. C0.3-as-run (heuristic on hard offset) was underpowered. **C0.3-redo** (live-bank oracle, [`12a_c03_redo.md`](12a_c03_redo.md)): oracle-replay **94%**, CEM-L2 **50%** (gap 44 pts), but Oracle-imagine toward-goal **2%** (‖ẑ_end−z\*‖ 8.23 vs ‖z_start−z\*‖ 2.61). **Outcome B — model fidelity.** Redirect to C-alt; do not build C1. Scope: reachable on-policy `short_horizon` futures, **not** the kinematic hard-offset band. D6 keep stands.

---

## Packing (load-bearing for C0.2)

Installed SWM CEM uses `action_dim = env_dim * action_block` (2×5=10). Box bounds are `tensor.repeat(action_block)` → `[ax, ay]` tiled five times. Execution reshapes `(H, 10) → (H*5, 2)`. Dump `pad_action` zero-filled dims 2–9 (OOD). C0 dumps use `pack_action_token` = `np.tile(a, 5)`. Dump imagination remains **per env-frame** (not CEM’s extra `n_steps+1` predict). Meta records `action_pack=tile_block`.

Kinematic seed-1 dump with the **same pack** still has shuffle gap **0.004** → the B2 0.002 number was the **bank**, not only packing.

---

## C0.2 — diverse-action liveness (first, highest risk)

Bank: PushT `collector=random` (live `env.step`, `from_state_step`), 40 eps, 48×26 segments. Artifact: `le-wm/eval_results/pusht/phase_b_dump_diverse/seed0/`.

| | predicted-only ‖ẑ−z‖ | at h=5 | end | shuffle−true predicted gap |
|--|--|--|--|--|
| Diverse / random | 5.26 | 2.94 | 8.45 | **1.773** |
| Kinematic (B2 seed 0) | 9.83 | 5.26 | 15.46 | **0.002** |

**Pass** (≥ 0.15 and ≥ 10% of true predicted-only). `P` responds to actions.

---

## C0.3 — oracle on the same kinematic offset pairs

Protocol matched to B2 T=5: `--collector kinematic --collect-episodes 200 --pair-mode offset --episodes 50 --seed 0 --eval_budget 50`. CEM L2 T=5 was **4% (2/50)**.

| Actor | Success | Notes |
|--|--|--|
| CEM L2 (B2) | **4%** (2/50) | T=5 |
| GoalPush | **8%** (4/50) | several “successes” at length 1–3 |
| WeakPolicy | **2%** (1/50) | one success at step 26 |

**Search-binds: fail / not shown.** Heuristics do not unlock offset. Artifacts: `eval_results/pusht/c0_oracle_{goal,weak}/`.

**C0.3b** GoalPush actions through `imagine_path` (tiled): n=50, env success 4, mean ‖ẑ_end−z\*‖=17.3 vs encoded true-end 15.6, fraction of imaginations that moved toward the goal **0.08**, model-fidelity fails among env successes **1/4**. Not “env works, `P` cannot represent.” (Superseded as a *gate* by the live-bank redo below.)

---

## C0.3-redo — live-bank oracle (seed 0)

Spec: [`12a_c03_redo.md`](12a_c03_redo.md). No HDF5. Bank: GoalPush then Weak, window `(t, t+25)` in the `short_horizon` band (pose ∈ [12, 25], angle ≤ 0.25), pack `tile_block`. Artifact: `eval_results/pusht/c0_oracle_livebank/seed0/`.

**Construction note.** Raw stride-25 hops on Weak/GoalPush have median pose change ~170, so they miss the band. Windows are found by scanning every start, then kept with non-overlap gap 25 (`scan_stride=1`). GoalPush alone yielded 4 windows; Weak filled to 66, subsampled to 50. Mean start→goal pose **22.4** (just above env pos-tol 20). Source `goal+weak`, 260 bank episodes.

| Arm | Result | Question |
|--|--|--|
| Oracle-replay | **94%** (47/50), budget 25 | Tautology (≥90% — pass; not broken) |
| CEM-L2 T=5 budget 50 | **50%** (25/50) | Search: oracle−CEM = **44 pts** (≥20) |
| Oracle-imagine | toward-goal **2%** (1/50); mean ‖ẑ_end−z\*‖ **8.23** vs ‖z_start−z\*‖ **2.61** | Can `P` represent the oracle? **No** |

Pre-registered Outcome A needed imagine ≥~60% toward-goal as well as the CEM gap. Imagine fails that cut, so the gate is **Outcome B (model fidelity)** even though search is also weak on these pairs. `d_true_end=0` in the json is because the last path frame *is* the goal; the load-bearing numbers are toward-goal and ‖ẑ_end−z\*‖.

**Scope (load-bearing):** this is a pass/fail **on reachable on-policy futures**, closer to paper `short_horizon` than the kinematic hard-offset band. Do not report it as a hard-offset result. Seeds 1–2 not run (seed 0 is decisive for B).

**Redirect:** C-alt (predictor action-conditioning / multi-step rollout). Keep D6. Do not build C1. `00` unchanged.

---

## C0.1 — seeds 1–2 + Reacher names

Kinematic dumps (comparable to B1). Live intervention on `block_x`. Pass: block_x ≳ 0.7, block_y ≳ 0.5, hit ≳ 0.8.

| Seed | block_x lin. | block_y lin. | live hit | mean state lin. |
|--|--|--|--|--|
| 0 (B1) | **0.90** | **0.78** | **1.0** | 0.70 |
| 1 | **0.93** | **0.86** | **1.0** | 0.45 |
| 2 | **0.67** | **0.96** | **1.0** | 0.72 |

Seed 2 `block_x` is slightly under 0.7; pose pair and hit still pass within noise. **D6 keep** not overturned.

Reacher columns (same seed-0 dump, renamed): `qpos_0` 0.56, `finger_y` 0.82, `finger_x` 0.41; `qpos_1` / qvel / target negative. Mean linear still **−0.34**. **Drop Reacher from the PushT legibility claim.** Diagnostic only.

---

## C0.4 — effective rank (not a blocker)

Seed-0 kinematic tokens, v2 `reach.pt`. Participation ratio of covariance eigenvalues:

| | dim | effective rank |
|--|--|--|
| `z` | 192 | **22.5** |
| `u=φ(z)` | 64 | **10.1** |
| rank_u / rank_z | | 0.45 |

`u` is thinner than `z` (consistent with a flat φ landscape) but not rank-1 collapse. Informs C3 if a learned cost returns. Artifact: `phase_b_dump/seed0/effective_rank.json`.

---

## Decision table

| Item | Result |
|--|--|
| D6 | **Keep / extract** (C0.1 replicate) |
| C-alt | **Redirect** (C0.3-redo Outcome B). Plan: [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md) (rollout fidelity / CA0 — not AdaLN). Not built in the C0 session; CA0 later reported INFIDELITY. |
| C1 actor | **Do not start** (Outcome B: `P` cannot imagine oracle reaching; C0.3-as-run was also underpowered). |
| D7 asymmetry | Proposed only; not written into `00` |
| Hierarchy / C2 | Still composition-only, untriggered |

---

## Non-goals honored

No policy prior, no new cost, no D6 flip, no Sep. D6 keep later recorded in `00` (cleanup, not a bet change).
