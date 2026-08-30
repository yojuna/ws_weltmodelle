# Phase C addendum — C0.3-redo (live-bank oracle)

**Date:** 2026-08-30  
**Status:** ran 2026-08-30 → Outcome B. Results in [`13_phase_c0_report.md`](13_phase_c0_report.md).  
**Amends:** [`12_phase_c_plan.md`](12_phase_c_plan.md) §6 C0.3 and §7/§12.  
**Change control:** D6 keep recorded in `00` (not a flip). D7 stays proposed. C1 not built.  
**Reason for addendum:** C0.3-as-run tested a *mediocre heuristic* (GoalPush/Weak) on *hard interpolated* offset poses. Env success 8%/2% vs CEM-L2 4% is a **weak-instrument null**, not a search-binds refutation. The gate had not resolved in either direction. This addendum specifies a real oracle and the rule that resolved it.

---

## 0. Corrections folded in (record)

- **C1 row (rewrite in `13` decision table and `experiment_log.md`):**
  ~~"Do not start (C0.3 did not show oracle≫CEM)"~~ →
  **"Do not start until a real oracle is run; C0.3-as-run was underpowered (heuristic controller on hard interpolated poses)."** The not-building decision stands; only the *reason* changes.
- **C0.3b 8% figure:** it is over **all 50** GoalPush rollouts, **not** conditioned on the 4 env successes (among successes the fidelity cut failed 1/4, too thin to read). No imagination-fidelity signal is claimed from C0.3b. Do not cite the 8% as evidence about successful trajectories.
- **Drift magnitude is bank-specific.** Kinematic + corrected `tile_block` (seeds 1–2) is still ~9.8–10.2 predicted-only, same order as B2's 9.83 — so the packing fix changed action-**liveness** (the shuffle−true gap), **not** drift **magnitude**. Quote dump drift **by bank** (kinematic ~9.8–10.2; diverse/random 5.26); never present one as the other's "correction." B2's T-sweep used the real CEM path, so its qualitative verdict stands.

---

## 1. Why a live-bank oracle (no HDF5)

`data_source.md` is explicit: no HF PushT HDF5/Lance in this tree; eval is checkpoint-only, `eval_live` is the live stand-in. So the oracle is **not** downloaded expert demos. It is constructed live:

1. Roll a physics controller (Weak or GoalPush) in `swm/PushT-v1`, collect **its own successful/observed rollouts**.
2. Form `(t, t+25)` windows: start = state at `t`, goal = state actually reached at `t+25` **by that same rollout**.
3. The stored action chunk `a_{t:t+25}` is, **by construction**, a sequence that reaches the goal from the start in sim.

This is a real oracle *for those goals*: env success on replay is tautological, which is exactly what lets us ask the two questions CEM/`P` cannot otherwise be pinned on.

---

## 2. Scope (both sentences are load-bearing — keep together)

- **What the redo tests:** whether CEM fails to *find* actions that reach **reachable** goals, and whether `P` can *imagine* the oracle actions reaching them. A clean search-binds test on on-policy futures.
- **What the redo does NOT test:** it does **not** establish that a better actor unlocks the **kinematic hard-offset** band, which remains the program's hard claim. The `(t, t+25)` on-policy goals are **reachable futures, closer to `short_horizon`/paper-style than the kinematic offset band.** A search-binds *pass here* is a pass **on reachable goals only** and must never be reported as a hard-offset result.

---

## 3. Oracle construction (concrete)

For each seed in {0} (extend to {1,2} only if seed-0 is decisive):

1. **Collect** N≈60 successful/served rollouts with a controller `π_o ∈ {GoalPush, Weak}` via `eval_live` live `env.step`, logging per-step `state`, `pixels`, `action`.
2. **Window** each rollout into `(t, t+25)` pairs (stride 25, drop tails < 25). Keep pairs whose realized pose change lands in the **`short_horizon` band** (pose ∈ [12,25], angle ≤ 0.25) — matches paper-style reachable goals; record the band explicitly in meta.
3. **Goal** `z* = encode(pixels_{t+25})`; **start** = state/pixels at `t`; **oracle actions** = `a_{t:t+25}` (pack with `tile_block`, per C0 packing note).
4. Target n ≈ 50 pairs to match B2/C0.3 episode budget.

Record `oracle_source`, `pair_band=short_horizon`, `action_pack=tile_block` in meta.

---

## 4. Three arms on the **same** oracle pairs

| Arm | What runs | Question |
|--|--|--|
| **Oracle-replay** | Execute `a_{t:t+25}` in env from start | Sanity: env success ≈ 100% (tautology check; if not, the collector/window is broken) |
| **CEM-L2** | Same start/goal, CEM as in B2 (`T=5`, budget 50) | Can search *find* reaching actions? |
| **Oracle-imagine** | Roll `a_{t:t+25}` through `imagine_path` (tiled), measure ‖ẑ_end − z*‖ and per-step | Can `P` *represent* the oracle reaching the goal? |

All three on identical pairs so success/regret are directly comparable.

---

## 5. Two-outcome decision rule (this resolves the gate)

Let **oracle-replay env success** be the tautology check (expected high). Then:

- **Outcome A — SEARCH BINDS (build C1).**
  Oracle-replay succeeds in env **and** CEM-L2 success on the same pairs is materially lower (pre-register: oracle_env − CEM ≥ 20 pts, e.g. ~100% vs ≤80%), **and** Oracle-imagine lands near `z*` (≥ ~60% of pairs reduce ‖ẑ−z*‖ toward goal).
  → Good actions exist, `P` can imagine them, CEM can't find them. **The actor/search verdict is demonstrated (on reachable goals).** Proceed to C1 (§6 of `12`, asymmetric propose-and-score). Record the §2 scope caveat in the C1 claim.

- **Outcome B — MODEL FIDELITY (go to C-alt / predictor).**
  Oracle-replay succeeds in env **but** Oracle-imagine does **not** track the goal (large ‖ẑ_end−z*‖, low toward-goal fraction) on actions that demonstrably work in sim.
  → The predictor cannot represent reaching transitions it was *given* the correct actions for. **This is the clean model-fidelity fail C0.3b could not show** (real oracle, not a heuristic). Redirect to predictor action-conditioning / multi-step rollout (C-alt), keep D6.

- **Ambiguous guard.** If oracle-replay env success is itself low (< ~90%), the collector/window is broken (goals not actually reached) — fix the oracle before reading A vs B. Do **not** interpret a broken tautology as either outcome.

Pre-register the thresholds above **before** running so the gate can't be read post-hoc.

---

## 6. What does NOT change

- D6 **keep** (C0.1 replicated seeds 1–2). Unaffected by this redo.
- C0.2 liveness **pass** stands (bank-vs-packing isolated). Unaffected.
- C0.4 effective rank (`u` 10.1 vs `z` 22.5) — writeup line; `z`'s 22.5/192 noted as headroom caveat. Unaffected.
- Hierarchy / C2 untriggered. No new cost, no policy prior, no Sep until the gate resolves.

---

## 7. Immediate action (GPU)

Run in order; stop after the tautology check if it fails.

```bash
# 1) collect the oracle bank (controller's own successful rollouts), window (t, t+25) short_horizon
python scripts/oracle_bank.py --env pusht --controller goalpush \
  --collect 60 --window 25 --band short_horizon --pack tile_block \
  --seed 0 --out eval_results/pusht/c0_oracle_livebank/seed0/

# 2) three arms on the SAME pairs
python eval_live.py --env pusht --actor oracle_replay \
  --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0/ --episodes 50 --seed 0   # tautology check
python eval_live.py --env pusht --actor cem_l2 \
  --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0/ --episodes 50 --seed 0 --horizon 5
python scripts/oracle_imagine.py \
  --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0/ --pack tile_block

# 3) record outcome A / B / ambiguous in experiment_log.md; rewrite the C1 reason row per §0
```

Extend to seeds 1–2 only if seed-0 is decisive. Ratify D7 into `00` independently — it does not depend on this gate.
