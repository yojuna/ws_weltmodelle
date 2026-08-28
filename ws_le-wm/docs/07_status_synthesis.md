# Status synthesis — Phase A evidence (2026-08-28)

**Purpose:** Single place that consolidates what we measured, what it means for the v1 claim, and what is *not* settled.  
**Chronicle:** [`experiment_log.md`](experiment_log.md) · protocols `03`–`06`.

---

## Original claim (still the question)

> Frozen LeWM + thin `φ` + C1 goals beats L2-in-`z` CEM on PushT, without success rewards or trunk value-shaping (D6).

---

## Master results table (planning success %)

All rows: frozen `hf_pusht`, C1 goal cache, kinematic eval banks. Multi-seed = mean±std over seeds **0,1,2**.

### Short-horizon (n=20)

| Condition | Seeds | Success % | Notes |
|-----------|-------|-----------|--------|
| E1 L2 `z` | 0–2 | **16.7±7.6** | Baseline |
| E2 Euclidean v2 | 0–2 | **36.7±7.6** | **Replicated win** |
| E2 Imagined-φ | 0–2 | 31.7±10.4 | ≥ E1; &lt; v2 |
| E4 random `φ` | 0–2 | 26.7±2.9 | |
| E2 IQL (seed 0 only) | 0 | 10.0 | Worse than E1 & E4 |

### Offset (n=50) — firm protocol

| Condition | Seeds | Success % | Notes |
|-----------|-------|-----------|--------|
| E1 L2 `z` | 0–2 | **4.7±3.1** | Low absolute |
| E2 Euclidean v2 | 0–2 | **6.7±1.2** | Slightly &gt; E1; ≤ random historically |
| E4 random `φ` | 0–2* | ~4–7 | Often ties/beats trained |
| E2 Imagined-φ | 0–2 | **2.0±0.0** | **Worse** than v2 & E1 |
| E2 IQL (seed 0) | 0 | 4.0 | ≤ E1 |

\*Random offset mean depends on campaign; euclid multiseed had 7.3±2.3, imagined campaign 4.0±2.0 (different bank draws).

### Training diagnostics (not planning)

| Run | Best held-out signal | Path |
|-----|----------------------|------|
| Euclidean v2 | corr(d,k) ≈ 0.54 | `lewm_phi_v2/` |
| IQL T3 | val L_VF ≈ 0.035 | `lewm_phi_iql_v1/` |
| Imagined-φ | corr(d,k) ≈ **0.75** | `lewm_phi_imagined_v1/` |

**Lesson:** Higher train/val corr does **not** imply better offset CEM.

---

## Autopsy (why short ≠ offset)

| Finding | Evidence | Hypothesis |
|---------|----------|------------|
| Real-path ranking excellent | Spearman(d, remaining) ≈ 0.99 for L2/`φ`/random | Not H3/H4 |
| Imagined latents far from real | ‖ẑ−z‖₂ ≈ 6.3 on true-action rollouts | **H1** |
| `φ` CEM costs flat | candidate std φ≈0.48 vs L2≈6.3 | **H2** |

H1 fix (train on ẑ) **failed** planning gate despite better corr → seeing ẑ under temporal-`k` regression is not sufficient (and hurt offset).

---

## What is falsified vs open

| Statement | Status |
|-----------|--------|
| Thin Euclidean `φ` can beat L2 on **short** hops (multi-seed) | **Supported** |
| Same `φ` reliably beats L2 on **offset** | **Not supported** (tiny / noisy edge; random competitive) |
| Quasimetric IQL on frozen-`φ` (T3 under D6) | **Falsified** for our gate |
| Train `φ` on imagined futures (H1) fixes offset | **Falsified** |
| Phase A “`φ` alone is enough for hard goals” | **Weak / leaning false** |
| JEPA trunk useless | **Not claimed** — short win needs the stack; absolute offset success is low for *all* costs |

---

## Artifact index

| Campaign | Summary | Raw |
|----------|---------|-----|
| v1–v3 Euclidean | `lewm_phi_v{1,2,3}_summary.md` | `stablewm/checkpoints/pusht/lewm_phi*` |
| IQL T3 | `lewm_phi_iql_v1_summary.md` | `…/lewm_phi_iql_v1/`, `eval_results/…/lewm_phi_iql_v1/` |
| Multi-seed Euclidean | `lewm_phi_euclid_multiseed_summary.md` | `eval_results/…/lewm_phi_euclid_multiseed/` |
| Offset autopsy | `lewm_phi_offset_autopsy_summary.md` | `eval_results/…/offset_autopsy/` |
| Imagined-φ | `lewm_phi_imagined_v1_summary.md` | `…/lewm_phi_imagined_v1/`, `eval_results/…/lewm_phi_imagined_v1_eval/` |

---

## Decision fork (reconsideration)

Two coherent next moves — pick **one** research posture:

### A. Close the thin-`φ` cost loop (small, remaining autopsy item)

**H2 hybrid / normalize** at eval: `cost = L2 + α·d_φ` with **v2** weights.  
**Proves:** whether `φ` adds anything once L2 carries CEM dynamic range.  
**Does not prove:** high PushT success or robot readiness.  
**Effort:** low (no retrain).

### B. Declare Phase A cost-head largely capped; change the system

Treat offset ~5% for all heads as evidence that **open-loop CEM + frozen predictor + thin cost** is the bottleneck. Next experiments leave pure “better `d_φ`”:

- more aggressive **receding horizon / shorter imagination**, or  
- **policy / residual** on top of JEPA, or  
- **Sep** value encoder (explicit D6 exception), documented as a decision flip.

**Proves:** whether the *architecture* of planning—not the readout loss—is what fails.

**Recommendation:** If the goal is still “falsify thin-`φ` under D6,” do **A** once, then stop cost-head churn. If the goal is “make PushT actually work toward robots,” prefer **B** and treat A as optional bookkeeping.
