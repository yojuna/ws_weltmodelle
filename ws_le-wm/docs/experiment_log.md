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
| B1/B2 runs | latent dump, probes, drift@h=5, CEM `T` sweep | **Pending** — D6 still held |
