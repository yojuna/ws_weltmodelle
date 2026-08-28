# Experiment log — lewm-phi sequence

Running chronicle of tests (newest at bottom of each section). Normative protocols live in numbered `0x_*.md` docs.

---

## 2026-08-28 — Timeline

| Step | What | Outcome | Doc / artifacts |
|------|------|---------|-----------------|
| T1 Euclidean v1–v3 | Hindsight-`k` on frozen LeWM | Soft short win (v2); offset no-go (v3) | `lewm_phi_v{1,2,3}_summary.md` |
| T3 IQL | Quasimetric IQL on thin `φ` | No-go vs L2 (short+offset) | `03_…`, `lewm_phi_iql_v1_summary.md` |
| Multi-seed Euclidean | Seeds 0–2, short+offset, E1/E2_v2/E4 | Short **PASS** (36.7% vs 16.7%); offset weak (φ≈L2≤random) | `04_…`, `lewm_phi_euclid_multiseed_summary.md` |
| Offset autopsy A | Real ranking / imagination / CEM spread | **H1** + **H2**; not H3/H4 | `05_…`, `lewm_phi_offset_autopsy_summary.md` |
| **Next: imagined-φ** | Train `φ` on predictor futures (H1 fix) | **Train done** — best val corr **0.747** @ ep14 (`lewm_phi_imagined_v1/`); multi-seed eval running | `06_imagined_phi.md` |

---

## Locked reading of autopsy

- Real-path Spearman ~0.99 for L2 / `φ` / random → geometry on **real** `z` is fine.
- Mean ‖ẑ−z‖₂ ≈ 6.3 under true-action rollouts → CEM scores **OOD** latents.
- φ CEM candidate std ≪ L2 → flat landscape (H2).

**Chosen single fix:** H1 — train Euclidean `φ` with `d(φ(z_t), φ(ẑ_{t+k})) → k` (real start, **imagined** future). Optional light mix of real pairs for stability. H2 hybrid deferred until after this re-eval.
