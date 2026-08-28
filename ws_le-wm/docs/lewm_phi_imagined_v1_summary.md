# Imagined-φ (H1) multi-seed results

Weights: `lewm_phi_imagined_v1` vs `lewm_phi_v2` / L2 / random. Protocol: `06_imagined_phi.md`.

## Go / no-go

| Check | Result |
|-------|--------|
| Offset imagined > v2 | FAIL |
| Offset imagined ≥ E1 | FAIL |
| Short imagined ≥ E1 | PASS |

## short

| Cond | success % (mean±std) | mean min pose | mean min state |
|------|----------------------|---------------|----------------|
| E1_l2_c1 | 16.7±7.6 | 28.09 | 101.1 |
| E2_phi_v2 | 36.7±7.6 | 24.35 | 94.5 |
| E2_phi_imagined | 31.7±10.4 | 24.76 | 99.8 |
| E4_random | 26.7±2.9 | 26.35 | 104.9 |

### Per seed

| Cond | seed0 | seed1 | seed2 |
|------|-------|-------|-------|
| E1_l2_c1 | 25% (5/20) | 15% (3/20) | 10% (2/20) |
| E2_phi_v2 | 35% (7/20) | 30% (6/20) | 45% (9/20) |
| E2_phi_imagined | 20% (4/20) | 35% (7/20) | 40% (8/20) |
| E4_random | 25% (5/20) | 25% (5/20) | 30% (6/20) |

## offset

| Cond | success % (mean±std) | mean min pose | mean min state |
|------|----------------------|---------------|----------------|
| E1_l2_c1 | 4.7±3.1 | 47.25 | 151.0 |
| E2_phi_v2 | 6.7±1.2 | 44.78 | 129.3 |
| E2_phi_imagined | 2.0±0.0 | 50.31 | 154.9 |
| E4_random | 4.0±2.0 | 47.65 | 154.0 |

### Per seed

| Cond | seed0 | seed1 | seed2 |
|------|-------|-------|-------|
| E1_l2_c1 | 8% (4/50) | 4% (2/50) | 2% (1/50) |
| E2_phi_v2 | 6% (3/50) | 8% (4/50) | 6% (3/50) |
| E2_phi_imagined | 2% (1/50) | 2% (1/50) | 2% (1/50) |
| E4_random | 2% (1/50) | 4% (2/50) | 6% (3/50) |

