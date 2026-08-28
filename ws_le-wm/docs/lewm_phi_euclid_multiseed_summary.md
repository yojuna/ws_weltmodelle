# Euclidean φ multi-seed results

**Aggregate:** generated from `lewm_phi_euclid_multiseed/aggregate.json`

## Go / no-go

| Check | Result |
|-------|--------|
| Short E2 > E1 (replicate) | PASS |
| Short E2 ≥ E4 (vs random) | PASS |
| Offset E2 ≥ E1 (transfer) | PASS |

## short

| Cond | success % (mean±std) | mean min pose | mean min state |
|------|----------------------|---------------|----------------|
| E1_l2_c1 | 16.7±7.6 | 28.09 | 101.1 |
| E2_phi_v2 | 36.7±7.6 | 24.35 | 94.5 |
| E4_random | 25.0±8.7 | 27.16 | 111.0 |

### Per seed

| Cond | seed0 | seed1 | seed2 |
|------|-------|-------|-------|
| E1_l2_c1 | 25% (5/20) | 15% (3/20) | 10% (2/20) |
| E2_phi_v2 | 35% (7/20) | 30% (6/20) | 45% (9/20) |
| E4_random | 35% (7/20) | 20% (4/20) | 20% (4/20) |

## offset

| Cond | success % (mean±std) | mean min pose | mean min state |
|------|----------------------|---------------|----------------|
| E1_l2_c1 | 4.7±3.1 | 47.25 | 151.0 |
| E2_phi_v2 | 6.7±1.2 | 44.78 | 129.3 |
| E4_random | 7.3±2.3 | 47.99 | 166.4 |

### Per seed

| Cond | seed0 | seed1 | seed2 |
|------|-------|-------|-------|
| E1_l2_c1 | 8% (4/50) | 4% (2/50) | 2% (1/50) |
| E2_phi_v2 | 6% (3/50) | 8% (4/50) | 6% (3/50) |
| E4_random | 10% (5/50) | 6% (3/50) | 6% (3/50) |

