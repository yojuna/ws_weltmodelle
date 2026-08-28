# Protocol T3 — Quasimetric IQL (paper-faithful)

**Status:** Implementation in progress  
**Date opened:** 2026-08-28  
**Trigger:** v3 offset n=50 pivot — hindsight `k` regression learns temporal structure but does not beat L2 CEM on the firm protocol (`lewm_phi_v3_summary.md`).  
**Depends on:** `00_decisions.md` (D3 flip), `01_design_spec.md` Protocol T3, `02_implementation_plan.md` Phase 5.

This document is the **normative reference** for the T3 code path. If brainstorm notes disagree, this file + the cited papers win.

---

## 1. Papers (source of truth)

| Role | Citation |
|------|----------|
| IQL value shaping for JEPA planning | Destrade et al., *Value-Guided Action Planning with JEPA World Models*, [arXiv:2601.00844](https://arxiv.org/abs/2601.00844) |
| Quasimetric head (IQE-sum) | Wang & Isola, *Interval Quasimetric Embeddings*, [arXiv:2211.15120](https://arxiv.org/abs/2211.15120) (cited in Destrade §3.2) |

Related (context only): Park et al. 2024b (Hilbert / distance-as-value); Kostrikov et al. 2021 (IQL).

---

## 2. Adaptation to our stack (locked)

Destrade’s best table entry is **VF_quasi** with **Sep**: train the state encoder with \(\mathcal{L}_{\mathrm{VF}}\) alone, then train the predictor with \(\mathcal{L}_{\mathrm{pred}}\).

Our locked **D6** keeps the pretrained LeWM trunk frozen. We follow Destrade Appendix 7.3’s *separate predictive and planning representations* pattern:

| Role | Module | Trained by |
|------|--------|------------|
| Dynamics / rollout | Frozen LeWM \(E, P\) | — (HF PushT ckpt) |
| Planning geometry | Thin \(\phi\) + IQE-sum | \(\mathcal{L}_{\mathrm{VF}}\) only |
| Stop-grad | \(z = E(\mathrm{pixels}).\mathrm{detach()}\) | \(\partial L_{\mathrm{VF}}/\partial E = 0\) |

```text
pixels → E (frozen) → z.detach() → φ → u ∈ R^{k×l} → d_IQE-sum → V = -d
                                                              ↓
                                                         L_VF (Eq. 1)
                                                              ↓
                                                         CEM cost = d
```

---

## 3. Equations (implement verbatim)

### 3.1 Value (Destrade §3.2 + IQE)

Paper Euclidean:

\[
V_\theta(s,g) = -\| \mathcal{E}_\theta(s) - \mathcal{E}_\theta(g) \|_2
\]

Ours:

\[
V_\phi(s,g) = -\, d_{\mathrm{IQE\text{-}sum}}\big(\phi(z_s),\,\phi(z_g)\big),
\quad z = E(\mathrm{pixels}).\mathrm{detach()}
\]

### 3.2 IQL expectile loss — Destrade Eq. (1)

\[
\mathcal{L}_{\mathrm{VF}}
=
\sum_n \sum_{t=0}^{T-1}
L_\tau^2\Big(
  -\mathbf{1}_{s_t \neq g_n}
  + \gamma\, V_{\bar\phi}(s_{t+1}, g_n)
  - V_\phi(s_t, g_n)
\Big)
\]

\[
L_\tau^2(x) = |\tau - \mathbf{1}_{x < 0}|\, x^2
\]

- \(\bar\phi\): stop-gradient on the bootstrap value (same net; **no** target EMA required by the paper formula).
- Immediate “reward” term is \(-\mathbf{1}_{s\neq g}\) (reaching cost \(C=\mathbf{1}_{s\neq g}\)).

**VF_quasi defaults** (Destrade Appendix 7.2): \(\gamma = 0.93\), \(\tau = 0.60\).

**Goals** (Destrade §3.2): mix of (1) last state of the trajectory, (2) random states from the training batch.

**Equality \(\mathbf{1}_{s\neq g}\):** exact identity on **stored frame indices** `(episode_id, t)`. Do **not** use pose tolerance or task success \(S\).

### 3.3 IQE-sum — Wang & Isola Eqs. (2)–(3)

Reshape \(\phi(z)\in\mathbb{R}^{d}\) → \(u\in\mathbb{R}^{k\times l}\) with \(k\cdot l=d\) (default \(d=64\Rightarrow k=8,l=8\)).

\[
d_i(u,v)=\Big|\bigcup_{j=1}^{l}[u_{ij},\max(u_{ij},v_{ij})]\Big|
\]

\[
d_{\mathrm{IQE\text{-}sum}}(u,v)=\sum_{i=1}^{k}d_i(u,v)
\]

Required properties: \(d(u,u)=0\), \(d\ge 0\), asymmetry allowed, no soft triangle penalties.

CEM planning cost = \(d_{\mathrm{IQE\text{-}sum}}\) (i.e. minimize \(-V\)).

---

## 4. File map

| Concern | Path |
|---------|------|
| IQE-sum | `le-wm/iqe.py` |
| Expectile / Eq. (1) | `le-wm/iql_loss.py` |
| \(\phi\) + distance mode | `le-wm/reachability.py` |
| Transition + goal sampler | `le-wm/phi_iql_data.py` |
| Trainer | `le-wm/train_phi_iql.py` |
| Eval attach | `le-wm/eval_setup.py` |
| Tests | `le-wm/scripts/test_iql_iqe.py` |
| Checkpoints | `stablewm/checkpoints/pusht/lewm_phi_iql_v1/` (**new**; never overwrite prior φ runs) |
| Eval artifacts | `le-wm/eval_results/pusht/lewm_phi_iql_v1/` |
| Results summary | `docs/lewm_phi_iql_v1_summary.md` (after eval) |

Keep `train_phi.py` (Huber-on-`k`) intact for regression ablations.

---

## 5. Training protocol

1. Load frozen `hf_pusht`; `requires_grad_(False)`.
2. Collect live bank — default **kinematic** (Destrade §5: highly suboptimal trajectories hurt IQL); `--collector weak` allowed.
3. Episode-held-out train/val (reuse `phi_data.split_episodes`).
4. Sample \((s_t, s_{t+1}, g)\) with terminal/random goal mix.
5. Encode under `torch.no_grad`; train \(\phi\) only; assert no trunk grads.
6. Log: loss, mean \(d\), mean \(V\), TD residual, fraction \(s=g\); `metrics.csv` + `training_curves.png`.

---

## 6. Eval gate

Same claim as design spec §11, new dirs only:

1. **short_horizon n=20** — continuity with v2 weak-go: E1 (`l2_z`) vs E2 (`phi_d` + IQL ckpt) vs E4 (random φ).
2. **offset n=50** — firm protocol: same trio.

**Go:** E2_iql > E1 on short_horizon **and** does not lose to L2 on offset n=50.  
Otherwise document pivot with TD-residual diagnostics.

---

## 7. Explicit non-goals

- No VCReg on \(\phi\) (Destrade: VF_VCReg hurts).
- No joint \(L_{\mathrm{pred}}+L_{\mathrm{VF}}\) on the trunk.
- No pose-tolerance / success \(S\) in the loss.
- No IQE-maxmean in v1 of T3 (start with IQE-sum).
- No unfreezing \(E\) in this milestone.

---

## 8. Decision control

When this ships:

1. Flip **D3** in `00_decisions.md` to quasimetric IQL (date + reason: v3 offset pivot).
2. Leave Euclidean Huber regression available as ablation via `train_phi.py` / `distance_mode=euclidean`.

---

## 9. Implementation checklist

- [x] `iqe.py` + property tests
- [x] `iql_loss.py` + expectile / TD tests
- [x] `ReachabilityHead` `iqe_sum` + `value = -d`
- [x] `phi_iql_data.py` (terminal/random goals, exact `s==g`)
- [x] `train_phi_iql.py` + logging under `lewm_phi_iql_v1/`
- [x] Eval attach loads `distance_mode`
- [ ] Matched E1/E2/E4 campaigns + `lewm_phi_iql_v1_summary.md`
- [ ] Update `00_decisions.md` D3
