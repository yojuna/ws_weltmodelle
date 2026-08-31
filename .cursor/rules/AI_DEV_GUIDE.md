# Developing on ws_weltmodelle with an AI coding assistant

Practical guide for Cursor (or any agent) on this workspace, and for keeping
`.cursor/rules/` sharp. Read once; skim the checklist before a session.

---

## 1. What the rules are and how they're organized

The agent starts every chat with zero memory of this project. `.cursor/rules/`
injects persistent context. Layout:

| File | Activation | Purpose |
|------|-----------|---------|
| `00-project-context.mdc` | **Always** | Identity, layout, research gates, working agreement. Kept small. |
| `05-research-discipline.mdc` | **Always** | JEPA-native taste, localize-before-swap, plain language. |
| `30-git.mdc` | **Always** | Parent vs submodule: where to edit, commit, and bump. |
| `31-git-hygiene.mdc` | **Always** | Logical commits; `exp/` `feat/` `tooling/` `fix/` (same name in both repos); PRs into `main` on both. |
| `40-docker-tooling.mdc` | **Always** | `run.sh`, image venv, how to actually execute code. |
| `10-python.mdc` | **Auto** (`**/*.py`) | Hydra vs argparse surfaces, D6, tensors, tests. |
| `20-experiments.mdc` | **Auto** (train/eval/scripts/config) | Entrypoints, checkpoints, protocol, reporting. |

Precedence is Team > Project (`.cursor/rules/`) > User rules. These affect
**Agent Chat only** — not Tab or inline Cmd/K.

**Token hygiene:** always-on files compete with the task on every turn. Keep
them tight. Push detail into glob-scoped files. Do not paste campaign numbers
into always-on rules — point at `ws_le-wm/docs/README.md`.

---

## 2. The single most important habit: rules grow from mistakes

Treat `.cursor/rules/` as a living log of corrections. Every time the agent
does something you undo, encode the fix. Commit rule changes like code.

Good triggers:

- Edited files at the workspace root as if this were `lucas-maes/le-wm`.
- Committed docs inside the submodule, or code only in the parent.
- Created `ws_le-wm/le-wm/.venv` or `pip install` on the host.
- Ran tests or `python` on the host instead of `docker/run.sh`.
- Used `docker compose run` / `down` instead of `./run.sh`.
- Added a third loss / EMA / stop-grad "to help" the trunk, or a new cost head.
- Started C1 / Part B / Sep without being asked.
- Hardcoded a hyperparameter; used `policy=…_object.ckpt`; retuned
  `thresholds.yaml` after seeing numbers.
- Broke column normalization or `frameskip → action_encoder.input_dim`.
- Mixed docs/docker/code in one commit, `git add -A`, or experimented on `main`.

---

## 3. Working with the agent on this repo

**Plan first, then diff.** For anything beyond a one-file tweak, ask for a
short plan and approve it before code is written.

**Give it the right files.** `@ws_le-wm/le-wm/jepa.py` `@reachability.py`
`@train_phi.py` when touching the model or `φ`. `@ws_le-wm/docs/00_decisions.md`
when scope is in doubt. The agent reasons from real code, not a guess of
upstream LeWM.

**Name the invariants.** Loss = MSE + SIGReg; D6 detach; embeddings `(B, T, D)`;
live bank not HDF5 for `φ`. Discipline: `05-research-discipline.mdc`. Specificity
beats "make it clean" and beats "be creative."

**Keep changes reversible and small.** Match surrounding style; prefer editing
existing files. Code in the submodule, writeups in `ws_le-wm/docs/`.

**Ask, don't guess.** Coordinate/units/shape/dtype/dataset-version ambiguity →
stop and ask.

**Plain language.** Ask the agent to answer in short, ordinary sentences so
you can discuss the same claim back. Jargon is for named quantities (`frac`,
`d_end`), not for style.

---

## 4. Reproducibility checklist (before you trust a number)

- [ ] Ran inside `docker/run.sh` (including CPU tests) — never host/system Python.
- [ ] Knobs from Hydra / argparse / `thresholds.yaml` — nothing new hardcoded.
- [ ] Seed threaded (`cfg.seed` or `--seed` + `torch.Generator`).
- [ ] Upstream train: resolved `config.yaml` in the run dir; WandB has config + SHA.
- [ ] Checkpoint path relative to `$STABLEWM_HOME` without `_object.ckpt`.
- [ ] Same `history_size` / `num_preds` / `frameskip` / CEM horizon as comparators.
- [ ] Bank or dataset named; submodule git SHA recorded; multi-seed if it matters.
- [ ] Did not retune a frozen cut in `thresholds.yaml` after seeing the metric.

---

## 5. Stack-specific pitfalls

- **Two git repos.** Parent `origin` is `yojuna/ws_weltmodelle` (branch `main`).
  Submodule `origin` is the `yojuna/le-wm` fork; `upstream` is `lucas-maes/le-wm`.
  Push topic branches (`exp/` `feat/` `tooling/` `fix/`, same name as the
  parent). Merge submodule PRs into **`origin/main`**, never into `upstream`.
  Bump the parent gitlink after the submodule PR lands. Local author is `yojuna` /
  `datamongeraami@gmail.com` with `~/.ssh/yojuna` (not the work identity).
- **Two Python surfaces.** `train.py`/`eval.py` = Hydra + HDF5. `train_phi.py` /
  `eval_live.py` / `scripts/` = argparse + live sim. φ training does not use
  the author PushT HDF5.
- **Runtime is the image.** Every `python` invocation is `cd docker && ./run.sh
  python …` — CPU tests included. `/opt/venv`; bind-mount `/workspace`. Package
  changes = Dockerfile rebuild. Never host/system Python.
- **Loss integrity + D6.** Trunk = pred + SIGReg. `L_reach` stops at `φ`.
- **Gated work.** C1, CA-train/Part B, Sep, new cost heads — only when
  explicitly launched. Normative: `00_decisions.md`.
- **Checkpoint flavors.** `_object.ckpt` vs weights. Eval `policy=` has no suffix.
- **GL.** EGL offscreen on NVIDIA; GLFW on host Intel X. No NVIDIA GLX force.

---

## 6. Extending the rules later

- New file types (CUDA, ROS, hardware): add a scoped `.mdc` with `globs:`.
  Actuator-safety is omitted — this is sim-only.
- Keep one concern per file, under a few hundred lines. Split when a file
  covers unrelated topics.
- If you also use Claude Code / Copilot, a root `AGENTS.md` can be the portable
  source of truth; keep `.mdc` for Cursor glob scoping only.
