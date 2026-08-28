# Training data for lewm-phi v1

**Decision:** do **not** use the HuggingFace PushT HDF5 / Lance expert dataset.

`train_phi.py` builds a live `TrajectoryBank` inside `swm/PushT-v1` using the
same collectors as `eval_live` (`weak`, `kinematic`, or `goal`), then samples
hindsight temporal pairs from those rollouts.

Eval still uses the pretrained HF **checkpoint** (`hf_pusht` weights only).
