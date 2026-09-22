# FurnitureBench intermediate-state bank

The state bank is collected live from an RL-expert rollout. It does not derive
states from the historical main-experiment pickles. Every record uses schema
`rr-furniturebench-state-v2` and stores:

- the full per-actor Isaac Gym root state, including linear and angular velocity;
- all Franka DOF positions and velocities;
- reward/reset runtime state and RNG state;
- scripted semantic skill/stage, active part, optional legacy part-FSM audit
  state, assembly step, visit index, and zero-based frame offset from the start
  of that skill;
- the RL-expert checkpoint path/hash and the exact collection command;
- the scripted geometry annotation, camera calibration, annotation-FSM state,
  and optional raw wrist/front RGB previews.

Collection is exposed by `src.eval.evaluate_model`. Use a new output directory
for every campaign and pass scripted provenance explicitly:

```bash
/home/hy/anaconda3/envs/rr/bin/python -m src.eval.evaluate_model \
  --wt-path checkpoints/rppo/one_leg/low/actor_chkpt.pt \
  --gpu 0 --task one_leg --n-envs 1 --n-rollouts 1 --seed 310001 \
  --randomness low --max-rollout-steps 700 --action-type pos \
  --observation-space image --annotate-skill --enable-annotation-verify \
  --annotation-source scripted \
  --state-bank-out-dir logs/fb-state-bank-smoke/one_leg \
  --state-bank-skill-offsets 0 8 16 32 --state-bank-stride 0
```

Run the restoration/contact capability probe on a selected record:

```bash
/home/hy/anaconda3/envs/rr/bin/python \
  scripts/evaluate_furniturebench_state_restore.py \
  --state logs/fb-state-bank-smoke/one_leg/states/STATE.pkl.gz \
  --gpu 0 --horizon 8 --annotation-source scripted
```

To inspect a restored pose in an interactive viewer, add `--visualize-state`.
Under Isaac Gym's GPU pipeline, tensor setters are only committed by a physics
step, so this mode advances exactly one static step to commit the state and
rebuild derived/contact data, then freezes the viewer. It must not be interpreted
as an exact zero-step physics checkpoint. The default path zeros actor/DOF
velocities to match the proposed fixed-state evaluation protocol; add
`--inspect-restore-velocity` to retain the recorded velocities for that commit
step.

```bash
/home/hy/anaconda3/envs/rr/bin/python \
  scripts/evaluate_furniturebench_state_restore.py \
  --state logs/fb-state-bank-smoke/one_leg/states/STATE.pkl.gz \
  --gpu 0 --annotation-source scripted --visualize-state \
  --inspect-seconds 600
```

The probe compares two full-velocity restores and two zero-velocity restores.
Isaac Gym does not expose the contact solver's warm-start/cache state. Therefore
the default probe takes one static physics step to rebuild derived articulated
and contact state, then reapplies the exact saved actor/DOF tensors. Use
`--contact-rebuild-steps 0` to measure raw tensor restoration. Tensor equality
immediately after restore is necessary but insufficient: grasp retention and
short-horizon repeatability are mandatory empirical gates.

## Current Isaac Gym capability result

The original `fb-state-bank-smoke-v2-20260922` records are invalid for physics
restoration and remain diagnostic evidence only. FurnitureBench refreshes its
rigid-body tensor after each step but not its separate actor-root tensor; the
old schema therefore saved reset-time furniture root poses. Schema v2 now
refreshes actor-root state immediately before every capture and requires the
`root_state_refreshed=true` certification. Old schema-v1 records fail closed.

The corrected 2026-09-22 RL-expert sample is under
`logs/fb-state-bank-grasp-recovery-v3-20260922/one_leg`. A paired probe used
`leg-top-place`, visit 0, offset 12 (rollout frame 127): the saved gripper width
was 30.145 mm and the active leg was moving at 0.114 m/s before the requested
zero-velocity restore.

Direct pose/DOF restoration retained the stable grasp without reconstruction:
after 60 close-hold physics steps and another 96 release-test steps, the gripper
width remained 30.134 mm and the leg moved less than 1 micrometer. The fallback
protocol--reapplying every furniture part's saved root pose and zero velocity
for 60 steps while the fingers settle, then releasing physics--also passed all
gates: minimum release width 30.133 mm, final bilateral finger contact, and
active-leg displacement about 2.5 micrometers. The report is
`logs/fb-state-bank-grasp-recovery-v3-20260922/one_leg/place-o12.grasp_recovery_probe.json`.

For stable grasps, direct zero-velocity restoration is therefore the preferred
protocol. Pinned settling remains an optional fallback and capability gate for
states that do not retain bilateral finger contact after direct restoration.
