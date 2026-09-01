# Data collection tools

This directory keeps maintained collection and camera-contract entry points.

## FurnitureBench

The two-stage FurnitureBench pipeline is derived from
`gpu-snatcher/auto_data_preparation.sh`, but exposes the dataset-contract
choices directly.

## Contract

1. Collection saves unmodified wrist/front RGB-D frames. It records `skill`,
   3-D guidance geometry, `guidance_point_2d`, and camera calibration as
   metadata. Collection must never pass `--guidance-point-on-image`,
   `--grasp-annotation-on-image`, `--grasp-part-annotate`, or
   `--skill-on-image`.
2. `--annotation-source` is required for every collection launch. The current
   campaign is fixed to `scripted`/geometry GT and must not use VLM.
3. Image markers are rendered only while converting pickle to LMDB. The
   conversion requires an explicit `--image-annotation-mode`, records that
   value in LMDB metadata, and does not modify the source pickle.
4. Use a new `--output-suffix` for each campaign. The collector refuses to mix
   with existing pickles unless `--allow-existing-output` is given explicitly.

Run the collector from the host's FurnitureBench `rr` environment:

Both entry points automatically prepend that environment's `lib` directory to
the child `LD_LIBRARY_PATH`, as required by the project imports and Isaac Gym's
Python binding.

```bash
python scripts/data_collection/collect_furniturebench.py \
  --tasks one_leg round_table lamp \
  --target-successes 200 \
  --annotation-source scripted \
  --output-suffix rgbd-skill-guidance-metadata-0823 \
  --gpu 0 \
  --n-envs 4
```

Convert the clean pickles and draw the saved 2-D points into the LMDB images:

```bash
python scripts/data_collection/process_furniturebench_pickles_to_lmdb.py \
  --tasks one_leg round_table lamp \
  --input-suffix rgbd-skill-guidance-metadata-0823 \
  --output-suffix rgbd-skill-point-0823 \
  --image-annotation-mode guidance-point \
  --episodes-per-task 200
```

Pass `--dry-run` to either command to inspect the exact underlying command.

## ManiSkill front-camera tuning

`debug_maniskill_camera.py` replays one deterministic successful PPO recording
in the SAPIEN viewer. It applies the saved Panda qpos and task-part pose at each
recorded frame instead of attempting a new policy rollout, so local dynamics
drift cannot turn the camera check into a failed episode. It does not collect a
dataset or draw into raw images. The moving red wireframe box is the saved
scripted/geometry-GT target point transformed back to world coordinates at each
frame.

Controls are `P` pause/resume, `R` restart, `C` confirm/save, and `Q` quit.
Use the mouse to orbit, pan, and zoom. The script overrides SAPIEN's unsuitable
0.5-metre wheel default with `--scroll-speed 0.02` (2 cm per notch); hold Shift
for another 0.1x multiplier (2 mm per notch). SAPIEN otherwise resets its
interactive viewer to a 90-degree FOV. The tool therefore reapplies the dataset
FOV after scene attachment, locks it while saving, and uses an 800x800 square
viewport by default. The complete GUI viewport then represents the same aspect
ratio and FOV as the final 224x224 front image. The saved JSON records the
actual viewer pose, locked contract FOV, scroll speed, trajectory/checkpoint
hashes, and proposed `RR_FRONT_*` constants.
Pass `--apply-on-confirm` only when the same `C` keypress should also update the
specified `camera_contract.py`.

Run the interactive camera check on the r218 desktop with the configured local
`rr-maniskill` environment. Reuse this environment for subsequent local
ManiSkill diagnostics; do not create a task-specific environment per run:

```bash
DISPLAY=:1 \
XAUTHORITY=/run/user/1000/gdm/Xauthority \
PYTHONNOUSERSITE=1 \
/home/hy/anaconda3/envs/rr-maniskill/bin/python \
  scripts/data_collection/debug_maniskill_camera.py \
  --trajectory logs/joint-training-full-0821/videos/lift_peg_upright_camera_20260829_v1/pickles/LiftPegUpright-v1-seed824102.pkl.xz \
  --checkpoint logs/joint-training-full-0821/assets/maniskill/LiftPegUpright-v1/ppo_pd_ee_delta_pose_ckpt.pt \
  --output logs/joint-training-full-0821/local-camera-debug/lift_peg_upright_camera_view_20260831.json \
  --camera-contract /data/hy/ManiSkill/mani_skill/trajectory/pickle/camera_contract.py
```

The default CPU PhysX backend is sufficient because recorded-state playback
does not step the simulator. SAPIEN still needs the local 3060 to remain visible
for Vulkan rendering, so do not clear `CUDA_VISIBLE_DEVICES` for this GUI. The
replay always rejects trajectories whose top-level `annotation_source` is not
`scripted`. The checkpoint is hashed as policy provenance but is not executed.
A single replay is only a semantic and framing inspection. It must not be used
to claim task-level camera coverage. Production readiness is decided from fresh
multi-seed task-success rollouts whose pre-validation records retain the skill,
3D target, projected 2D target, camera calibration, and strict-gate result even
when the trajectory is rejected.
Only add `--apply-on-confirm` after the proposal is explicitly approved; rerun
the multi-seed pre-validation audit before treating it as a production camera.

The local environment is host-specific and must not be substituted for the NAS
Conda environment used by 4090-server collection jobs.

## AutoMate shared front-camera tuning

`debug_automate_camera.py` provides the same recorded-state camera-tuning
workflow for Isaac Sim. AutoMate production contains 99 tasks: assembly
`00755` is explicitly excluded and must not be used for collection, camera
tuning, quota calculations, or manifests. The tuning sample is the retained
task `00410` hardest-init success (43 transitions). `AssemblyCameraCfg.front`
is shared by all 99 retained assemblies, so one approved pose is written to
that shared config and then checked with a multi-task projection gate.

Run on the r218 desktop with the existing local `rr-isaaclab` environment and
3060; do not create another environment:

```bash
DISPLAY=:1 \
XAUTHORITY=/run/user/1000/gdm/Xauthority \
OMNI_KIT_ACCEPT_EULA=YES \
/home/hy/anaconda3/envs/rr-isaaclab/bin/python \
  scripts/data_collection/debug_automate_camera.py \
  --trajectory logs/joint-training-full-0821/videos/automate_low_success_review_20260831_v1/source_pickles/00410.pkl.xz \
  --asset-root logs/joint-training-full-0821/automate_camera_debug_20260901_v1/assets \
  --output logs/joint-training-full-0821/automate_camera_debug_20260901_v1/front_camera_proposal.json \
  --device cuda:0
```

Use the mouse to orbit, pan, and dolly. Press comma/period for 1-cm fine
backward/forward movement, `P` to pause/resume, `R` to restart, `C` to save the
shared-camera proposal, and `Q` to quit without saving. Saving a proposal does
not itself prove coverage; apply it to `AssemblyCameraCfg.front` only after
review, then rerun the 99-task camera/projection validation.

If the Isaac viewport cannot be manipulated reliably, review an explicit
candidate through the real dataset sensor path instead. `--render-video`
replays the saved states once, renders the front sensor at 320x240, performs
the production 224x224 center crop, and writes three panels: recorded old raw,
candidate raw, and candidate with the standard offline 2-pixel guidance point.
It does not modify `AssemblyCameraCfg.front`.

The user approved v2 on 2026-09-01. It moves the camera 15 cm toward the task
and 8 cm upward relative to the original pose while retaining the existing
orientation and intrinsics. The shared source config is now
`pos=(1.05, 0.0, 0.315)` with OpenGL quaternion
`(0.5434064844747748, 0.4524482209388897, 0.45244822093888976,
0.5434064844747747)`. The local and 236 config SHA256 is
`52b3204b8f07346ca04aa2e3566200271e03228db703d7011f6b6f80b405124f`.
The command below reproduces the reviewed comparison video; it is no longer a
proposal to apply:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/hy/anaconda3/envs/rr-isaaclab/bin/python \
  scripts/data_collection/debug_automate_camera.py \
  --trajectory logs/joint-training-full-0821/videos/automate_low_success_review_20260831_v1/source_pickles/00410.pkl.xz \
  --asset-root logs/joint-training-full-0821/automate_camera_debug_20260901_v1/assets \
  --output logs/joint-training-full-0821/automate_front_camera_candidate_20260901_v2/unused_proposal.json \
  --front-pos 1.05 0.0 0.315 \
  --render-video logs/joint-training-full-0821/automate_front_camera_candidate_20260901_v2/00410-old_vs_candidate-x1p05-z0p315.mp4 \
  --playback-fps 5 \
  --headless \
  --device cuda:0
```

The no-override regression after applying the shared source config is
`logs/joint-training-full-0821/automate_front_camera_shared_applied_20260901_v1/00410-shared-v2-applied.mp4`:
44/44 frames have a visible scripted front point. Formal AutoMate collection
must pass `--annotation-source scripted`, must not pass `--enable-sbc`, and the
collector rejects excluded assembly `00755` before Isaac Sim starts.
