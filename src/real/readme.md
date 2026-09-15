Steps for real world demo collection
- Set up a few things we use in the real world (our lab's "real deployment tools", `meshcat`, `polymetis`, `pyrealsense2`, etc.). Detailed setup instructions still needed here, but it's essentially using `conda` to install `polymetis`, using `pip` to install `meshcat` and `pyrealsense2`, and using `pip` to install some of our [RDT](https://github.com/anthonysimeonov/improbable_rdt) tools. 
- Set up the [Spacemouse](https://3dconnexion.com/us/product/spacemouse-wireless/). Packages to install + commands to run to make it work are below (borrowed from the [diffusion policy repo](https://github.com/real-stanford/diffusion_policy?tab=readme-ov-file)):
```
# Needed for spacemouse
pip install numpy termcolor atomics scipy
pip install git+https://github.com/cheng-chi/spnav
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd
```
- Run `teleop_sm.py`. Example usage:
```
python teleop_sm.py -p 6000 --save_dir teleop_data/one_leg_color --furniture one_leg
```
- `-p` indicates what port to use for `meshcat` visualization. Make sure you have run `meshcat-server` in a background terminal (after `pip install meshcat`), and that the port that gets printed out matches what you use with `-p`

Steps for real world eval
- Run `minimal.py`. Example usage:
```
python minimal.py -p 6000 --run-id real-one_leg-cotrain-2/paxnbwsu # -w _1199.pt
```
- `--run-id` corresponds to the `wandb` run you want to evaluate
- `-w` indicates which specific checkpoint you may want to test out (optional - by default it uses the `best_test_loss` or `best_success_rate` checkpoint)

The Polymetis path above is legacy. New Deoxys RGBD evaluation and timestamp
alignment are documented in `reports/real_time_alignment_and_eval.md`; use
`python -m src.real.evaluate_policy` for the timestamped absolute-pose path.

### FrankaControl full-frame RGB-D eval (240x320)

The six deployed checkpoints are under
`checkpoints/rr_real_sim_fullframe_cotrain_0907`. They embed
`observation_type=rgbd`, `control.control_mode=pos`, and
`data.image_spatial_transform=none`. The evaluator records canonical 240x320
front/wrist RGB-D observations, preserves that spatial shape through both actor
camera transforms, and rejects a non-240x320 input instead of silently resizing
it.

Copy the complete block below. Change only `RR_RUN` and `RR_EPOCH`; supported
runs are `real40`, `real40_sim400`, and `real10_sim400`, and supported epochs
are `3000` and `5000`.

```bash
cd /home/hz/code/robust-rearrangement-custom

RR_RUN=real40
RR_EPOCH=3000
RR_PYTHON=/home/hz/miniconda3/envs/rr-real/bin/python
RR_CHECKPOINT_ROOT="$PWD/checkpoints/rr_real_sim_fullframe_cotrain_0907"
RR_INTERFACE_CFG=/home/hz/code/YueHu_deoxys/deoxys/config/charmander.yml

case "$RR_RUN" in
  real40)
    RR_RUN_DIR=rr_fullframe0907_real40_b256_seed2026090712
    ;;
  real40_sim400)
    RR_RUN_DIR=rr_fullframe0907_real40_sim400_b256_seed2026090711
    ;;
  real10_sim400)
    RR_RUN_DIR=rr_fullframe0907_real10_sim400_b256_seed2026090713
    ;;
  *)
    printf 'unsupported RR_RUN: %s\n' "$RR_RUN" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

case "$RR_EPOCH" in
  3000|5000) ;;
  *)
    printf 'unsupported RR_EPOCH: %s\n' "$RR_EPOCH" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

RR_CHECKPOINT="$RR_CHECKPOINT_ROOT/$RR_RUN/$RR_RUN_DIR/$RR_RUN_DIR/actor_chkpt_latest_${RR_EPOCH}.pt"
test -f "$RR_CHECKPOINT" || { printf 'missing checkpoint: %s\n' "$RR_CHECKPOINT" >&2; return 2 2>/dev/null || exit 2; }

"$RR_PYTHON" -m src.real.evaluate_policy \
  --checkpoint "$RR_CHECKPOINT" \
  --interface-cfg "$RR_INTERFACE_CFG" \
  --query-interval-steps 2 \
  --max-steps 100 \
  --show-input-dashboard \
  --save-input-video
```

`--query-interval-steps` remains a normal CLI setting. The recommended value
for this 5 Hz real-robot command is `2`; replace it with another positive value
when comparing query cadences. It is not hard-coded in the evaluator.

`--save-input-video` writes every successful policy query from `b` through `e`
to a four-panel MP4 containing front RGB, wrist RGB, front PromptDA depth, and
wrist PromptDA depth. The file is created next to the JSONL log with the suffix
`-rollout-XXX-rgbd-grid.mp4`.

This is a dry-run: it connects to the cameras and robot and performs policy
inference, but does not send robot or gripper actions. Run it from the
FrankaControl graphical desktop when using the dashboard.

Only after the dry-run, camera framing, workspace, and emergency stop have been
checked, append these arguments to the evaluator command to enable motion:

```bash
  --latency-profile "$PWD/src/real/latency_profile.measured_20260908.json" \
  --workspace-min 0.30 -0.35 0.03 \
  --workspace-max 0.75 0.35 0.60 \
  --min-ee-z 0.04 \
  --execute
```

### Deoxys policy input dashboard

Add `--show-input-dashboard` to `python -m src.real.evaluate_policy` when the
eval is launched from the FrankaControl graphical desktop. The window appears
after `b` and the first successful query, then updates once per query. It shows
the exact post-transform front/wrist RGB-D tensors, online annotation state,
robot proprioception, timing, and the predicted action chunk. PromptDA depth is
shown in meters with a per-camera TURBO colormap and numeric min/median/max.
Gripper diagnostics distinguish all close predictions in the chunk from close
predictions in the next query interval, because farther actions may be replaced
by the next receding-horizon query before they are executed.

The evaluator derives its annotation mode from the checkpoint config. A
checkpoint with `data.annotate_guidance_point_colored=true` runs
`RealSkillAnnotationSession` online and draws the colored guidance point into
`color_image2` (front) before policy inference. `color_image1` (wrist) remains
unmodified because that is the training contract; its projected guidance UV is
listed in the dashboard for diagnosis only. Each `policy_query` JSONL event
also stores the annotation mode, skill state, assembly step, guidance point and
both camera projections, full part poses, part detection/validity flags, and
annotation debug provenance.

The dashboard is inspection-only. Closing or failing the OpenCV window disables
the preview and writes `input_dashboard_failed` without rejecting or changing
robot actions. Keep terminal focus when using the evaluator's `r/b/e/q` keys.
