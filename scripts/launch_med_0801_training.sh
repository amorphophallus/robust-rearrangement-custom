#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: launch_med_0801_training.sh EXP HOST GPU_CSV FAST_DATA_ROOT [--wandb-continue-run-id RUN_ID]

EXP is an integer from 1 through 8. HOST may be a numeric zju host suffix or SSH alias.
USAGE
}

if (( $# < 4 )); then
    usage >&2
    exit 2
fi

EXP="$1"
HOST="$2"
GPU_CSV="$3"
FAST_DATA_ROOT="$4"
shift 4

LAUNCHER=/data/hy/gpu-snatcher/auto_train_multi_card.sh
export POLL_TIMEOUT_SECONDS="${POLL_TIMEOUT_SECONDS:-1800}"
COMMON_ARGS=(
    --ssh-name "$HOST"
    --gpu-id "$GPU_CSV"
    --num-gpus 2
    --data-dir-processed "$FAST_DATA_ROOT"
    --task-spec '[one_leg, round_table, lamp]'
    --storage-format lmdb
    --load-into-memory false
    --dataloader-workers 4
    --batch-size 512
    --num-epochs 3000
    --steps-per-epoch 100
    --save-per-epoch 500
    --randomness med
    --dryrun false
    --wandb-project multi-task-rgbd-skill-med-0801
    --wandb-mode online
    --ddp-shard-enabled true
    --no-grasp
)

case "$EXP" in
    1)
        EXP_ARGS=(
            --data-suffix rgbd
            --experiment rgbd/dit
            --no-guidance-point --no-skill-one-hot --no-guidance-point-colored
            --no-grasp-colored --no-grasp-part
        )
        ;;
    2)
        EXP_ARGS=(
            --data-suffix rgbd-skill-point
            --experiment rgbd/dit
            --guidance-point --no-skill-one-hot --no-guidance-point-colored
            --no-grasp-colored --no-grasp-part
        )
        ;;
    3)
        EXP_ARGS=(
            --data-suffix rgbd-skill-point-colored
            --experiment rgbd/dit
            --guidance-point --no-skill-one-hot --guidance-point-colored
            --no-grasp-colored --no-grasp-part
        )
        ;;
    4)
        EXP_ARGS=(
            --data-suffix rgbd
            --experiment rgbd/dit
            --no-guidance-point --skill-one-hot --no-guidance-point-colored
            --no-grasp-colored --no-grasp-part
        )
        ;;
    5)
        EXP_ARGS=(
            --data-suffix rgbd-skill-point
            --experiment rgbd/dit
            --guidance-point --skill-one-hot --no-guidance-point-colored
            --no-grasp-colored --no-grasp-part
        )
        ;;
    6)
        EXP_ARGS=(
            --data-suffix rgbd
            --experiment image/dit
            --vision-encoder resnet
            --vision-encoder-pretrained false
            --no-guidance-point --no-skill-one-hot --no-guidance-point-colored
            --no-grasp-colored --no-grasp-part
        )
        ;;
    7)
        EXP_ARGS=(
            --data-suffix rgbd-skill-grasp-part
            --experiment rgbd/dit
            --no-guidance-point --no-skill-one-hot --no-guidance-point-colored
            --no-grasp-colored --grasp-part
        )
        ;;
    8)
        EXP_ARGS=(
            --data-suffix rgbd-skill-grasp-part-colored
            --experiment rgbd/dit
            --no-guidance-point --no-skill-one-hot --guidance-point-colored
            --grasp-colored --grasp-part
        )
        ;;
    *)
        printf 'EXP must be an integer from 1 through 8, got %s\n' "$EXP" >&2
        exit 2
        ;;
esac

exec bash "$LAUNCHER" "${COMMON_ARGS[@]}" "${EXP_ARGS[@]}" "$@"
