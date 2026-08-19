#!/usr/bin/env bash

set -euo pipefail

ROOT=/data/hy/robust-rearrangement
TARGET=200
CHUNK_TARGET=10

CAMPAIGN="${1:?usage: $0 CAMPAIGN [OUTPUT_SUFFIX] [RUN_NAME]}"
OUTPUT_SUFFIX="${2:-$CAMPAIGN}"
RUN_NAME="${3:-med-${CAMPAIGN}-0803}"

export DATA_DIR_RAW="$ROOT/raw"
export DATA_DIR_PROCESSED="$ROOT/data"
export LD_LIBRARY_PATH="/home/hy/anaconda3/envs/rr/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPLCONFIGDIR=/tmp/mplconfig-med-0801
export PYTHONPYCACHEPREFIX=/tmp/pycache-med-0801

case "$CAMPAIGN" in
    rgbd-skill-point)
        ANNOTATION_FLAGS=(--annotate-skill --guidance-point-on-image)
        ;;
    rgbd-skill-point-colored)
        ANNOTATION_FLAGS=(--annotate-skill --guidance-point-on-image --guidance-point-colored)
        ;;
    rgbd-skill-grasp-part)
        ANNOTATION_FLAGS=(--annotate-skill --grasp-part-annotate)
        ;;
    rgbd-skill-grasp-part-colored)
        ANNOTATION_FLAGS=(--annotate-skill --grasp-part-annotate --grasp-annotation-colored)
        ;;
    *)
        printf 'Unsupported campaign %q\n' "$CAMPAIGN" >&2
        exit 2
        ;;
esac

success_dir() {
    local task="$1"
    printf '%s/raw/raw/diffik/sim/%s/rollout/med/%s/%s/success\n' \
        "$ROOT" "$task" "$OUTPUT_SUFFIX" "$RUN_NAME"
}

success_count() {
    local directory
    directory="$(success_dir "$1")"
    [[ -d "$directory" ]] || { printf '0\n'; return; }
    find "$directory" -maxdepth 1 -type f -name '*.pkl' -printf '.' | wc -c
}

require_count() {
    local task="$1" count
    count="$(success_count "$task")"
    if [[ "$count" != "$TARGET" ]]; then
        printf 'ERROR: %s has %s saved successes; expected %s\n' "$task" "$count" "$TARGET" >&2
        exit 1
    fi
    printf '%s verified: %s saved successes\n' "$task" "$count"
}

collector_pids() {
    local task="$1"
    ps -u "$(id -u)" -o pid=,args= | awk \
        -v task="$task" -v run_name="$RUN_NAME" -v suffix="$OUTPUT_SUFFIX" '
            $2 == "python" && $3 == "-m" && $4 == "src.eval.evaluate_model" &&
            index($0, "-f " task " ") &&
            index($0, "--rollout-suffix-model-name " run_name) &&
            index($0, "--save-rollouts-suffix " suffix) { print $1 }
        '
}

collect_task() {
    local task="$1" checkpoint="$2" max_steps="$3" rollout_after_success="$4"
    local output_path log_path existing_count chunk_target pids
    output_path="$(success_dir "$task")"
    log_path="$ROOT/logs/med_0801_collect_${CAMPAIGN}_${task}.log"

    while true; do
        existing_count="$(success_count "$task")"
        if (( existing_count > TARGET )); then
            printf 'ERROR: %s has %s pickles, target is %s\n' "$task" "$existing_count" "$TARGET" >&2
            exit 1
        fi
        if (( existing_count == TARGET )); then
            break
        fi

        pids="$(collector_pids "$task")"
        if [[ -n "$pids" ]]; then
            printf 'Waiting for existing %s collector [%s]; saved=%s/%s\n' \
                "$task" "${pids//$'\n'/,}" "$existing_count" "$TARGET"
            sleep 60
            continue
        fi
        if (( existing_count == 0 )) && [[ -e "${output_path%/success}" ]]; then
            printf 'ERROR: non-empty campaign namespace exists with no pickles: %s\n' \
                "${output_path%/success}" >&2
            exit 1
        fi

        chunk_target=$((TARGET - existing_count))
        (( chunk_target > CHUNK_TARGET )) && chunk_target=$CHUNK_TARGET
        printf 'Starting %s/%s chunk: existing=%s target=%s chunk=%s\n' \
            "$CAMPAIGN" "$task" "$existing_count" "$TARGET" "$chunk_target"
        conda run --no-capture-output -n rr python -m src.eval.evaluate_model \
            --gpu 0 \
            --n-envs 4 \
            --n-rollouts 4 \
            --target-successes "$chunk_target" \
            -f "$task" \
            --if-exists append \
            --max-rollout-steps "$max_steps" \
            --action-type pos \
            --observation-space image \
            --randomness med \
            --wt-path "$checkpoint" \
            --save-rollouts \
            --save-depth-image \
            "${ANNOTATION_FLAGS[@]}" \
            --output-only-pickle \
            --max-saved-rollouts "$chunk_target" \
            --save-rollouts-suffix "$OUTPUT_SUFFIX" \
            --rollout-suffix-model-name "$RUN_NAME" \
            --rollout-after-success "$rollout_after_success" \
            2>&1 | tee -a "$log_path"
    done

    for task in one_leg round_table lamp; do require_count "$task"; done
}

cd "$ROOT"
for task_args in \
    "one_leg checkpoints/rppo/one_leg/med/actor_chkpt.pt 700 200" \
    "round_table checkpoints/rppo/round_table/med/actor_chkpt.pt 1000 100" \
    "lamp checkpoints/rppo/lamp/med/actor_chkpt.pt 1000 20"; do
    read -r task checkpoint max_steps after_success <<<"$task_args"
    collect_task "$task" "$checkpoint" "$max_steps" "$after_success"
done

printf 'Independent campaign %s completed at %s\n' "$CAMPAIGN" "$(date --iso-8601=seconds)"
