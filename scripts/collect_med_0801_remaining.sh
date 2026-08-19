#!/usr/bin/env bash

set -euo pipefail

ROOT=/data/hy/robust-rearrangement
RUN_NAME=med-rppo-base-0801
OUTPUT_SUFFIX=rgbd-only-skill
TARGET=200
CHUNK_TARGET=10

export LD_LIBRARY_PATH="/home/hy/anaconda3/envs/rr/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPLCONFIGDIR=/tmp/mplconfig-med-0801
export PYTHONPYCACHEPREFIX=/tmp/pycache-med-0801

success_dir() {
    local task="$1"
    printf '%s/raw/raw/diffik/sim/%s/rollout/med/%s/%s/success\n' \
        "$ROOT" "$task" "$OUTPUT_SUFFIX" "$RUN_NAME"
}

success_count() {
    local directory
    directory="$(success_dir "$1")"
    if [[ ! -d "$directory" ]]; then
        printf '0\n'
        return
    fi
    find "$directory" -maxdepth 1 -type f -name '*.pkl' -printf '.' | wc -c
}

require_count() {
    local task="$1"
    local count
    count="$(success_count "$task")"
    if [[ "$count" != "$TARGET" ]]; then
        printf 'ERROR: %s finished with %s saved successes; expected %s\n' \
            "$task" "$count" "$TARGET" >&2
        exit 1
    fi
    printf '%s verified: %s saved successes\n' "$task" "$count"
}

collector_pids() {
    local task="$1"
    ps -u "$(id -u)" -o pid=,args= | awk \
        -v task="$task" -v run_name="$RUN_NAME" '
            $2 == "python" &&
            $3 == "-m" &&
            $4 == "src.eval.evaluate_model" &&
            index($0, "-f " task " ") &&
            index($0, "--rollout-suffix-model-name " run_name) {
                print $1
            }
        '
}

collect_task() {
    local task="$1"
    local checkpoint="$2"
    local max_steps="$3"
    local rollout_after_success="$4"
    local output_path
    local log_path
    local existing_count
    local chunk_target
    local collector_pid_list

    output_path="$(success_dir "$task")"
    log_path="$ROOT/logs/med_0801_collect_${task}.log"
    while true; do
        existing_count="$(success_count "$task")"
        if (( existing_count > TARGET )); then
            printf 'ERROR: %s already has %s saved successes; target is %s\n' \
                "$task" "$existing_count" "$TARGET" >&2
            exit 1
        fi
        if (( existing_count == TARGET )); then
            printf '%s already complete: %s/%s\n' "$task" "$existing_count" "$TARGET"
            break
        fi

        collector_pid_list="$(collector_pids "$task")"
        if [[ -n "$collector_pid_list" ]]; then
            printf 'Waiting for existing %s collector pid(s) [%s] at %s; saved=%s/%s\n' \
                "$task" "${collector_pid_list//$'\n'/,}" "$(date --iso-8601=seconds)" \
                "$existing_count" "$TARGET"
            sleep 60
            continue
        fi

        if (( existing_count == 0 )) && [[ -e "${output_path%/success}" ]]; then
            printf 'ERROR: %s output exists but contains no success pickles: %s\n' \
                "$task" "${output_path%/success}" >&2
            exit 1
        fi

        chunk_target=$((TARGET - existing_count))
        if (( chunk_target > CHUNK_TARGET )); then
            chunk_target=$CHUNK_TARGET
        fi
        printf 'Starting %s chunk at %s; existing=%s target=%s chunk=%s\n' \
            "$task" "$(date --iso-8601=seconds)" "$existing_count" "$TARGET" "$chunk_target"
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
            --annotate-skill \
            --output-only-pickle \
            --max-saved-rollouts "$chunk_target" \
            --save-rollouts-suffix "$OUTPUT_SUFFIX" \
            --rollout-suffix-model-name "$RUN_NAME" \
            --rollout-after-success "$rollout_after_success" \
            2>&1 | tee -a "$log_path"
    done
    require_count "$task"
}

cd "$ROOT"

collect_task \
    one_leg \
    checkpoints/rppo/one_leg/med/actor_chkpt.pt \
    700 \
    200
collect_task \
    round_table \
    checkpoints/rppo/round_table/med/actor_chkpt.pt \
    1000 \
    100
collect_task \
    lamp \
    checkpoints/rppo/lamp/med/actor_chkpt.pt \
    1000 \
    20

printf 'All formal collection tasks completed at %s\n' "$(date --iso-8601=seconds)"
