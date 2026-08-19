#!/usr/bin/env bash

set -euo pipefail

ROOT=/data/hy/robust-rearrangement
REMOTE_DATA_ROOT=/home/hy/robust-rearrangement-custom/data
REMOTE_BASE="$REMOTE_DATA_ROOT/processed/diffik/sim/lamp-one_leg-round_table/rollout/med/success"
REGISTRY="$ROOT/logs/med_0801_training_registry.tsv"
POLL_SECONDS=120

declare -A HOSTS=(
    [1]=232 [2]=236 [3]=228 [4]=232
    [5]=236 [6]=232 [7]=240 [8]=243
)
declare -A GPUS=(
    [1]=0,1 [2]=0,1 [3]=4,5 [4]=2,3
    [5]=2,3 [6]=4,5 [7]=0,1 [8]=0,1
)
declare -A SUFFIXES=(
    [1]=rgbd
    [2]=rgbd-skill-point
    [3]=rgbd-skill-point-colored
    [4]=rgbd
    [5]=rgbd-skill-point
    [6]=rgbd
    [7]=rgbd-skill-grasp-part
    [8]=rgbd-skill-grasp-part-colored
)
declare -A DATA_ROOTS=(
    [1]="$REMOTE_DATA_ROOT"
    [2]="$REMOTE_DATA_ROOT"
    [3]="/var/tmp/hy/robust-rearrangement-med-0801/data"
    [4]="$REMOTE_DATA_ROOT"
    [5]="$REMOTE_DATA_ROOT"
    [6]="$REMOTE_DATA_ROOT"
    [7]="$REMOTE_DATA_ROOT"
    [8]="$REMOTE_DATA_ROOT"
)

free_gpu_count() {
    local host="zju_4090_$1"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits" \
        | awk -F, '{total=$1+0; used=$2+0; if (total > 0 && used / total < 0.1) count++} END {print count+0}'
}

dataset_ready() {
    local host="zju_4090_$1"
    local suffix="$2"
    local data_root="$3"
    local base="$data_root/processed/diffik/sim/lamp-one_leg-round_table/rollout/med/success"
    local test_path="$base/${suffix}.lmdb/.med0801-upload-complete"
    if [[ "$suffix" == rgbd-skill-point-colored ]]; then
        test_path="$base/.${suffix}.med0801-upload-complete"
    fi
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
        "test -s '$test_path'"
}

registry_has_exp() {
    local exp_id="$1"
    [[ -f "$REGISTRY" ]] && awk -F '\t' -v exp_id="$exp_id" '$2 == exp_id && $3 == "started" {found=1} END {exit !found}' "$REGISTRY"
}

record_launch() {
    local exp="$1"
    local output_file="$2"
    local host gpu_ids tmux_name run_name run_id
    host="$(awk -F ': ' '$1 == "server" {print $2}' "$output_file" | tail -n 1)"
    gpu_ids="$(awk -F ': ' '$1 == "gpu_ids" {print $2}' "$output_file" | tail -n 1)"
    tmux_name="$(awk -F ': ' '$1 == "tmux_name" {print $2}' "$output_file" | tail -n 1)"
    run_name="$(awk -F ': ' '$1 == "wandb_run_name" {print $2}' "$output_file" | tail -n 1)"
    run_id="$(awk -F ': ' '$1 == "wandb_run_id" {print $2}' "$output_file" | tail -n 1)"
    printf '%s\t%s\tstarted\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "$exp" "$host" "$gpu_ids" \
        "$tmux_name" "$run_name" "$run_id" "${SUFFIXES[$exp]}" \
        "${DATA_ROOTS[$exp]}" >>"$REGISTRY"
}

launch_when_ready() {
    local exp="$1"
    local host="${HOSTS[$exp]}"
    local suffix="${SUFFIXES[$exp]}"
    local data_root="${DATA_ROOTS[$exp]}"
    local preferred_gpus="${GPUS[$exp]}"
    local launching_marker="$ROOT/logs/.med_0801_exp${exp}.launching"
    local launched_marker="$ROOT/logs/.med_0801_exp${exp}.launched"
    local launch_log="$ROOT/logs/med_0801_train_exp${exp}_launch.log"
    local free_count

    if [[ -f "$launched_marker" ]] || registry_has_exp "$exp"; then
        printf 'Exp%s already registered as started\n' "$exp"
        return
    fi
    if [[ -f "$launching_marker" ]]; then
        printf 'ERROR: Exp%s has an unresolved launching marker: %s\n' \
            "$exp" "$launching_marker" >&2
        return 1
    fi

    while true; do
        if ! dataset_ready "$host" "$suffix" "$data_root"; then
            printf 'Exp%s waiting for dataset %s on %s at %s\n' \
                "$exp" "$suffix" "$host" "$(date --iso-8601=seconds)"
            sleep "$POLL_SECONDS"
            continue
        fi

        exec {lock_fd}>"/tmp/med0801-train-host-${host}.lock"
        flock "$lock_fd"
        if ! free_count="$(free_gpu_count "$host")"; then
            flock -u "$lock_fd"
            printf 'Exp%s could not query GPUs on host %s; retrying at %s\n' \
                "$exp" "$host" "$(date --iso-8601=seconds)"
            sleep "$POLL_SECONDS"
            continue
        fi
        if (( free_count < 2 )); then
            flock -u "$lock_fd"
            printf 'Exp%s dataset ready; host %s has %s free GPUs, waiting at %s\n' \
                "$exp" "$host" "$free_count" "$(date --iso-8601=seconds)"
            sleep "$POLL_SECONDS"
            continue
        fi

        printf '%s\n' "$(date --iso-8601=seconds)" >"$launching_marker"
        printf 'Launching Exp%s on host %s, preferred GPUs %s at %s\n' \
            "$exp" "$host" "$preferred_gpus" "$(date --iso-8601=seconds)"
        if scripts/launch_med_0801_training.sh \
            "$exp" "$host" "$preferred_gpus" "$data_root" \
            2>&1 | tee "$launch_log"; then
            if ! rg -q '^status: started$' "$launch_log"; then
                printf 'ERROR: Exp%s launcher exited zero without started status\n' \
                    "$exp" >&2
                flock -u "$lock_fd"
                return 1
            fi
            record_launch "$exp" "$launch_log"
            mv "$launching_marker" "$launched_marker"
            flock -u "$lock_fd"
            printf 'Exp%s registered as started\n' "$exp"
            return
        fi

        printf 'ERROR: Exp%s launch failed; leaving %s for manual audit\n' \
            "$exp" "$launching_marker" >&2
        flock -u "$lock_fd"
        return 1
    done
}

cd "$ROOT"
printf 'timestamp\texp\tstatus\thost\tgpu_ids\ttmux\twandb_run\twandb_run_id\tsuffix\tdata_root\n' >"$REGISTRY.tmp"
if [[ -f "$REGISTRY" ]]; then
    tail -n +2 "$REGISTRY" >>"$REGISTRY.tmp"
fi
mv "$REGISTRY.tmp" "$REGISTRY"

pids=()
for exp in {1..8}; do
    launch_when_ready "$exp" \
        >>"$ROOT/logs/med_0801_train_exp${exp}_scheduler.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if (( status != 0 )); then
    printf 'At least one med-0801 experiment failed to launch; inspect scheduler logs.\n' >&2
    exit "$status"
fi
printf 'All eight med-0801 experiments registered as started at %s\n' \
    "$(date --iso-8601=seconds)"
