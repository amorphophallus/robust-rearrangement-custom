#!/usr/bin/env bash

set -euo pipefail

ROOT=/data/hy/robust-rearrangement
REGISTRY="$ROOT/logs/med_0801_training_registry.tsv"
REMOTE_STATUS="$ROOT/logs/med_0801_training_remote_status.tsv"
WANDB_STATUS="$ROOT/logs/med_0801_training_wandb_status.tsv"
STARTUP_POLL_SECONDS=600
STABLE_POLL_SECONDS=3600
REGISTRY_WAKE_SECONDS=600

ssh_host_for() {
    case "$1" in
        zju_4090_*) printf '%s\n' "$1" ;;
        [0-9]*) printf 'zju_4090_%s\n' "$1" ;;
        *) return 1 ;;
    esac
}

registry_row() {
    local exp_id="$1"
    awk -F '\t' -v exp_id="$exp_id" '$2 == exp_id {row=$0} END {print row}' "$REGISTRY"
}

inspect_remote() {
    local exp="$1"
    local host="$2"
    local tmux_name="$3"
    local run_name="$4"
    local ssh_host
    local pane_log="$ROOT/logs/med_0801_train_exp${exp}_pane.log"
    local output
    ssh_host="$(ssh_host_for "$host")"

    output="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$ssh_host" bash -s -- "$tmux_name" "$run_name" <<'REMOTE'
set -euo pipefail
session_name="$1"
run_name="$2"
project=/mnt/nas/share/home/hy/robust-rearrangement-custom
output_root=/data/hy/robust-rearrangement-custom/outputs
meta="$(find "$output_root" -type f -path "*/models/$run_name/actor_chkpt_last.pt.meta.json" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 {print $2}')"
pane_state=missing
pane_exit=-
if tmux has-session -t "$session_name" 2>/dev/null; then
    pane_info="$(tmux list-panes -t "$session_name:train" -F '#{pane_dead}|#{pane_dead_status}' | head -n 1)"
    IFS='|' read -r pane_dead pane_exit <<<"$pane_info"
    if [[ "$pane_dead" == 0 ]]; then pane_state=running; else pane_state=dead; fi
fi
checkpoint_epoch=-
if [[ -n "$meta" && -s "$meta" ]]; then
    checkpoint_epoch="$(sed -n 's/.*"epoch"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$meta" | head -n 1)"
    checkpoint_epoch="${checkpoint_epoch:--}"
fi
printf '%s\t%s\t%s\n' "$pane_state" "$pane_exit" "$checkpoint_epoch"
REMOTE
)"

    ssh -o BatchMode=yes -o ConnectTimeout=5 "$ssh_host" \
        "tmux capture-pane -pt '$tmux_name:train' -S -500 2>/dev/null || true" \
        >"$pane_log" || true
    printf '%s\n' "$output"
}

cd "$ROOT"
while true; do
    printf 'timestamp\texp\thost\ttmux\tpane_state\tpane_exit\tcheckpoint_epoch\n' >"$REMOTE_STATUS.tmp"
    launched=0
    locally_complete=0
    failed=0

    for exp in {1..8}; do
        row="$(registry_row "$exp")"
        [[ -n "$row" ]] || continue
        launched=$((launched + 1))
        IFS=$'\t' read -r _ _ status host _ tmux_name run_name _ _ _ <<<"$row"
        [[ "$status" == started ]] || continue

        if ! remote="$(inspect_remote "$exp" "$host" "$tmux_name" "$run_name")"; then
            remote="ssh-error|-|-"
        fi
        IFS=$'\t|' read -r pane_state pane_exit checkpoint_epoch <<<"$remote"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "$exp" "$host" "$tmux_name" \
            "$pane_state" "$pane_exit" "$checkpoint_epoch" >>"$REMOTE_STATUS.tmp"

        if [[ "$pane_state" == dead && "$pane_exit" != 0 ]]; then
            failed=1
        elif [[ "$checkpoint_epoch" =~ ^[0-9]+$ ]] && (( checkpoint_epoch >= 2999 )); then
            if [[ "$pane_state" == dead && "$pane_exit" == 0 ]]; then
                locally_complete=$((locally_complete + 1))
            fi
        elif [[ "$pane_state" == missing ]]; then
            failed=1
        fi
    done
    mv "$REMOTE_STATUS.tmp" "$REMOTE_STATUS"

    wandb_query_ok=0
    if (( launched > 0 )); then
        set +e
        conda run --no-capture-output -n rr \
            python scripts/query_med_0801_wandb.py "$REGISTRY" \
            >"$WANDB_STATUS.tmp" 2>"$ROOT/logs/med_0801_training_wandb_query.err"
        wandb_query_status=$?
        set -e
        mv "$WANDB_STATUS.tmp" "$WANDB_STATUS" 2>/dev/null || true
        if (( wandb_query_status == 0 )); then
            wandb_query_ok=1
        elif (( wandb_query_status == 2 )); then
            printf 'ERROR: W&B config differs from approved matrix\n' >&2
            failed=1
        fi
    fi

    if (( failed != 0 )); then
        printf 'ERROR: at least one training pane failed or disappeared at %s\n' \
            "$(date --iso-8601=seconds)" >&2
        exit 1
    fi

    wandb_finished=0
    if (( wandb_query_ok == 1 )); then
        wandb_finished="$(awk -F '\t' 'NR > 1 && $2 == "finished" && $4 == "true" {count++} END {print count+0}' "$WANDB_STATUS")"
    fi
    printf 'Training monitor at %s: launched=%s local_complete=%s wandb_finished=%s\n' \
        "$(date --iso-8601=seconds)" "$launched" "$locally_complete" "$wandb_finished"

    if (( launched == 8 && locally_complete == 8 && wandb_finished == 8 )); then
        printf 'All eight med-0801 trainings completed and verified at %s\n' \
            "$(date --iso-8601=seconds)"
        break
    fi

    poll_seconds=$STABLE_POLL_SECONDS
    if (( launched > 0 )); then
        latest_launch="$(awk -F '\t' '$3 == "started" {ts=$1} END {print ts}' "$REGISTRY")"
        if [[ -n "$latest_launch" ]]; then
            latest_epoch="$(date -d "$latest_launch" +%s)"
            launch_age=$(( $(date +%s) - latest_epoch ))
            if (( launch_age < 3600 )); then
                poll_seconds=$STARTUP_POLL_SECONDS
            fi
        fi
    fi
    printf 'Next full training check in %ss (local registry wake check every %ss)\n' \
        "$poll_seconds" "$REGISTRY_WAKE_SECONDS"

    slept=0
    while (( slept < poll_seconds )); do
        sleep_for=$REGISTRY_WAKE_SECONDS
        if (( slept + sleep_for > poll_seconds )); then
            sleep_for=$((poll_seconds - slept))
        fi
        sleep "$sleep_for"
        slept=$((slept + sleep_for))
        current_launched="$(awk -F '\t' '$3 == "started" {seen[$2]=1} END {for (exp_id in seen) count++; print count+0}' "$REGISTRY")"
        if (( current_launched != launched )); then
            printf 'Registry launch count changed %s -> %s; waking full monitor early\n' \
                "$launched" "$current_launched"
            break
        fi
    done
done
