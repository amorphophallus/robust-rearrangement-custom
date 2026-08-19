#!/usr/bin/env bash

set -euo pipefail

ROOT=/data/hy/robust-rearrangement
CAMPAIGN="${1:?usage: $0 CAMPAIGN SOURCE_SUFFIX RUN_NAME HOST [--build]}"
SOURCE_SUFFIX="${2:?source suffix, e.g. rgbd-only-skill}"
RUN_NAME="${3:?rollout run name}"
HOST_NUMBER="${4:?4090 host number}"
BUILD=false
[[ "${5:-}" == "--build" ]] && BUILD=true

LOCAL_BASE="$ROOT/data/processed/diffik/sim/lamp-one_leg-round_table/rollout/med/success"
LOCAL_PATH="$LOCAL_BASE/${CAMPAIGN}.lmdb"
LOCAL_MARKER="$LOCAL_BASE/.${CAMPAIGN}.local-validated"
LOCAL_SHA_CACHE="$LOCAL_BASE/.${CAMPAIGN}.data-sha256"
REMOTE_DATA_ROOT=/home/hy/robust-rearrangement-custom/data
REMOTE_PATH="$REMOTE_DATA_ROOT/processed/diffik/sim/lamp-one_leg-round_table/rollout/med/success/${CAMPAIGN}.lmdb"
REMOTE_HOST="zju_4090_${HOST_NUMBER}"
MANIFEST="$ROOT/logs/med_0801_${CAMPAIGN}_manifest.sha256"
PROVENANCE="$ROOT/logs/med_0801_${CAMPAIGN}_provenance.json"
SOURCE_MARKER="$ROOT/logs/.med_0801_${CAMPAIGN}.source-finalized"
UPLOAD_LOG="$ROOT/logs/med_0801_upload_${CAMPAIGN}.log"
EVENTS="$ROOT/logs/med_0801_pipeline_events.tsv"

# The first rgbd campaign was finalized before this per-campaign uploader was
# introduced. Reuse its immutable manifest instead of rescanning 354 GB of
# source pickles.
if [[ "$CAMPAIGN" == "rgbd" && "$SOURCE_SUFFIX" == "rgbd-only-skill" && "$RUN_NAME" == "med-rppo-base-0801" ]]; then
    MANIFEST="$ROOT/logs/med_0801_source_manifest.sha256"
    PROVENANCE="$ROOT/logs/med_0801_provenance.json"
    SOURCE_MARKER="$ROOT/logs/.med_0801_provenance_finalized"
fi

export DATA_DIR_RAW="$ROOT/raw"
export DATA_DIR_PROCESSED="$ROOT/data"
export LD_LIBRARY_PATH="/home/hy/anaconda3/envs/rr/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPYCACHEPREFIX=/tmp/pycache-med-0801

ssh_clean() {
    env -u LD_LIBRARY_PATH ssh \
        -o ConnectTimeout=5 \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=2 \
        "$@"
}

source_dir() {
    printf '%s/raw/raw/diffik/sim/%s/rollout/med/%s/%s/success' \
        "$ROOT" "$1" "$SOURCE_SUFFIX" "$RUN_NAME"
}

sha256_cached() {
    local data_path="$1" cache_path="$2" identity cached_identity cached_sha sha tmp
    identity="$(stat -c '%s:%Y:%Z:%i' "$data_path")"
    if [[ -f "$cache_path" ]]; then
        IFS=$'\t' read -r cached_identity cached_sha <"$cache_path" || true
        if [[ "$cached_identity" == "$identity" && "$cached_sha" =~ ^[0-9a-f]{64}$ ]]; then
            printf '%s\n' "$cached_sha"
            return
        fi
    fi
    sha="$(sha256sum "$data_path" | awk '{print $1}')"
    tmp="${cache_path}.tmp.$$"
    printf '%s\t%s\n' "$identity" "$sha" >"$tmp"
    mv -f -- "$tmp" "$cache_path"
    printf '%s\n' "$sha"
}

finalize_source() {
    if [[ -f "$SOURCE_MARKER" && -f "$MANIFEST" && -f "$PROVENANCE" ]]; then
        return
    fi
    conda run --no-capture-output -n rr python scripts/finalize_med_campaign.py \
        --source-suffix "$SOURCE_SUFFIX" \
        --run-name "$RUN_NAME" \
        --manifest "$MANIFEST" \
        --provenance "$PROVENANCE" \
        --marker "$SOURCE_MARKER" \
        2>&1 | tee "$ROOT/logs/med_0801_finalize_${CAMPAIGN}.log"
}

build_and_validate() {
    mkdir -p "$LOCAL_BASE"
    if [[ ! -f "$LOCAL_MARKER" ]]; then
        [[ ! -e "$LOCAL_PATH" ]] || {
            printf 'Incomplete local output exists; inspect before continuing: %s\n' "$LOCAL_PATH" >&2
            exit 1
        }
        conda run --no-capture-output -n rr \
            python -m src.data_processing.process_pickles_to_lmdb \
            --controller diffik --domain sim \
            --task one_leg round_table lamp \
            --source rollout --randomness med --demo-outcome success \
            --suffix "$SOURCE_SUFFIX/$RUN_NAME" \
            --output-dir "$LOCAL_PATH" --output-suffix "$CAMPAIGN" \
            --task-episode-limit one_leg=200 round_table=200 lamp=200 \
            --image-annotation-mode none \
            --provenance-json "$PROVENANCE" \
            --n-cpus 2 --batch-size 2 --map-size-gb 450 --debug-storage-stats \
            2>&1 | tee "$ROOT/logs/med_0801_build_${CAMPAIGN}.log"
    fi

    conda run --no-capture-output -n rr \
        python scripts/validate_lmdb_dataset.py --full-stats "$LOCAL_PATH" \
        2>&1 | tee "$ROOT/logs/med_0801_validate_${CAMPAIGN}.log"
    conda run --no-capture-output -n rr \
        python scripts/validate_med_campaign.py "$LOCAL_PATH" \
        --source-suffix "$SOURCE_SUFFIX" --run-name "$RUN_NAME" \
        --manifest "$MANIFEST" --annotation-mode none --require-rollout-stage \
        2>&1 | tee -a "$ROOT/logs/med_0801_validate_${CAMPAIGN}.log"
    touch "$LOCAL_MARKER"
}

upload_and_cleanup() {
    local bytes local_sha remote_sha available required start_epoch end_epoch duration mbps
    [[ -f "$LOCAL_MARKER" && -f "$LOCAL_PATH/data.mdb" ]] || {
        printf 'Dataset is not locally validated: %s\n' "$LOCAL_PATH" >&2
        exit 1
    }
    bytes="$(stat -c %s "$LOCAL_PATH/data.mdb")"
    local_sha="$(sha256_cached "$LOCAL_PATH/data.mdb" "$LOCAL_SHA_CACHE")"

    ssh_clean -o BatchMode=yes "$REMOTE_HOST" \
        "mkdir -p '$REMOTE_PATH'; \
        if test -f '$REMOTE_PATH/.med0801-upload-complete'; then :; \
        elif test -e '$REMOTE_PATH/data.mdb' || test -e '$REMOTE_PATH/.med0801-upload-incomplete'; then touch '$REMOTE_PATH/.med0801-upload-incomplete'; \
        else touch '$REMOTE_PATH/.med0801-upload-incomplete'; fi; \
        printf 'mount='; findmnt -T '$REMOTE_PATH' -no TARGET,SOURCE,FSTYPE; \
        df -B1 --output=avail '$REMOTE_PATH' | tail -n 1"

    if ssh_clean -o BatchMode=yes "$REMOTE_HOST" "test -f '$REMOTE_PATH/.med0801-upload-complete'"; then
        remote_sha="$(ssh_clean -o BatchMode=yes "$REMOTE_HOST" "awk -F= '/^sha256=/{print \$2}' '$REMOTE_PATH/.med0801-upload-complete'")"
        [[ "$remote_sha" == "$local_sha" ]] || {
            printf 'Completed remote hash mismatch: local=%s remote=%s\n' "$local_sha" "$remote_sha" >&2
            exit 1
        }
    else
        available="$(ssh_clean -o BatchMode=yes "$REMOTE_HOST" "df -B1 --output=avail '$REMOTE_PATH' | tail -n 1")"
        required=$((bytes + 10 * 1000 * 1000 * 1000))
        (( available >= required )) || {
            printf 'Remote capacity insufficient: available=%s required=%s\n' "$available" "$required" >&2
            exit 1
        }

        : >"$UPLOAD_LOG"
        start_epoch="$(date +%s)"
        env -u LD_LIBRARY_PATH rsync -aS --partial --inplace --append-verify \
            -e 'ssh -o ConnectTimeout=5 -o ServerAliveInterval=15 -o ServerAliveCountMax=2' \
            --exclude lock.mdb --bwlimit=100000 --info=progress2,stats2 \
            "$LOCAL_PATH/" "$REMOTE_HOST:$REMOTE_PATH/" \
            2>&1 | tee "$UPLOAD_LOG"
        end_epoch="$(date +%s)"
        duration=$((end_epoch - start_epoch)); (( duration < 1 )) && duration=1
        mbps="$(awk -v bytes="$bytes" -v seconds="$duration" 'BEGIN {printf "%.2f", bytes / seconds / 1000000}')"

        remote_sha="$(ssh_clean -o BatchMode=yes "$REMOTE_HOST" "sha256sum '$REMOTE_PATH/data.mdb' | awk '{print \$1}'")"
        [[ "$remote_sha" == "$local_sha" ]] || {
            printf 'Remote hash mismatch: local=%s remote=%s\n' "$local_sha" "$remote_sha" >&2
            exit 1
        }
        ssh_clean -o BatchMode=yes "$REMOTE_HOST" \
            "cd /mnt/nas/share/home/hy/robust-rearrangement-custom && \
            DATA_DIR_PROCESSED='$REMOTE_DATA_ROOT' /mnt/nas/share/home/hy/miniconda3/bin/conda run --no-capture-output -n rr \
            python scripts/validate_lmdb_dataset.py --sample-episodes 5 '$REMOTE_PATH'" \
            2>&1 | tee -a "$UPLOAD_LOG"
        ssh_clean -o BatchMode=yes "$REMOTE_HOST" \
            "printf 'campaign=%s\\nsource_suffix=%s\\nrun_name=%s\\nannotation_stage=rollout\\nsha256=%s\\nbytes=%s\\nuploaded_at=%s\\n' \
            '$CAMPAIGN' '$SOURCE_SUFFIX' '$RUN_NAME' '$local_sha' '$bytes' '$(date --iso-8601=seconds)' >'$REMOTE_PATH/.med0801-upload-complete'; \
            rm -f '$REMOTE_PATH/.med0801-upload-incomplete'"
        printf '%s\tupload\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "$CAMPAIGN" "$HOST_NUMBER" "$REMOTE_PATH" \
            "$bytes" "$duration" "$mbps" >>"$EVENTS"
        printf 'Uploaded %s to %s at %s MB/s\n' "$CAMPAIGN" "$REMOTE_HOST" "$mbps"
    fi

    # The complete marker, matching SHA256 and remote loader check are the
    # deletion gate. Only this campaign's source namespace is removed.
    for task in one_leg round_table lamp; do
        rm -rf -- "$(source_dir "$task")"
    done
    rm -rf -- "$LOCAL_PATH"
    rm -f -- "$LOCAL_MARKER" "$LOCAL_SHA_CACHE"
    printf '%s\n' "$(date --iso-8601=seconds) deleted local pickles and LMDB after verified upload" \
        >>"$ROOT/logs/med_0801_cleanup_${CAMPAIGN}.log"
}

cd "$ROOT"
if [[ "$BUILD" == true ]]; then
    finalize_source
    build_and_validate
else
    [[ -f "$MANIFEST" && -f "$PROVENANCE" ]] || finalize_source
fi
upload_and_cleanup
