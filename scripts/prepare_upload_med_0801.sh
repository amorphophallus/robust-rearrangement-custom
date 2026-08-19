#!/usr/bin/env bash

set -euo pipefail

ROOT=/data/hy/robust-rearrangement
SOURCE_SUFFIX=rgbd-only-skill/med-rppo-base-0801
LOCAL_BASE="$ROOT/data/processed/diffik/sim/lamp-one_leg-round_table/rollout/med/success"
REMOTE_DATA_ROOT=/home/hy/robust-rearrangement-custom/data
REMOTE_BASE="$REMOTE_DATA_ROOT/processed/diffik/sim/lamp-one_leg-round_table/rollout/med/success"
REMOTE_228_DATA_ROOT=/var/tmp/hy/robust-rearrangement-med-0801/data
REMOTE_228_AUX_ROOT=/data/hy/robust-rearrangement-med-0801/data
REMOTE_228_BASE="$REMOTE_228_DATA_ROOT/processed/diffik/sim/lamp-one_leg-round_table/rollout/med/success"
REMOTE_228_AUX_BASE="$REMOTE_228_AUX_ROOT/processed/diffik/sim/lamp-one_leg-round_table/rollout/med/success"
PROVENANCE="$ROOT/logs/med_0801_provenance.json"
EVENTS="$ROOT/logs/med_0801_pipeline_events.tsv"
TARGET=200

export DATA_DIR_RAW="$ROOT/raw"
export DATA_DIR_PROCESSED="$ROOT/data"
export LD_LIBRARY_PATH="/home/hy/anaconda3/envs/rr/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPYCACHEPREFIX=/tmp/pycache-med-0801

declare -a SUFFIXES=(
    rgbd
    rgbd-skill-point
    rgbd-skill-point-colored
    rgbd-skill-grasp-part
    rgbd-skill-grasp-part-colored
)
declare -A MODES=(
    [rgbd]=none
    [rgbd-skill-point]=guidance-point
    [rgbd-skill-point-colored]=guidance-point-colored
    [rgbd-skill-grasp-part]=grasp-part
    [rgbd-skill-grasp-part-colored]=grasp-part-colored
)
declare -A HOSTS=(
    [rgbd]=232
    [rgbd-skill-point]=236
    [rgbd-skill-point-colored]=228
    [rgbd-skill-grasp-part]=240
    [rgbd-skill-grasp-part-colored]=243
)

success_dir() {
    local task="$1"
    printf '%s/raw/raw/diffik/sim/%s/rollout/med/rgbd-only-skill/med-rppo-base-0801/success\n' \
        "$ROOT" "$task"
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

wait_for_sources() {
    local complete task count
    while true; do
        complete=true
        for task in one_leg round_table lamp; do
            count="$(success_count "$task")"
            if (( count > TARGET )); then
                printf 'ERROR: %s has %s source pickles; expected no more than %s\n' \
                    "$task" "$count" "$TARGET" >&2
                exit 1
            fi
            if (( count != TARGET )); then
                complete=false
            fi
        done
        if [[ "$complete" == true ]]; then
            break
        fi
        printf 'Waiting for sources at %s: one_leg=%s round_table=%s lamp=%s\n' \
            "$(date --iso-8601=seconds)" \
            "$(success_count one_leg)" \
            "$(success_count round_table)" \
            "$(success_count lamp)"
        sleep 120
    done
}

require_local_build_capacity() {
    local source_bytes available reclaimable effective_available required
    local suffix marker output
    source_bytes="$(
        for task in one_leg round_table lamp; do
            find "$(success_dir "$task")" -maxdepth 1 -type f -name '*.pkl' -printf '%s\n'
        done | awk '{total += $1} END {printf "%.0f", total}'
    )"
    available="$(df -B1 --output=avail "$ROOT/data" | tail -n 1)"
    reclaimable=0
    for suffix in "${SUFFIXES[@]}"; do
        marker="$LOCAL_BASE/.${suffix}.local-validated"
        [[ -f "$marker" ]] || continue

        if [[ "$suffix" == rgbd-skill-point-colored ]]; then
            if [[ -d "$LOCAL_BASE/${suffix}-1.lmdb" && -d "$LOCAL_BASE/${suffix}-2.lmdb" ]]; then
                reclaimable=$((
                    reclaimable + $(
                        du -sb -- \
                            "$LOCAL_BASE/${suffix}-1.lmdb" \
                            "$LOCAL_BASE/${suffix}-2.lmdb" \
                            | awk '{total += $1} END {printf "%.0f", total}'
                    )
                ))
            fi
        else
            output="$LOCAL_BASE/${suffix}.lmdb"
            if [[ -d "$output" ]]; then
                reclaimable=$((reclaimable + $(du -sb -- "$output" | awk '{print $1}')))
            fi
        fi
    done
    effective_available=$((available + reclaimable))
    required=$((source_bytes + 15 * 1000 * 1000 * 1000))
    printf 'Local LMDB capacity check: source=%s available=%s reclaimable_validated=%s effective_available=%s required=%s\n' \
        "$source_bytes" "$available" "$reclaimable" "$effective_available" "$required"
    if (( effective_available < required )); then
        printf 'ERROR: local fast disk lacks source-size plus 15 GB build margin\n' >&2
        exit 1
    fi
}

record_event() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >>"$EVENTS"
}

ssh_clean() {
    env -u LD_LIBRARY_PATH ssh "$@"
}

sha256_cached() {
    local data_path="$1"
    local cache_path="$2"
    local identity cached_identity cached_sha sha tmp_path

    identity="$(stat -c '%s:%Y:%Z:%i' "$data_path")"
    if [[ -f "$cache_path" ]]; then
        IFS=$'\t' read -r cached_identity cached_sha <"$cache_path" || true
        if [[ "$cached_identity" == "$identity" && "$cached_sha" =~ ^[0-9a-f]{64}$ ]]; then
            printf 'Reusing cached SHA-256 for %s\n' "$data_path" >&2
            printf '%s\n' "$cached_sha"
            return
        fi
    fi

    sha="$(sha256sum "$data_path" | awk '{print $1}')"
    tmp_path="${cache_path}.tmp.$$"
    printf '%s\t%s\n' "$identity" "$sha" >"$tmp_path"
    mv -f -- "$tmp_path" "$cache_path"
    printf '%s\n' "$sha"
}

build_dataset() {
    local suffix="$1"
    local mode="$2"
    local output="$LOCAL_BASE/${suffix}.lmdb"
    local complete_marker="$LOCAL_BASE/.${suffix}.local-validated"

    if [[ -f "$complete_marker" && -d "$output" ]]; then
        printf 'Reusing validated local dataset: %s\n' "$output"
        return
    fi
    if [[ -e "$output" || -e "$complete_marker" ]]; then
        printf 'ERROR: incomplete local output requires manual inspection: %s\n' \
            "$output" >&2
        exit 1
    fi

    mkdir -p "$LOCAL_BASE"
    printf 'Building %s with annotation mode %s at %s\n' \
        "$suffix" "$mode" "$(date --iso-8601=seconds)"
    conda run --no-capture-output -n rr \
        python -m src.data_processing.process_pickles_to_lmdb \
        --controller diffik \
        --domain sim \
        --task one_leg round_table lamp \
        --source rollout \
        --randomness med \
        --demo-outcome success \
        --suffix "$SOURCE_SUFFIX" \
        --output-dir "$output" \
        --output-suffix "$suffix" \
        --task-episode-limit one_leg=200 round_table=200 lamp=200 \
        --image-annotation-mode "$mode" \
        --provenance-json "$PROVENANCE" \
        --n-cpus 2 \
        --batch-size 2 \
        --map-size-gb 450 \
        --debug-storage-stats \
        2>&1 | tee "$ROOT/logs/med_0801_build_${suffix}.log"

    conda run --no-capture-output -n rr \
        python scripts/validate_lmdb_dataset.py --full-stats "$output" \
        2>&1 | tee "$ROOT/logs/med_0801_validate_${suffix}.log"
    conda run --no-capture-output -n rr \
        python scripts/validate_med_0801_lmdb.py "$output" \
        --suffix "$suffix" \
        --annotation-mode "$mode" \
        --source-manifest "$ROOT/logs/med_0801_source_manifest.sha256" \
        2>&1 | tee -a "$ROOT/logs/med_0801_validate_${suffix}.log"
    touch "$complete_marker"
}

build_sharded_dataset() {
    local suffix="$1"
    local mode="$2"
    local output_1="$LOCAL_BASE/${suffix}-1.lmdb"
    local output_2="$LOCAL_BASE/${suffix}-2.lmdb"
    local complete_marker="$LOCAL_BASE/.${suffix}.local-validated"
    local build_log="$ROOT/logs/med_0801_build_${suffix}.log"
    local validate_log="$ROOT/logs/med_0801_validate_${suffix}.log"

    if [[ -f "$complete_marker" && -d "$output_1" && -d "$output_2" ]]; then
        printf 'Reusing validated local sharded dataset: %s\n' "$suffix"
        return
    fi
    if [[ -e "$output_1" || -e "$output_2" || -e "$complete_marker" ]]; then
        printf 'ERROR: incomplete local sharded output requires manual inspection: %s\n' \
            "$suffix" >&2
        exit 1
    fi

    mkdir -p "$LOCAL_BASE"
    : >"$build_log"
    printf 'Building sharded %s with annotation mode %s at %s\n' \
        "$suffix" "$mode" "$(date --iso-8601=seconds)"
    conda run --no-capture-output -n rr \
        python -m src.data_processing.process_pickles_to_lmdb \
        --controller diffik --domain sim \
        --task one_leg round_table lamp \
        --source rollout --randomness med --demo-outcome success \
        --suffix "$SOURCE_SUFFIX" \
        --output-dir "$output_1" --output-suffix "$suffix" \
        --task-episode-limit one_leg=200 round_table=200 lamp=200 \
        --num-pickles 360 --offset 0 \
        --image-annotation-mode "$mode" --provenance-json "$PROVENANCE" \
        --n-cpus 2 --batch-size 2 --map-size-gb 300 --debug-storage-stats \
        2>&1 | tee -a "$build_log"
    conda run --no-capture-output -n rr \
        python -m src.data_processing.process_pickles_to_lmdb \
        --controller diffik --domain sim \
        --task one_leg round_table lamp \
        --source rollout --randomness med --demo-outcome success \
        --suffix "$SOURCE_SUFFIX" \
        --output-dir "$output_2" --output-suffix "$suffix" \
        --task-episode-limit one_leg=200 round_table=200 lamp=200 \
        --num-pickles 240 --offset 360 \
        --image-annotation-mode "$mode" --provenance-json "$PROVENANCE" \
        --n-cpus 2 --batch-size 2 --map-size-gb 200 --debug-storage-stats \
        2>&1 | tee -a "$build_log"

    conda run --no-capture-output -n rr \
        python scripts/validate_lmdb_dataset.py --full-stats "$output_1" "$output_2" \
        2>&1 | tee "$validate_log"
    conda run --no-capture-output -n rr \
        python scripts/validate_med_0801_lmdb.py "$output_1" "$output_2" \
        --suffix "$suffix" --annotation-mode "$mode" \
        --source-manifest "$ROOT/logs/med_0801_source_manifest.sha256" \
        2>&1 | tee -a "$validate_log"
    touch "$complete_marker"
}

upload_dataset() {
    local suffix="$1"
    local host_number="$2"
    local host="zju_4090_${host_number}"
    local local_path="$LOCAL_BASE/${suffix}.lmdb"
    local local_marker="$LOCAL_BASE/.${suffix}.local-validated"
    local remote_path="$REMOTE_BASE/${suffix}.lmdb"
    local upload_marker="$remote_path/.med0801-upload-incomplete"
    local remote_complete="$remote_path/.med0801-upload-complete"
    local local_sha local_sha_cache remote_sha bytes available required start_epoch end_epoch duration mbps

    [[ -f "$local_marker" && -f "$local_path/data.mdb" ]] || {
        printf 'ERROR: local dataset is not validated: %s\n' "$local_path" >&2
        exit 1
    }

    bytes="$(stat -c %s "$local_path/data.mdb")"
    local_sha_cache="$LOCAL_BASE/.${suffix}.data-sha256"
    local_sha="$(sha256_cached "$local_path/data.mdb" "$local_sha_cache")"

    ssh_clean -o BatchMode=yes "$host" \
        "mkdir -p '$remote_path'; if test -f '$remote_complete'; then cat '$remote_complete'; elif test -z \"\$(find '$remote_path' -mindepth 1 -maxdepth 1 ! -name .med0801-upload-incomplete -print -quit)\"; then touch '$upload_marker'; else echo 'ERROR: unrecognized existing destination $remote_path' >&2; exit 2; fi; findmnt -T '$remote_path' -no TARGET,SOURCE,FSTYPE; df -h '$remote_path'"

    if ssh_clean -o BatchMode=yes "$host" "test -f '$remote_complete'"; then
        remote_sha="$(ssh_clean -o BatchMode=yes "$host" "awk -F= '/^sha256=/{print \$2}' '$remote_complete'")"
        if [[ "$remote_sha" == "$local_sha" ]]; then
            printf 'Remote dataset already complete and matches: %s:%s\n' "$host" "$remote_path"
            rm -rf -- "$local_path"
            rm -f -- "$local_marker" "$local_sha_cache"
            return
        fi
        printf 'ERROR: completed remote hash differs for %s\n' "$suffix" >&2
        exit 1
    fi

    available="$(ssh_clean -o BatchMode=yes "$host" "df -B1 --output=avail '$remote_path' | tail -n 1")"
    required=$((bytes + 10 * 1000 * 1000 * 1000))
    if (( available < required )); then
        printf 'ERROR: %s:%s has %s bytes free; need at least %s for %s\n' \
            "$host" "$remote_path" "$available" "$required" "$suffix" >&2
        exit 1
    fi

    start_epoch="$(date +%s)"
    env -u LD_LIBRARY_PATH rsync -aS --exclude lock.mdb --bwlimit=100000 --info=progress2,stats2 \
        "$local_path/" "$host:$remote_path/" \
        2>&1 | tee "$ROOT/logs/med_0801_upload_${suffix}.log"
    end_epoch="$(date +%s)"
    duration=$((end_epoch - start_epoch))
    if (( duration < 1 )); then duration=1; fi
    mbps="$(awk -v bytes="$bytes" -v seconds="$duration" \
        'BEGIN {printf "%.2f", bytes / seconds / 1000000}')"

    remote_sha="$(ssh_clean -o BatchMode=yes "$host" "sha256sum '$remote_path/data.mdb' | awk '{print \$1}'")"
    if [[ "$remote_sha" != "$local_sha" ]]; then
        printf 'ERROR: remote hash mismatch for %s: local=%s remote=%s\n' \
            "$suffix" "$local_sha" "$remote_sha" >&2
        exit 1
    fi
    ssh_clean -o BatchMode=yes "$host" \
        "cd /mnt/nas/share/home/hy/robust-rearrangement-custom && DATA_DIR_PROCESSED='$REMOTE_DATA_ROOT' /mnt/nas/share/home/hy/miniconda3/bin/conda run --no-capture-output -n rr python scripts/validate_lmdb_dataset.py --sample-episodes 5 '$remote_path'" \
        2>&1 | tee -a "$ROOT/logs/med_0801_upload_${suffix}.log"
    ssh_clean -o BatchMode=yes "$host" \
        "printf 'sha256=%s\\nbytes=%s\\nuploaded_at=%s\\n' '$local_sha' '$bytes' '$(date --iso-8601=seconds)' >'$remote_complete'; rm -f '$upload_marker'"

    record_event \
        "$(date --iso-8601=seconds)" upload "$suffix" "$host_number" \
        "$remote_path" "$bytes" "$duration" "$mbps"
    printf 'Verified upload %s -> %s:%s, %s bytes in %ss (%s MB/s)\n' \
        "$suffix" "$host" "$remote_path" "$bytes" "$duration" "$mbps"

    rm -rf -- "$local_path"
    rm -f -- "$local_marker" "$local_sha_cache"
}

upload_sharded_dataset_228() {
    local suffix="$1"
    local host=zju_4090_228
    local local_1="$LOCAL_BASE/${suffix}-1.lmdb"
    local local_2="$LOCAL_BASE/${suffix}-2.lmdb"
    local local_marker="$LOCAL_BASE/.${suffix}.local-validated"
    local remote_1="$REMOTE_228_BASE/${suffix}-1.lmdb"
    local remote_2="$REMOTE_228_AUX_BASE/${suffix}-2.lmdb"
    local logical_2="$REMOTE_228_BASE/${suffix}-2.lmdb"
    local remote_complete="$REMOTE_228_BASE/.${suffix}.med0801-upload-complete"
    local upload_log="$ROOT/logs/med_0801_upload_${suffix}.log"
    local sha_1 sha_2 sha_cache_1 sha_cache_2 remote_sha_1 remote_sha_2 bytes_1 bytes_2 total_bytes
    local available_1 available_2 required_1 required_2
    local start_epoch end_epoch duration mbps

    [[ -f "$local_marker" && -f "$local_1/data.mdb" && -f "$local_2/data.mdb" ]] || {
        printf 'ERROR: local sharded dataset is not validated: %s\n' "$suffix" >&2
        exit 1
    }

    bytes_1="$(stat -c %s "$local_1/data.mdb")"
    bytes_2="$(stat -c %s "$local_2/data.mdb")"
    total_bytes=$((bytes_1 + bytes_2))
    sha_cache_1="$LOCAL_BASE/.${suffix}-1.data-sha256"
    sha_cache_2="$LOCAL_BASE/.${suffix}-2.data-sha256"
    sha_1="$(sha256_cached "$local_1/data.mdb" "$sha_cache_1")"
    sha_2="$(sha256_cached "$local_2/data.mdb" "$sha_cache_2")"

    ssh_clean -o BatchMode=yes "$host" \
        "mkdir -p '$REMOTE_228_BASE' '$REMOTE_228_AUX_BASE'; if test -f '$remote_complete'; then cat '$remote_complete'; else for path in '$remote_1' '$remote_2'; do if test -e \"\$path\"; then echo \"ERROR: unrecognized existing shard \$path\" >&2; exit 2; fi; done; mkdir -p '$remote_1' '$remote_2'; touch '$remote_1/.med0801-upload-incomplete' '$remote_2/.med0801-upload-incomplete'; fi; findmnt -T '$remote_1' -no TARGET,SOURCE,FSTYPE; findmnt -T '$remote_2' -no TARGET,SOURCE,FSTYPE"

    if ssh_clean -o BatchMode=yes "$host" "test -f '$remote_complete'"; then
        remote_sha_1="$(ssh_clean -o BatchMode=yes "$host" "awk -F= '/^sha256_1=/{print \$2}' '$remote_complete'")"
        remote_sha_2="$(ssh_clean -o BatchMode=yes "$host" "awk -F= '/^sha256_2=/{print \$2}' '$remote_complete'")"
        if [[ "$remote_sha_1" == "$sha_1" && "$remote_sha_2" == "$sha_2" ]]; then
            printf 'Remote sharded dataset already complete and matches: %s\n' "$suffix"
            rm -rf -- "$local_1" "$local_2"
            rm -f -- "$local_marker" "$sha_cache_1" "$sha_cache_2"
            return
        fi
        printf 'ERROR: completed remote shard hash differs for %s\n' "$suffix" >&2
        exit 1
    fi

    available_1="$(ssh_clean -o BatchMode=yes "$host" "df -B1 --output=avail '$remote_1' | tail -n 1")"
    available_2="$(ssh_clean -o BatchMode=yes "$host" "df -B1 --output=avail '$remote_2' | tail -n 1")"
    required_1=$((bytes_1 + 10 * 1000 * 1000 * 1000))
    required_2=$((bytes_2 + 10 * 1000 * 1000 * 1000))
    if (( available_1 < required_1 || available_2 < required_2 )); then
        printf 'ERROR: 228 shard capacity insufficient: root %s/%s bytes, /data %s/%s bytes\n' \
            "$available_1" "$required_1" "$available_2" "$required_2" >&2
        exit 1
    fi

    : >"$upload_log"
    start_epoch="$(date +%s)"
    env -u LD_LIBRARY_PATH rsync -aS --exclude lock.mdb --bwlimit=100000 --info=progress2,stats2 \
        "$local_1/" "$host:$remote_1/" 2>&1 | tee -a "$upload_log"
    env -u LD_LIBRARY_PATH rsync -aS --exclude lock.mdb --bwlimit=100000 --info=progress2,stats2 \
        "$local_2/" "$host:$remote_2/" 2>&1 | tee -a "$upload_log"
    end_epoch="$(date +%s)"
    duration=$((end_epoch - start_epoch))
    if (( duration < 1 )); then duration=1; fi
    mbps="$(awk -v bytes="$total_bytes" -v seconds="$duration" \
        'BEGIN {printf "%.2f", bytes / seconds / 1000000}')"

    remote_sha_1="$(ssh_clean -o BatchMode=yes "$host" "sha256sum '$remote_1/data.mdb' | awk '{print \$1}'")"
    remote_sha_2="$(ssh_clean -o BatchMode=yes "$host" "sha256sum '$remote_2/data.mdb' | awk '{print \$1}'")"
    if [[ "$remote_sha_1" != "$sha_1" || "$remote_sha_2" != "$sha_2" ]]; then
        printf 'ERROR: remote shard hash mismatch for %s\n' "$suffix" >&2
        exit 1
    fi

    ssh_clean -o BatchMode=yes "$host" \
        "ln -s '$remote_2' '$logical_2'; cd /mnt/nas/share/home/hy/robust-rearrangement-custom && DATA_DIR_PROCESSED='$REMOTE_228_DATA_ROOT' /mnt/nas/share/home/hy/miniconda3/bin/conda run --no-capture-output -n rr python scripts/validate_lmdb_dataset.py --sample-episodes 5 '$remote_1' '$logical_2'" \
        2>&1 | tee -a "$upload_log"
    ssh_clean -o BatchMode=yes "$host" \
        "printf 'sha256_1=%s\\nsha256_2=%s\\nbytes_1=%s\\nbytes_2=%s\\nuploaded_at=%s\\n' '$sha_1' '$sha_2' '$bytes_1' '$bytes_2' '$(date --iso-8601=seconds)' >'$remote_complete'; rm -f '$remote_1/.med0801-upload-incomplete' '$remote_2/.med0801-upload-incomplete'"

    record_event \
        "$(date --iso-8601=seconds)" upload-sharded "$suffix" 228 \
        "$remote_1 + $remote_2" "$total_bytes" "$duration" "$mbps"
    printf 'Verified sharded upload %s -> 228, %s bytes in %ss (%s MB/s)\n' \
        "$suffix" "$total_bytes" "$duration" "$mbps"
    rm -rf -- "$local_1" "$local_2"
    rm -f -- "$local_marker" "$sha_cache_1" "$sha_cache_2"
}

cd "$ROOT"
wait_for_sources
require_local_build_capacity

if [[ ! -f "$ROOT/logs/.med_0801_provenance_finalized" ]]; then
    conda run --no-capture-output -n rr \
        python scripts/finalize_med_0801_provenance.py \
        2>&1 | tee "$ROOT/logs/med_0801_finalize_provenance.log"
fi

for suffix in "${SUFFIXES[@]}"; do
    if [[ "$suffix" == rgbd-skill-point-colored ]]; then
        build_sharded_dataset "$suffix" "${MODES[$suffix]}"
        upload_sharded_dataset_228 "$suffix"
    else
        build_dataset "$suffix" "${MODES[$suffix]}"
        upload_dataset "$suffix" "${HOSTS[$suffix]}"
    fi
done

printf 'All five med-0801 LMDB datasets built, validated, and uploaded at %s\n' \
    "$(date --iso-8601=seconds)"
