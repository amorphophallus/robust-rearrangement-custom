#!/usr/bin/env -S -u LD_LIBRARY_PATH bash
set -euo pipefail

# Avoid Isaac Gym/conda runtime library paths leaking into bash/modelscope CLI.
unset LD_LIBRARY_PATH

ACTION="all"
VLM_DIR="${VLM_DIR:-/data/hy/robust-rearrangement/data/processed/vlm}"
UPLOAD_DIR="${UPLOAD_DIR:-/data/hy/robust-rearrangement/data/processed/vlm_modelscope_upload}"
MAX_WORKERS="${MAX_WORKERS:-1}"
MODELSCOPE_BYPASS_PROXY="${MODELSCOPE_BYPASS_PROXY:-0}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload robust rearrangement VLM SFT dataset}"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/upload_vlm_to_modelscope.sh [prepare|upload|upload-index-update|upload-readme|upload-light|all] [--local-dir DIR] [--upload-dir DIR]
  scripts/upload_vlm_to_modelscope.sh --action all --local-dir DIR

Environment:
  MODELSCOPE_TOKEN   Optional compatibility token passed as a global CLI option.
  MODELSCOPE_API_TOKEN
                     Optional official ModelScope token environment variable.
                     If neither is set, the CLI uses `modelscope login` credentials.
  REPO_ID            Required for upload, e.g. <namespace>/<dataset_name>.
  VLM_DIR            Optional. Default: /data/hy/robust-rearrangement/data/processed/vlm
  UPLOAD_DIR         Optional. Default: /data/hy/robust-rearrangement/data/processed/vlm_modelscope_upload
  MAX_WORKERS        Optional. Default: 1 (serial uploads are more reliable for large files).
  MODELSCOPE_BYPASS_PROXY
                     Set to 1 to connect directly and ignore all HTTP(S)/ALL proxy variables.
  COMMIT_MESSAGE     Optional.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    prepare|upload|upload-index-update|upload-readme|upload-light|all)
      ACTION="$1"
      shift
      ;;
    --action)
      ACTION="${2:?Missing value for --action}"
      shift 2
      ;;
    --local-dir|--vlm-dir)
      VLM_DIR="${2:?Missing value for $1}"
      shift 2
      ;;
    --upload-dir)
      UPLOAD_DIR="${2:?Missing value for --upload-dir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

modelscope_cli() {
  local -a command=(modelscope)
  if [[ -n "${MODELSCOPE_TOKEN:-}" ]]; then
    command+=(--token "${MODELSCOPE_TOKEN}")
  fi

  if [[ "${MODELSCOPE_BYPASS_PROXY}" == "1" ]]; then
    env -u http_proxy -u https_proxy -u all_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u no_proxy -u NO_PROXY \
      "${command[@]}" "$@"
  else
    "${command[@]}" "$@"
  fi
}

prepare_upload_dir() {
  require_file "${VLM_DIR}/README.md"
  require_file "${VLM_DIR}/manifest.json"
  require_file "${VLM_DIR}/messages.jsonl"
  require_file "${VLM_DIR}/qwen_llava_sharegpt.json"
  require_file "${VLM_DIR}/llamafactory_base.json"
  require_file "${VLM_DIR}/llamafactory_base_dataset_info.json"

  rm -rf "${UPLOAD_DIR}"
  mkdir -p "${UPLOAD_DIR}"

  cp "${VLM_DIR}/README.md" "${UPLOAD_DIR}/README.md"
  cp "${VLM_DIR}/manifest.json" "${UPLOAD_DIR}/manifest.json"
  cp "${VLM_DIR}/messages.jsonl" "${UPLOAD_DIR}/messages.jsonl"
  cp "${VLM_DIR}/qwen_llava_sharegpt.json" "${UPLOAD_DIR}/qwen_llava_sharegpt.json"
  cp "${VLM_DIR}/llamafactory_base.json" "${UPLOAD_DIR}/llamafactory_base.json"
  cp "${VLM_DIR}/llamafactory_base_dataset_info.json" "${UPLOAD_DIR}/llamafactory_base_dataset_info.json"

  tar -C "${VLM_DIR}" -cf "${UPLOAD_DIR}/images_one_leg.tar" images/one_leg
  tar -C "${VLM_DIR}" -cf "${UPLOAD_DIR}/images_round_table.tar" images/round_table
  tar -C "${VLM_DIR}" -cf "${UPLOAD_DIR}/images_lamp.tar" images/lamp

  tar -C "${VLM_DIR}" -cf "${UPLOAD_DIR}/depth_one_leg.tar" depth/one_leg
  tar -C "${VLM_DIR}" -cf "${UPLOAD_DIR}/depth_round_table.tar" depth/round_table
  tar -C "${VLM_DIR}" -cf "${UPLOAD_DIR}/depth_lamp.tar" depth/lamp

  echo "Prepared upload directory: ${UPLOAD_DIR}"
  du -sh "${UPLOAD_DIR}"
  find "${UPLOAD_DIR}" -maxdepth 1 -type f -printf "%s %p\n" | sort -n
}

upload_to_modelscope() {
  if [[ -z "${REPO_ID:-}" ]]; then
    echo "Set REPO_ID, for example: export REPO_ID='<namespace>/<dataset_name>'" >&2
    exit 1
  fi
  if [[ ! -d "${UPLOAD_DIR}" ]]; then
    echo "Upload directory does not exist: ${UPLOAD_DIR}" >&2
    echo "Run: $0 prepare --local-dir '${VLM_DIR}'" >&2
    exit 1
  fi

  python -m pip show modelscope >/dev/null 2>&1 || python -m pip install -U modelscope

  modelscope_cli upload "${REPO_ID}" "${UPLOAD_DIR}" \
    --repo-type dataset \
    --commit-message "${COMMIT_MESSAGE}" \
    --use-cache \
    --max-workers "${MAX_WORKERS}"
}

upload_index_update_to_modelscope() {
  if [[ -z "${REPO_ID:-}" ]]; then
    echo "Set REPO_ID before uploading the index-only update." >&2
    exit 1
  fi
  require_file "${UPLOAD_DIR}/README.md"
  require_file "${UPLOAD_DIR}/manifest.json"
  require_file "${UPLOAD_DIR}/messages.jsonl"
  require_file "${UPLOAD_DIR}/qwen_llava_sharegpt.json"
  require_file "${UPLOAD_DIR}/llamafactory_base.json"
  require_file "${UPLOAD_DIR}/llamafactory_base_dataset_info.json"
  require_file "${UPLOAD_DIR}/preview/llamafactory_preview.jsonl"
  require_file "${UPLOAD_DIR}/rotation6d_enrichment_audit_20260831.json"

  if find "${UPLOAD_DIR}" -type f \( -name '*.tar' -o -name '*.tmp' \) -print -quit | grep -q .; then
    echo "Index-only update directory must not contain tar or temporary files: ${UPLOAD_DIR}" >&2
    exit 1
  fi

  python -m pip show modelscope >/dev/null 2>&1 || python -m pip install -U modelscope
  modelscope_cli upload "${REPO_ID}" "${UPLOAD_DIR}" \
    --repo-type dataset \
    --commit-message "${COMMIT_MESSAGE}" \
    --use-cache \
    --max-workers "${MAX_WORKERS}"
}

upload_readme_to_modelscope() {
  if [[ -z "${REPO_ID:-}" ]]; then
    echo "Set REPO_ID, for example: export REPO_ID='<namespace>/<dataset_name>'" >&2
    exit 1
  fi
  require_file "${UPLOAD_DIR}/README.md"

  python -m pip show modelscope >/dev/null 2>&1 || python -m pip install -U modelscope

  modelscope_cli upload "${REPO_ID}" "${UPLOAD_DIR}/README.md" \
    --repo-type dataset \
    --commit-message "${COMMIT_MESSAGE}"
}

upload_light_to_modelscope() {
  if [[ -z "${REPO_ID:-}" ]]; then
    echo "Set REPO_ID, for example: export REPO_ID='<namespace>/<dataset_name>'" >&2
    exit 1
  fi
  require_file "${UPLOAD_DIR}/README.md"
  require_file "${UPLOAD_DIR}/preview/llamafactory_preview.jsonl"

  python -m pip show modelscope >/dev/null 2>&1 || python -m pip install -U modelscope

  modelscope_cli upload "${REPO_ID}" "${UPLOAD_DIR}/README.md" "README.md" \
    --repo-type dataset \
    --commit-message "${COMMIT_MESSAGE}: README metadata"

  modelscope_cli upload "${REPO_ID}" "${UPLOAD_DIR}/preview/llamafactory_preview.jsonl" "preview/llamafactory_preview.jsonl" \
    --repo-type dataset \
    --commit-message "${COMMIT_MESSAGE}: preview subset"
}

case "${ACTION}" in
  prepare)
    prepare_upload_dir
    ;;
  upload)
    upload_to_modelscope
    ;;
  upload-index-update)
    upload_index_update_to_modelscope
    ;;
  upload-readme)
    upload_readme_to_modelscope
    ;;
  upload-light)
    upload_light_to_modelscope
    ;;
  all)
    prepare_upload_dir
    upload_to_modelscope
    ;;
  *)
    echo "Usage: $0 [prepare|upload|upload-index-update|upload-readme|upload-light|all]" >&2
    exit 1
    ;;
esac
