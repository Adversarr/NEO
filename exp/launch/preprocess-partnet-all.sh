#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=""
OUTPUT_ROOT=""
H5_COMPRESSION="lzf"
H5_COMPRESSION_OPTS="4"
NUM_WORKERS="$(nproc || echo 1)"
DRY_RUN="0"

usage() {
  echo "Usage: $0 --root-dir DIR --output-root DIR [--compression lzf|gzip|none] [--compression-opts N] [--num-workers N] [--dry-run]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-dir)
      ROOT_DIR="$2"; shift 2;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2;;
    --compression)
      H5_COMPRESSION="$2"; shift 2;;
    --compression-opts)
      H5_COMPRESSION_OPTS="$2"; shift 2;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2;;
    --dry-run)
      DRY_RUN="1"; shift 1;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$ROOT_DIR" || -z "$OUTPUT_ROOT" ]]; then
  usage; exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Root dir not found: $ROOT_DIR" >&2; exit 1
fi

mkdir -p "$OUTPUT_ROOT"

for cls in "$ROOT_DIR"/*; do
  [[ -d "$cls" ]] || continue
  cls_name="$(basename "$cls")"
  echo "Processing class: $cls_name"
  if [[ "$DRY_RUN" == "1" ]]; then
    python exp/downstream/preprocess_partnet.py \
      --data-dir "$cls" \
      --output-dir "$OUTPUT_ROOT" \
      --h5-compression "$H5_COMPRESSION" \
      --h5-compression-opts "$H5_COMPRESSION_OPTS" \
      --num-workers "$NUM_WORKERS" \
      --dry-run
  else
    python exp/downstream/preprocess_partnet.py \
      --data-dir "$cls" \
      --output-dir "$OUTPUT_ROOT" \
      --h5-compression "$H5_COMPRESSION" \
      --h5-compression-opts "$H5_COMPRESSION_OPTS" \
      --num-workers "$NUM_WORKERS"
  fi
done