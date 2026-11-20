#!/usr/bin/env bash
main() {
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
OUT=${OUT:?output directory}
GPUS=${GPUS:-0,1}
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
FUSION=${FUSION:?the fusion-stage checkpoint, <its OUT>/ckpt}
export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p "$OUT"
torchrun --nproc_per_node "$NGPU" -m smind.train.perspective --config configs/perspective.yaml --fusion_ckpt "$FUSION" \
  --out "$OUT" "$@" 2>&1 | tee -a "$OUT/train.log"
}
main "$@"; exit $?
