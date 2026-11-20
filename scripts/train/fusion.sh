#!/usr/bin/env bash
main() {
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
if [ ! -d "$MODEL" ]; then
  [ -e "models/${MODEL##*/}/config.json" ] || hf download "$MODEL" --local-dir "models/${MODEL##*/}"
  MODEL=models/${MODEL##*/}
fi
OUT=${OUT:?output directory}
GPUS=${GPUS:-0,1}
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p "$OUT"
{
  torchrun --nproc_per_node "$NGPU" -m smind.train.fusion --config configs/fusion.yaml --model "$MODEL" \
    --out "$OUT/pretrain" "$@"
  python -m smind.train.refit --config configs/refit.yaml --ckpt "$OUT/pretrain/ckpt" --out "$OUT/ckpt"
} 2>&1 | tee -a "$OUT/train.log"
}
main "$@"; exit $?
