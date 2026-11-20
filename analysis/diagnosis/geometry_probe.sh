#!/usr/bin/env bash
main() {
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
MODEL=${MODEL:?the model to probe (a model directory)}
OUT=${OUT:?output directory}
GPU=${GPU:-0}
D=data/smind/probe
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT"
{
  [ -e "$OUT/probe/DONE" ] || python -m smind.train.fusion --model "$MODEL" --layout 2,4 --out "$OUT/probe" \
    --lora_rank 0 --conf_alpha 0.5 --epochs 4 --evals 4 --seed 17 --micro_tokens "${MICRO_TOKENS:-8192}" \
    --eval_micro_tokens "${EVAL_MICRO_TOKENS:-4096}" \
    --train $D/maps_train $D/maps_scannet_train --dev $D/maps_dev $D/maps_scannet_dev
  python analysis/diagnosis/probe_scene.py --ckpt "$OUT/probe/ckpt" --maps $D --out "$OUT/geometry_probe.json"
} 2>&1 | tee -a "$OUT/probe.log"
}
main "$@"; exit $?
