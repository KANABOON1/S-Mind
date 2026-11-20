#!/usr/bin/env bash
main() {
set -euo pipefail
NAME=${1:?the model name given to make_configs.py}
GPU=${GPU:-0}
cd "$(dirname "$0")/../../../third_party/s-space"
export CUDA_VISIBLE_DEVICES=$GPU
for cfg in outputs/smind/configs/"$NAME"/*.json; do
  out=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$cfg")
  mkdir -p "$out"
  echo "$(date -Is) start $cfg"
  scripts/run_experiment.sh "$cfg" >> "$out/run.log" 2>&1
  echo "$(date -Is) done $cfg"
done
}
main "$@"; exit $?
