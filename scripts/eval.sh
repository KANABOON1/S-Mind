#!/usr/bin/env bash
main() {
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
CKPT=${CKPT:-qwen3.5-4b/rl}
OUT=${OUT:-runs/eval/${CKPT//\//-}}
[ -e "models/S-Mind/$CKPT/spatial_config.json" ] || hf download KANABOON1/S-Mind --include "$CKPT/*" --local-dir models/S-Mind
python -m smind.eval.generate_ours --ckpt "models/S-Mind/$CKPT" --out "$OUT"
python -m smind.eval.summarize --out "$OUT"
}
main "$@"; exit $?
