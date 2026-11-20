#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu129
pip install https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
pip install -e ".[notebooks]"

clone() {
  [ -d "third_party/$2" ] || git clone "$1" "third_party/$2"
  git -C "third_party/$2" checkout -q "$3"
}
clone https://github.com/facebookresearch/vggt.git vggt 9e4fa662a8893ed348d048e8b57816c12593448b
clone https://github.com/mll-lab-nu/MindCube.git MindCube b8b7062adf6d3e49d588a7d014a0a787553d09ec
clone https://github.com/fudan-zvg/spar.git spar e3060b0b36ab27ac3f608929a145c6c6546a849a
hf download facebook/VGGT-1B model.pt --local-dir models/VGGT-1B
