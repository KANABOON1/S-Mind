import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SSPACE = (REPO / "third_party/s-space").resolve()
OUT = REPO / "runs/sspace"
LINK = "outputs/smind"
BASE = ".cache/assets/models/qwen35_4b"


def model_path(name, path):
    if path is None:
        if name != "base":
            raise SystemExit(f"--model_path is required for {name!r}; only base defaults to S-Space's Qwen3.5-4B")
        return BASE
    return os.path.relpath(Path(path).resolve(), SSPACE)


def link_outputs():
    OUT.mkdir(parents=True, exist_ok=True)
    link = SSPACE / LINK
    if link.is_symlink() or link.exists():
        if link.resolve() != OUT.resolve():
            raise SystemExit(f"{link} exists and does not point to {OUT}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(OUT)
