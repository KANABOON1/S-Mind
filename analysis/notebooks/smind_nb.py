import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/notebooks"
HF_REPO = "KANABOON1/S-Mind"
BASE_REPO = {"4b": "Qwen/Qwen3.5-4B", "9b": "Qwen/Qwen3.5-9B"}


def setup():
    os.chdir(ROOT)
    RUNS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.precision", 2)


def checkpoint(size, stage):
    path = ROOT / "models/S-Mind" / f"qwen3.5-{size}" / stage
    if not (path / "spatial_config.json").exists():
        subprocess.run(["hf", "download", HF_REPO, "--include", f"qwen3.5-{size}/{stage}/*",
                        "--local-dir", str(ROOT / "models/S-Mind")], check=True)
    return path


def base_model(size):
    path = ROOT / "models" / BASE_REPO[size].split("/")[1]
    if not (path / "config.json").exists():
        subprocess.run(["hf", "download", BASE_REPO[size], "--local-dir", str(path)], check=True)
    return path


def need(*paths, build="the Data section of README.md"):
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing}: build them first ({build})")


def run(cmd, log, done=None, gpu="0", env=None, pythonpath=None):
    done = Path(done) if done else None
    if done is not None and done.exists():
        print(f"done already: {done.relative_to(ROOT)}")
        return
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=pythonpath or f"{ROOT / 'src'}:{ROOT}", CUDA_VISIBLE_DEVICES=str(gpu),
               CUDA_DEVICE_ORDER="PCI_BUS_ID", TOKENIZERS_PARALLELISM="false", **(env or {}))
    cmd = [sys.executable if c == "python" else str(c) for c in cmd]
    shown = ["python" if c == sys.executable else os.path.relpath(c, ROOT) if c.startswith(str(ROOT)) else c for c in cmd]
    print("$", " ".join(shown))
    with open(log, "a") as h:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=h, stderr=subprocess.STDOUT).returncode
    if code:
        print("".join(open(log).readlines()[-30:]))
        raise RuntimeError(f"failed with exit code {code}; full log: {log}")


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in open(path)]


def compare(rows, paper=None):
    df = pd.DataFrame({"this run": pd.Series(rows, dtype=float)})
    if paper:
        df["paper"] = pd.Series(paper, dtype=float).reindex(df.index)
    return df
