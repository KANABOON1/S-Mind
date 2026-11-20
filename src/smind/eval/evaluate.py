import argparse
import json
import subprocess
import sys
from pathlib import Path

from smind.eval.sets import BENCHES, ROOT, set_path, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output directory: one folder per benchmark")
    ap.add_argument("--sets", nargs="+", default=BENCHES)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--kv_gb", type=float, default=None)
    ap.add_argument("--batched_tokens", type=int, default=None,
                    help="vLLM's tokens a step, which also sizes its encoder cache (generate.py)")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=4096)
    args = ap.parse_args()
    for s in args.sets:
        out_dir = Path(args.out) / s
        out_dir.mkdir(parents=True, exist_ok=True)
        gen = out_dir / "predictions.jsonl"
        if not gen.exists():
            subprocess.run([sys.executable, "-m", "smind.eval.generate", "--model", args.model, "--inputs",
                            str(set_path(s)), "--out", str(gen), "--temperature", "0", "--max_tokens", str(args.max_tokens),
                            "--gpu_mem", str(args.gpu_mem), "--tp", str(args.tp), "--max_images", "32"] + (["--kv_gb", str(args.kv_gb)] if args.kv_gb else [])
                           + (["--batched_tokens", str(args.batched_tokens)] if args.batched_tokens else [])
                           + (["--enable_thinking"] if args.enable_thinking else []),
                           check=True, cwd=ROOT)
        s_ = summary([json.loads(line) for line in open(gen)])
        print(s, json.dumps({k: v for k, v in s_.items() if k != "per_task"}))
        (out_dir / "summary.json").write_text(json.dumps(dict(model=args.model, **s_), indent=1))


if __name__ == "__main__":
    main()
