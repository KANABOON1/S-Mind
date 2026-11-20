import argparse
import json
import os
from pathlib import Path

from smind import formats

def resolve_model(name_or_path):
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_path)


def _content(rec, text_fn):
    content = [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in rec["images"]]
    content.append({"type": "text", "text": text_fn(rec, image_token="")})
    return content


def build_messages(rec):
    return [{"role": "user", "content": _content(rec, formats.user_text)}]


def extract(text, rec):
    return formats.extract_answer(text, len(rec["options"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--max_images", type=int, default=32)
    ap.add_argument("--kv_gb", type=float, default=None)
    ap.add_argument("--batched_tokens", type=int, default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    recs = []
    for path in args.inputs:
        recs += [json.loads(l) for l in open(path)]
    if args.limit:
        recs = recs[: args.limit]
    model = resolve_model(args.model)
    llm = LLM(
        model=model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len, limit_mm_per_prompt={"image": args.max_images, "video": 0},
        allowed_local_media_path=str((Path(__file__).resolve().parents[3] / "data").resolve()), seed=args.seed,
        enable_prefix_caching=True, max_num_seqs=args.max_num_seqs,
        **({"kv_cache_memory_bytes": int(args.kv_gb * 2**30)} if args.kv_gb else {}),
        **({"max_num_batched_tokens": args.batched_tokens} if args.batched_tokens else {}),
    )
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                        seed=args.seed)
    msgs = [build_messages(r) for r in recs]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": args.enable_thinking})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_valid = n_all = 0
    with open(args.out, "w") as f:
        for rec, out in zip(recs, outs):
            for i, o in enumerate(out.outputs):
                ext = extract(o.text, rec)
                ok = ext == rec["answer"]
                n_all += 1
                n_ok += ok
                n_valid += ext is not None
                f.write(json.dumps(dict(
                    question_id=rec["question_id"], source=rec["source"], task=rec["task"], sample_idx=i,
                    response=o.text, extracted=ext, answer=rec["answer"], correct=ok,
                    finish_reason=o.finish_reason, n_prompt_tokens=len(out.prompt_token_ids),
                    n_response_tokens=len(o.token_ids),
                ), ensure_ascii=False) + "\n")
    print(json.dumps(dict(model=model, n=n_all, acc=n_ok / n_all, valid=n_valid / n_all, out=args.out)))


if __name__ == "__main__":
    main()
