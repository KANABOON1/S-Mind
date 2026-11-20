import argparse
import json
import re
from pathlib import Path

from PIL import Image

from smind.eval.generate import extract
from smind.eval.sets import BENCHES, SETS, set_path
from smind.model.checkpoint import SpatialCheckpoint
from smind.model.insertion import Insertion, question_text



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a checkpoint with spatial_config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=BENCHES, help=f"names in {sorted(SETS)}, or jsonl paths")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--max_images", type=int, default=32)
    ap.add_argument("--no_prefix_cache", action="store_true",
                    help="disable vLLM prefix caching")
    ap.add_argument("--chunk", type=int, default=200,
                    help="questions answered together; each chunk is appended to the predictions when it ends, and a "
                         "rerun skips what is written")
    ap.add_argument("--batched_tokens", type=int,
                    help="vLLM max_num_batched_tokens")
    ap.add_argument("--max_seqs", type=int, default=128,
                    help="vLLM max_num_seqs")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("I", "N"),
                    help="every N-th question from the I-th, written as predictions_shard<I>.jsonl (parallel GPUs)")
    ap.add_argument("--max_rounds", type=int, default=256,
                    help="insertion rounds per answer; past it the answer counts as truncated")
    args = ap.parse_args()
    ckpt = SpatialCheckpoint.load(args.ckpt)
    if ckpt.insertion is None:
        raise ValueError(f"{ckpt.path} is a fusion-stage checkpoint: it was not trained to answer with spatial tokens")
    from vllm import LLM
    llm = LLM(model=str(ckpt.path), gpu_memory_utilization=args.gpu_mem, max_model_len=16384,
              enable_prefix_caching=not args.no_prefix_cache,
              limit_mm_per_prompt={"image": args.max_images, "video": 0}, max_num_seqs=args.max_seqs, seed=0,
              **({"max_num_batched_tokens": args.batched_tokens} if args.batched_tokens else {}))
    tok = llm.get_tokenizer()
    ckpt.token_ids(tok)
    ins = Insertion(ckpt.layout, tok)
    for name in args.sets:
        recs_all = [json.loads(l) for l in open(set_path(name))][args.shard[0]::args.shard[1]]
        d = Path(args.out) / (name if name in SETS else Path(name).stem)
        d.mkdir(parents=True, exist_ok=True)
        pred = d / f"predictions_shard{args.shard[0]}.jsonl"
        written = {json.loads(l)["question_id"] for l in open(pred)} if pred.exists() else set()
        left = [r for r in recs_all if r["question_id"] not in written]
        print(name, f"{len(written)} written, {len(left)} to go", flush=True)
        for start in range(0, len(left), args.chunk):
            rows = answer_chunk(llm, tok, left[start:start + args.chunk], args, ckpt.layout, ins)
            with open(pred, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(name, f"chunk done: {len(written) + start + len(rows)}/{len(recs_all)}", flush=True)
        rows = [json.loads(l) for l in open(pred)]
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(name, json.dumps(dict(n=len(rows), acc=acc, triggers=sum(r["triggers"] for r in rows) / len(rows),
                                    truncated=sum(r["finish_reason"] == "length" for r in rows) / len(rows))), flush=True)


def answer_chunk(llm, tok, recs, args, layout, ins):
    rows = generate_answers(llm, tok, recs, args, layout, ins)
    for rec, row in zip(recs, rows):
        row.update(source=rec["source"], task=rec["task"], answer=rec["answer"],
                   correct=row["extracted"] == rec["answer"])
    return rows


def generate_answers(llm, tok, recs, args, layout, ins, *, use_tqdm=True):
    from vllm import SamplingParams
    prompts, images = [], []
    for rec in recs:
        content = [dict(type="image") for _ in rec["images"]] + [dict(type="text", text=question_text(rec, layout))]
        prompts.append(tok.apply_chat_template([dict(role="user", content=content)], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False))
        loaded = []
        for path in rec["images"]:
            with Image.open(path) as im:
                loaded.append(im.convert("RGB"))
        images.append(loaded)
    answer = [[] for _ in recs]
    done, triggers, finish = [False] * len(recs), [0] * len(recs), ["length"] * len(recs)
    rounds = 0
    while not all(done):
        todo = [i for i in range(len(recs)) if not done[i]]
        rounds += 1
        if rounds > args.max_rounds:
            break
        if rounds % 25 == 0:
            print(f"round {rounds}: {len(todo)} answers going", flush=True)
        reqs, sps = [], []
        for i in todo:
            cont = re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", prompts[i] + tok.decode(answer[i], skip_special_tokens=False))
            reqs.append(dict(prompt=cont, multi_modal_data=dict(image=images[i])))
            n = sum(t not in ins.inserted for t in answer[i])
            sps.append(SamplingParams(temperature=0.0, max_tokens=max(1, args.max_tokens - n),
                                      stop_token_ids=sorted(ins.stop_ids), skip_special_tokens=False))
        outs = llm.generate(reqs, sps, use_tqdm=use_tqdm)
        for i, o in zip(todo, outs):
            ids = list(o.outputs[0].token_ids)
            answer[i] += ids
            n = sum(t not in ins.inserted for t in answer[i])
            if ids and ids[-1] in ins.stop_ids and n < args.max_tokens:
                run = ins.due(answer[i], len(recs[i]["images"]))
                answer[i] += run
                triggers[i] += bool(run)
                continue
            done[i] = True
            finish[i] = "stop" if o.outputs[0].finish_reason == "stop" else "length"
    rows = []
    for rec, ans, t, f in zip(recs, answer, triggers, finish):
        text = tok.decode(ins.text_ids(ans), skip_special_tokens=True)
        ext = extract(text, rec)
        rows.append(dict(question_id=rec["question_id"], extracted=ext, response=text,
                         triggers=t, finish_reason=f))
    return rows


if __name__ == "__main__":
    main()
