import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

from PIL import Image

from smind.eval.generate import extract
from smind.eval.sets import set_path
from smind.model.checkpoint import SpatialCheckpoint
from smind.model.insertion import Insertion, question_text


async def answer_ours(engine, tok, rec, ins, layout, args, qid):
    from vllm import SamplingParams
    content = [dict(type="image") for _ in rec["images"]] + [dict(type="text", text=question_text(rec, layout))]
    prompt = tok.apply_chat_template([dict(role="user", content=content)], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    images = [Image.open(p).convert("RGB") for p in rec["images"]]
    answer, triggers, finish = [], 0, "length"
    for rnd in range(args.max_rounds):
        cont = re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", prompt + tok.decode(answer, skip_special_tokens=False))
        n = sum(t not in ins.inserted for t in answer)
        sp = SamplingParams(temperature=0.0, max_tokens=max(1, args.max_tokens - n), stop_token_ids=sorted(ins.stop_ids),
                            skip_special_tokens=False)
        out = None
        async for out in engine.generate(dict(prompt=cont, multi_modal_data=dict(image=images)), sp, f"{qid}|{rnd}"):
            pass
        ids = list(out.outputs[0].token_ids)
        answer += ids
        n = sum(t not in ins.inserted for t in answer)
        if ids and ids[-1] in ins.stop_ids and n < args.max_tokens:
            run = ins.due(answer, len(rec["images"]))
            answer += run
            triggers += bool(run)
            continue
        finish = "stop" if out.outputs[0].finish_reason == "stop" else "length"
        break
    return tok.decode(ins.text_ids(answer), skip_special_tokens=True), triggers, finish


async def run_set(engine, tok, recs, ins, layout, args, pred):
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def one(rec):
        async with sem:
            t0 = time.perf_counter()
            text, trig, fin = await answer_ours(engine, tok, rec, ins, layout, args, rec["question_id"])
            wall = time.perf_counter() - t0
        ext = extract(text, rec)
        row = dict(question_id=rec["question_id"], source=rec["source"], task=rec["task"], answer=rec["answer"],
                   extracted=ext, correct=ext == rec["answer"], response=text, triggers=trig, finish_reason=fin,
                   wall_time_sec=round(wall, 3))
        async with lock:
            with open(pred, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    t0 = time.perf_counter()
    await asyncio.gather(*(one(r) for r in recs))
    return time.perf_counter() - t0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="an S-Mind checkpoint after the perspective stage or RL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["tiny", "viewspatial"])
    ap.add_argument("--concurrency", type=int, default=8, help="questions in flight")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_rounds", type=int, default=256)
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--max_images", type=int, default=32)
    ap.add_argument("--limit", type=int, help="the first N questions of each set (quick checks)")
    ap.add_argument("--warmup", type=int, default=16,
                    help="questions answered untimed before each set: the engine's first requests pay one-off kernel "
                         "compilation, far slower than a warm question")
    args = ap.parse_args()
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    ckpt = SpatialCheckpoint.load(args.ckpt)
    if ckpt.insertion is None:
        raise ValueError(f"{ckpt.path} is a fusion-stage checkpoint")
    model, layout = str(ckpt.path), ckpt.layout
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=model, gpu_memory_utilization=args.gpu_mem, max_model_len=16384, enable_prefix_caching=False,
        limit_mm_per_prompt={"image": args.max_images, "video": 0}, max_num_seqs=32, seed=0))
    tok = engine.get_tokenizer()
    ckpt.token_ids(tok)
    ins = Insertion(layout, tok)
    try:
        for name in args.sets:
            recs = [json.loads(l) for l in open(set_path(name))][:args.limit]
            d = Path(args.out) / name
            d.mkdir(parents=True, exist_ok=True)
            pred = d / "predictions.jsonl"
            if pred.exists():
                raise FileExistsError(f"{pred}: timings are not resumable, remove it first")
            await run_set(engine, tok, recs[-args.warmup:], ins, layout, args, d / "warmup.jsonl")
            (d / "warmup.jsonl").unlink()
            total = await run_set(engine, tok, recs, ins, layout, args, pred)
            rows = [json.loads(l) for l in open(pred)]
            walls = [r["wall_time_sec"] for r in rows]
            summary = dict(model=args.ckpt, concurrency=args.concurrency, n=len(rows),
                           acc=sum(r["correct"] for r in rows) / len(rows),
                           invalid=sum(r["extracted"] is None for r in rows) / len(rows),
                           truncated=sum(r["finish_reason"] == "length" for r in rows) / len(rows),
                           mean_triggers=sum(r["triggers"] for r in rows) / len(rows),
                           mean_wall_sec=statistics.mean(walls), median_wall_sec=statistics.median(walls),
                           total_wall_sec=total, sec_per_question_throughput=total / len(rows))
            (d / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
            print(name, json.dumps(summary), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
