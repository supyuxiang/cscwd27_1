#!/usr/bin/env python3
"""Sample Weibo debate trajectories with vLLM, one slot across every sample before the next.

Order: iter0_real, iter0_fake, iter1_real, iter1_fake.
Round 1 sees the speaker's own round-0 answer and the opponent's round-0 answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_jsonl_samples
from src.modeling_loopfuse_chinese import Debater

OPPONENT = {"iter1_real": "iter0_fake", "iter1_fake": "iter0_real"}
OWN = {"iter1_real": "iter0_real", "iter1_fake": "iter0_fake"}


def read_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_samples(paths: list[str]) -> list[dict]:
    seen: set[str] = set()
    samples: list[dict] = []
    for path in paths:
        loaded, missing = load_jsonl_samples(path)
        print(f"[data] {path} usable={len(loaded)} missing_images={missing}", flush=True)
        for sample in loaded:
            if sample["id"] in seen:
                continue
            seen.add(sample["id"])
            samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        action="append",
        default=[
            "/home/yxfeng/cscwd27_loopfuse/data/weibo/train.jsonl",
            "/home/yxfeng/cscwd27_loopfuse/data/weibo/test.jsonl",
        ],
    )
    parser.add_argument(
        "--output",
        default="/data3/yxfeng/cscwd27_loopfuse/weibo_debate/trajectories.jsonl",
    )
    parser.add_argument("--model", default=Debater.default_path)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    samples = load_samples(args.jsonl)
    out_path = Path(args.output)
    saved = read_rows(out_path)
    rows = []
    for sample in samples:
        row = saved.get(sample["id"], {})
        row.update({"id": sample["id"], "text": sample["text"], "image": sample["image"], "label": int(sample["label"])})
        rows.append(row)
    by_id = {row["id"]: row for row in rows}
    print(f"[data] samples={len(rows)} already={out_path}", flush=True)

    # Long Weibo posts plus image tokens can exceed 8k, so cap the image at 448x448
    # (about 1k visual tokens) and keep headroom for the two-turn debate.
    mm_kwargs = {
        "max_pixels": args.max_pixels,
        "min_pixels": 4 * 28 * 28,
    } if args.max_pixels > 0 else {}
    llm = LLM(
        args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.85,
        mm_processor_kwargs=mm_kwargs,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    for slot in Debater.slots:
        pending = [row for row in rows if not str(row.get(slot) or "").strip()]
        print(f"[stage] {slot} todo={len(pending)} / {len(rows)}", flush=True)
        for start in range(0, len(pending), args.chunk):
            chunk = pending[start:start + args.chunk]
            conversations = []
            for row in chunk:
                own = str(row.get(OWN.get(slot, ""), "") or "")
                opponent = str(row.get(OPPONENT.get(slot, ""), "") or "")
                if slot in OPPONENT and (not own.strip() or not opponent.strip()):
                    raise RuntimeError(f"{slot} is missing round-0 text for id={row['id']}")
                conversations.append(Debater.stage_messages(
                    news=row["text"],
                    image=row["image"],
                    slot=slot,
                    own=own,
                    opponent=opponent,
                ))
            try:
                outputs = llm.chat(conversations, params, use_tqdm=False)
            except Exception as exc:
                if "longer than the maximum model length" not in str(exc):
                    raise
                # Fall back to one prompt at a time so a single long post does not
                # lose the rest of the chunk.
                outputs = []
                for conversation in conversations:
                    outputs.extend(llm.chat([conversation], params, use_tqdm=False))
            if len(outputs) != len(chunk):
                raise RuntimeError(f"{slot} returned {len(outputs)} outputs for {len(chunk)} prompts")
            for row, output in zip(chunk, outputs):
                text = output.outputs[0].text.strip()
                if not text:
                    raise RuntimeError(f"empty {slot} generation for id={row['id']}")
                by_id[row["id"]][slot] = text
            write_rows(out_path, rows)
            print(f"[stage] {slot} saved {start + len(chunk)}/{len(pending)}", flush=True)
        print(f"[stage] {slot} done", flush=True)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
