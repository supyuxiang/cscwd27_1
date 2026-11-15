#!/usr/bin/env python3
"""Pool each debate turn with XLM-RoBERTa-large and save (N, 4, hidden) features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModel, AutoTokenizer

from src.modeling_loopfuse_chinese import Debater


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        default="/data3/yxfeng/cscwd27_loopfuse/weibo_debate/trajectories.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/data3/yxfeng/cscwd27_loopfuse/weibo_debate/hidden.pt",
    )
    parser.add_argument("--bert", default="/data3/yxfeng/models/bert/xlm-roberta-large")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    rows = read_rows(Path(args.trajectories))
    if not rows:
        raise SystemExit(f"no trajectories in {args.trajectories}")
    for row in rows:
        missing = [slot for slot in Debater.slots if not str(row.get(slot) or "").strip()]
        if missing:
            raise SystemExit(f"id={row.get('id')} missing {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.bert)
    model = AutoModel.from_pretrained(args.bert)
    model.to(device)
    model.eval()
    hidden_size = int(model.config.hidden_size)

    pooled = []
    texts = [str(row[slot]) for row in rows for slot in Debater.slots]
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start:start + args.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            cls = model(**encoded).last_hidden_state[:, 0]
            pooled.append(cls.cpu())
            print(f"[bert] {min(start + len(batch), len(texts))}/{len(texts)}", flush=True)

    flat = torch.cat(pooled, dim=0)
    hidden = flat.view(len(rows), len(Debater.slots), hidden_size).contiguous()
    payload = {
        "ids": [str(row["id"]) for row in rows],
        "order": list(Debater.slots),
        "hidden": hidden,
        "bert": args.bert,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(f"[done] {out} shape={tuple(hidden.shape)}", flush=True)


if __name__ == "__main__":
    main()
