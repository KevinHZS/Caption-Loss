#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from array import array


def build_random_offsets(jsonl_path: str, sample_size: int, seed: int) -> tuple[array, int]:
    offsets = array("Q")
    rng = random.Random(seed)
    record_count = 0

    with open(jsonl_path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue

            record_count += 1
            if len(offsets) < sample_size:
                offsets.append(offset)
                continue

            sample_idx = rng.randrange(record_count)
            if sample_idx < sample_size:
                offsets[sample_idx] = offset

    if record_count < sample_size:
        raise ValueError(f"Cannot sample {sample_size} rows from {jsonl_path}; only {record_count} rows found.")

    return offsets, record_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a random JSONL byte-offset index without copying rows.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    expected_size = args.sample_size * array("Q").itemsize
    if os.path.exists(args.output) and os.path.getsize(args.output) == expected_size:
        print(f"Reusing existing index: {args.output}")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    offsets, record_count = build_random_offsets(args.jsonl, args.sample_size, args.seed)

    tmp_output = f"{args.output}.tmp.{os.getpid()}"
    with open(tmp_output, "wb") as f:
        offsets.tofile(f)
    os.replace(tmp_output, args.output)

    meta = {
        "jsonl": args.jsonl,
        "jsonl_size": os.path.getsize(args.jsonl),
        "jsonl_mtime": os.path.getmtime(args.jsonl),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "total_records": record_count,
        "index_file": args.output,
    }
    with open(f"{args.output}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(offsets)} offsets from {record_count} rows to {args.output}")


if __name__ == "__main__":
    main()
