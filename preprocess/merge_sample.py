#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_INPUT_DIR = "/gemini/space/FG-CLIP/FineHARD/json_files"
DEFAULT_SAMPLE_SIZE = 2_000_000
DEFAULT_SEED = 20260429


def progress_iter(iterable, *, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def iter_json_list_items(input_dir: Path, show_progress: bool = False):
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found under: {input_dir}")

    file_iter = progress_iter(json_files, enabled=show_progress, desc="JSON files", unit="file")
    for json_file in file_iter:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{json_file} must contain a JSON list, got {type(data).__name__}.")
        item_iter = progress_iter(data, enabled=show_progress, desc=json_file.name, unit="item", leave=False)
        for item in item_iter:
            yield item


def reservoir_sample(items, sample_size: int, seed: int, show_progress: bool = False) -> tuple[list[Any], int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    rng = random.Random(seed)
    sample = []
    total_count = 0

    item_iter = progress_iter(items, enabled=show_progress, desc="Sampling", unit="item")
    for item in item_iter:
        total_count += 1
        if len(sample) < sample_size:
            sample.append(item)
            if show_progress and tqdm is not None and total_count % 10_000 == 0:
                item_iter.set_postfix(sample=len(sample), scanned=total_count)
            continue

        sample_idx = rng.randrange(total_count)
        if sample_idx < sample_size:
            sample[sample_idx] = item
        if show_progress and tqdm is not None and total_count % 10_000 == 0:
            item_iter.set_postfix(sample=len(sample), scanned=total_count)

    if total_count < sample_size:
        raise ValueError(f"Cannot sample {sample_size} items; only {total_count} items found.")

    if show_progress and tqdm is not None:
        print("Shuffling sampled items...")
    rng.shuffle(sample)
    return sample, total_count


def write_json_atomic(data: list[Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    with open(tmp_output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_output, output)


def sample_json_list_files(
    input_dir: str | Path,
    output: str | Path,
    sample_size: int,
    seed: int,
    show_progress: bool = False,
) -> int:
    input_dir = Path(input_dir)
    output = Path(output)
    sample, total_count = reservoir_sample(
        iter_json_list_items(input_dir, show_progress=show_progress),
        sample_size=sample_size,
        seed=seed,
        show_progress=show_progress,
    )
    if show_progress and tqdm is not None:
        print(f"Writing {len(sample)} sampled items to {output}...")
    write_json_atomic(sample, output)
    return total_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge JSON-list files from a directory and write a random sampled JSON list."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()
    show_progress = not args.no_progress

    total_count = sample_json_list_files(
        input_dir=args.input_dir,
        output=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        show_progress=show_progress,
    )
    print(
        f"Wrote {args.sample_size} sampled items from {total_count} total items "
        f"under {args.input_dir} to {args.output}"
    )


if __name__ == "__main__":
    main()
