#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    smoothed = []
    running_sum = 0.0
    queue = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        smoothed.append(running_sum / len(queue))
    return smoothed


def read_logs(trainer_state_path: Path) -> tuple[list[dict], dict | None]:
    with trainer_state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    logs = []
    train_summary = None
    for record in state.get("log_history", []):
        if "loss" in record and "step" in record:
            logs.append(record)
        if "train_loss" in record:
            train_summary = record

    return logs, train_summary


def write_csv(logs: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "epoch", "loss", "grad_norm", "learning_rate"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in logs:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_svg(logs: list[dict], output_path: Path, smooth_window: int, title: str | None) -> None:
    steps = [int(record["step"]) for record in logs]
    losses = [float(record["loss"]) for record in logs]
    smoothed_losses = moving_average(losses, smooth_window)

    width, height = 1100, 620
    left, right, top, bottom = 90, 40, 60, 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    min_step, max_step = min(steps), max(steps)
    min_loss, max_loss = min(losses), max(losses)
    if min_step == max_step:
        max_step += 1
    if min_loss == max_loss:
        min_loss -= 0.5
        max_loss += 0.5

    loss_padding = (max_loss - min_loss) * 0.08
    min_loss -= loss_padding
    max_loss += loss_padding

    def xy(step: int, loss: float) -> tuple[float, float]:
        x = left + (step - min_step) / (max_step - min_step) * plot_width
        y = top + (max_loss - loss) / (max_loss - min_loss) * plot_height
        return x, y

    raw_points = [xy(step, loss) for step, loss in zip(steps, losses)]
    smooth_points = [xy(step, loss) for step, loss in zip(steps, smoothed_losses)]

    x_ticks = []
    for i in range(6):
        value = round(min_step + (max_step - min_step) * i / 5)
        x, _ = xy(value, min_loss)
        x_ticks.append((x, value))

    y_ticks = []
    for i in range(6):
        value = min_loss + (max_loss - min_loss) * i / 5
        _, y = xy(min_step, value)
        y_ticks.append((y, value))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#1f2933">{title or "Training Loss"}</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#2f3a45" stroke-width="1"/>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#2f3a45" stroke-width="1"/>
"""
        )
        for x, value in x_ticks:
            f.write(f'<line x1="{x:.2f}" y1="{height - bottom}" x2="{x:.2f}" y2="{height - bottom + 6}" stroke="#2f3a45"/>\n')
            f.write(f'<text x="{x:.2f}" y="{height - bottom + 26}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#46515c">{value}</text>\n')
        for y, value in y_ticks:
            f.write(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#2f3a45"/>\n')
            f.write(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#d7dde3" stroke-width="0.7"/>\n')
            f.write(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#46515c">{value:.4g}</text>\n')

        f.write(f'<polyline points="{polyline(raw_points)}" fill="none" stroke="#8aa0b8" stroke-width="1.2" opacity="0.45"/>\n')
        if smooth_window > 1:
            f.write(f'<polyline points="{polyline(smooth_points)}" fill="none" stroke="#d14f3f" stroke-width="2.2"/>\n')
            f.write(f'<text x="{width - right - 12}" y="{top + 24}" text-anchor="end" font-family="Arial, sans-serif" font-size="14" fill="#d14f3f">loss MA({smooth_window})</text>\n')
        f.write(f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#1f2933">Step</text>\n')
        f.write(f'<text x="24" y="{height / 2}" transform="rotate(-90 24 {height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#1f2933">Loss</text>\n')
        f.write("</svg>\n")


def plot_loss(logs: list[dict], output_path: Path, smooth_window: int, title: str | None) -> Path:
    if output_path.suffix.lower() == ".svg":
        write_svg(logs, output_path, smooth_window, title)
        return output_path

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        fallback_path = output_path.with_suffix(".svg")
        write_svg(logs, fallback_path, smooth_window, title)
        return fallback_path

    steps = [int(record["step"]) for record in logs]
    losses = [float(record["loss"]) for record in logs]
    smoothed_losses = moving_average(losses, smooth_window)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.plot(steps, losses, color="#8aa0b8", linewidth=1.0, alpha=0.45, label="loss")
    if smooth_window > 1:
        ax.plot(steps, smoothed_losses, color="#d14f3f", linewidth=1.8, label=f"loss MA({smooth_window})")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Training Loss")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss trend from HuggingFace trainer_state.json.")
    parser.add_argument("trainer_state", type=Path, help="Path to trainer_state.json.")
    parser.add_argument("-o", "--output", type=Path, help="Output plot path. Defaults to loss_trend.svg next to input.")
    parser.add_argument("--csv", type=Path, help="Output CSV path. Defaults to loss_trend.csv next to input.")
    parser.add_argument("--csv-only", action="store_true", help="Only write CSV, do not create a PNG.")
    parser.add_argument("--smooth-window", type=int, default=20, help="Moving-average window for the plotted loss.")
    parser.add_argument("--title", help="Plot title.")
    args = parser.parse_args()

    trainer_state = args.trainer_state.expanduser().resolve()
    output = args.output or trainer_state.with_name("loss_trend.svg")
    csv_output = args.csv or trainer_state.with_name("loss_trend.csv")

    logs, train_summary = read_logs(trainer_state)
    if not logs:
        raise SystemExit(f"No per-step loss records found in {trainer_state}")

    write_csv(logs, csv_output)
    plot_output = None
    if not args.csv_only:
        plot_output = plot_loss(logs, output, args.smooth_window, args.title)

    print(f"Read {len(logs)} loss records from {trainer_state}")
    print(f"Wrote CSV: {csv_output}")
    if plot_output is not None:
        print(f"Wrote plot: {plot_output}")
    if train_summary is not None:
        print(f"Final train_loss: {train_summary.get('train_loss')}")


if __name__ == "__main__":
    main()
