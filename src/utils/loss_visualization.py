import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ScalarSeries:
    tag: str
    label: str
    steps: np.ndarray
    values: np.ndarray


def _event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise ImportError(
            "Reading TensorBoard event files requires tensorboard. "
            "Install it or run this script in an environment that provides it."
        ) from exc
    return event_accumulator


def available_scalar_tags(event_path: str) -> List[str]:
    event_accumulator = _event_accumulator()
    accumulator = event_accumulator.EventAccumulator(
        event_path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    return list(accumulator.Tags().get("scalars", []))


def load_scalars(
    event_path: str,
    tags: Sequence[str],
    labels: Optional[Dict[str, str]] = None,
) -> Dict[str, ScalarSeries]:
    event_accumulator = _event_accumulator()
    accumulator = event_accumulator.EventAccumulator(
        event_path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

    available_tags = set(accumulator.Tags().get("scalars", []))
    missing_tags = [tag for tag in tags if tag not in available_tags]
    if missing_tags:
        raise ValueError(
            "Missing scalar tag(s): {}. Available scalar tags: {}".format(
                ", ".join(missing_tags),
                ", ".join(sorted(available_tags)),
            )
        )

    labels = labels or {}
    series = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        series[tag] = ScalarSeries(
            tag=tag,
            label=labels.get(tag, tag),
            steps=np.asarray([event.step for event in events], dtype=np.float64),
            values=np.asarray([event.value for event in events], dtype=np.float64),
        )
    return series


def smooth_ema(values: np.ndarray, weight: float) -> np.ndarray:
    if not 0.0 <= weight < 1.0:
        raise ValueError("smooth weight must be in [0, 1), got {}".format(weight))
    if values.size == 0 or weight == 0.0:
        return values.copy()

    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for idx in range(1, values.size):
        smoothed[idx] = weight * smoothed[idx - 1] + (1.0 - weight) * values[idx]
    return smoothed


def infer_steps_per_epoch(train_steps: np.ndarray, val_steps: np.ndarray) -> Optional[float]:
    if train_steps.size == 0 or val_steps.size == 0:
        return None
    max_val_step = float(np.nanmax(val_steps))
    if max_val_step <= 0.0:
        return None
    return float(np.nanmax(train_steps)) / max_val_step


def _downsample(x_values: np.ndarray, y_values: np.ndarray, max_points: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0 or x_values.size <= max_points:
        return x_values, y_values

    indices = np.linspace(0, x_values.size - 1, num=max_points, dtype=np.int64)
    indices = np.unique(indices)
    return x_values[indices], y_values[indices]


def _series_x_values(
    series: ScalarSeries,
    x_axis: str,
    train_tag: str,
    steps_per_epoch: Optional[float],
) -> np.ndarray:
    if x_axis == "step":
        return series.steps
    if x_axis != "epoch":
        raise ValueError("x_axis must be 'epoch' or 'step', got {}".format(x_axis))
    if series.tag == train_tag:
        if steps_per_epoch is None or steps_per_epoch <= 0.0:
            raise ValueError("steps_per_epoch is required when plotting train scalars by epoch")
        return series.steps / steps_per_epoch
    return series.steps


def plot_loss_curves(
    event_path: str,
    output_path: str,
    train_tag: str = "loss",
    val_tag: str = "val_loss",
    smooth: float = 0.9,
    x_axis: str = "epoch",
    train_steps_per_epoch: Optional[float] = None,
    show_raw: bool = True,
    max_train_points: Optional[int] = 12000,
    dpi: int = 180,
    title: Optional[str] = None,
) -> str:
    labels = {
        train_tag: "train_loss",
        val_tag: "val_loss",
    }
    series_by_tag = load_scalars(event_path, [train_tag, val_tag], labels=labels)
    train_series = series_by_tag[train_tag]
    val_series = series_by_tag[val_tag]

    steps_per_epoch = train_steps_per_epoch
    if x_axis == "epoch" and steps_per_epoch is None:
        steps_per_epoch = infer_steps_per_epoch(train_series.steps, val_series.steps)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = {
        train_tag: "#2563eb",
        val_tag: "#dc2626",
    }

    for series in (train_series, val_series):
        x_values = _series_x_values(series, x_axis, train_tag, steps_per_epoch)
        y_raw = series.values
        y_smooth = smooth_ema(y_raw, smooth)

        if show_raw:
            raw_x, raw_y = _downsample(
                x_values,
                y_raw,
                max_train_points if series.tag == train_tag else None,
            )
            ax.plot(
                raw_x,
                raw_y,
                color=colors.get(series.tag),
                alpha=0.16,
                linewidth=0.7,
                label="{} raw".format(series.label),
            )

        smooth_x, smooth_y = _downsample(
            x_values,
            y_smooth,
            max_train_points if series.tag == train_tag else None,
        )
        ax.plot(
            smooth_x,
            smooth_y,
            color=colors.get(series.tag),
            linewidth=2.0,
            label="{} smoothed".format(series.label),
        )

    ax.set_xlabel("Epoch" if x_axis == "epoch" else "Step")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Train and Validation Loss")
    ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
