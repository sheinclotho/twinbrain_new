#!/usr/bin/env python3
"""
plot_log.py — TwinBrain V5 训练日志可视化工具
==============================================

从 outputs/ 中的 training.log 解析训练指标，输出一张 2×2 四格曲线图：

  ┌─────────────────────┬──────────────────────┐
  │  损失曲线            │  预测 R² ★ (主指标)   │
  │  train_loss          │  pred_r2_eeg/fmri    │
  │  val_loss            │  + 参考线             │
  ├─────────────────────┼──────────────────────┤
  │  重建 R²             │  技能分数 / 基线       │
  │  r2_eeg/fmri         │  decorr_*, ar1_r2_*  │
  │  + 参考线            │  pred_r2_h1_*        │
  └─────────────────────┴──────────────────────┘

技能分数面板只在日志级别为 DEBUG（config 中 log_level: DEBUG）时才有数据，
否则显示提示文字。

用法:
  python plot_log.py                              # 自动找 outputs/ 最新 run
  python plot_log.py outputs/twinbrain_v5_xxx/   # 指定 run 目录
  python plot_log.py outputs/.../training.log    # 直接指定日志文件
  python plot_log.py --list                       # 列出所有 run 目录
  python plot_log.py -o /tmp/plots ...            # 指定输出目录
  python plot_log.py --filename my_plot.png ...  # 自定义文件名

依赖: matplotlib (pip install matplotlib) — 其余均为标准库。
"""

from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Package-level constants
# ---------------------------------------------------------------------------

#: Default parent directory for all run outputs.
_OUTPUTS_DIR: Path = Path(__file__).parent / "outputs"

# Regex that matches the INFO-level epoch summary line written by main.py:
#   ✓ Epoch 5/100: train_loss=0.1234, val_loss=0.2345, pred_r2_eeg=0.123 ...
_EPOCH_RE: re.Pattern = re.compile(r"Epoch\s+(\d+)/\d+\s*:", re.UNICODE)

# Matches floating-point key=value pairs such as `train_loss=0.1234`.
# The trailing `\b` ensures `time=47.2s` is NOT captured (word boundary
# fails before the letter 's'), so no manual exclusion of 'time' is needed.
_KV_RE: re.Pattern = re.compile(
    r"\b([a-z][a-z0-9_]*)=(-?\d+\.\d+)\b", re.ASCII
)

# Matches DEBUG-level NPI metric lines (h=1 skill scores):
#   📐 超NPI指标(h=1): decorr_h1_eeg=0.12 ar1_r2_h1_eeg=0.45 pred_r2_h1_eeg=0.23
_NPI_RE: re.Pattern = re.compile(r"📐.*?h=1.*?:\s*(.+)", re.UNICODE)

# Matches DEBUG-level multi-step baseline lines:
#   📐 多步基线: decorr_eeg=0.34 ar1_r2_eeg=-0.12
_BASE_RE: re.Pattern = re.compile(r"📐\s*多步基线:\s*(.+)", re.UNICODE)

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_log(log_path: Path) -> Dict[str, List]:
    """Parse a ``training.log`` file and return per-metric value lists.

    Each epoch-summary line (INFO level) contributes one entry to:
    ``train_loss`` — one per every logged epoch.

    Validation-only lines also contribute to:
    ``val_loss``, ``pred_r2_eeg``, ``r2_eeg``, etc.

    Optional DEBUG-level lines immediately following the epoch line
    contribute: ``decorr_*``, ``ar1_r2_*``, ``pred_r2_h1_*``.

    Returns
    -------
    dict
        Keys include every metric name encountered.  Special keys:

        * ``epoch``       – epoch numbers for *every* logged epoch line
          (x-axis for ``train_loss``).
        * ``val_epoch``   – epoch numbers only for *validation* epochs
          (x-axis for ``val_loss``, ``r2_*``, ``pred_r2_*`` etc.).
    """
    # Build two lists of row dicts:
    #   all_rows   – one per every epoch-summary line
    #   val_rows   – subset that includes 'val_loss'
    all_rows: List[Dict] = []
    cur: Optional[Dict] = None

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            epoch_m = _EPOCH_RE.search(line)
            if epoch_m:
                # Flush previous row
                if cur is not None:
                    all_rows.append(cur)
                cur = {"epoch": int(epoch_m.group(1))}
                for k, v in _KV_RE.findall(line):
                    cur[k] = float(v)
                continue

            # Pick up DEBUG-level metric lines that follow the epoch line
            if cur is not None:
                for pat in (_NPI_RE, _BASE_RE):
                    dm = pat.search(line)
                    if dm:
                        for k, v in _KV_RE.findall(dm.group(1)):
                            # Don't overwrite values already set from the INFO line
                            cur.setdefault(k, float(v))
                        break  # only one DEBUG pattern per line

    # Flush last row
    if cur is not None:
        all_rows.append(cur)

    if not all_rows:
        return {}

    # Sort rows by epoch number (safety, in case log was incomplete/restarted).
    # 'epoch' is always set when a row is created (see _EPOCH_RE match above).
    all_rows.sort(key=lambda r: r["epoch"])

    # Separate validation rows from training-only rows
    val_rows = [r for r in all_rows if "val_loss" in r]

    # Convert to {metric: [values]} lists
    out: Dict[str, List] = {}
    for row in all_rows:
        for k, v in row.items():
            out.setdefault(k, []).append(v)

    # Build separate val_epoch key (x-axis for validation-only metrics)
    out["val_epoch"] = [r["epoch"] for r in val_rows]

    return out


# ---------------------------------------------------------------------------
# Run-directory discovery
# ---------------------------------------------------------------------------


def find_log(path_arg: Optional[str] = None) -> Path:
    """Resolve which ``training.log`` to use.

    Resolution order:

    1. If *path_arg* points to an existing ``.log`` / ``.txt`` file → use it.
    2. If *path_arg* points to a directory containing ``training.log`` → use that.
    3. If *path_arg* is ``None`` → find the most recently modified run
       directory in ``outputs/`` that contains ``training.log``.

    Raises
    ------
    FileNotFoundError
        When no suitable log file can be found.
    """
    if path_arg is not None:
        p = Path(path_arg)
        if p.is_file():
            return p
        if p.is_dir():
            candidate = p / "training.log"
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"No training.log found in directory: {p}")
        raise FileNotFoundError(f"Path not found: {p}")

    # Auto-discover the latest run in outputs/
    if not _OUTPUTS_DIR.exists():
        raise FileNotFoundError(
            f"outputs/ directory not found at {_OUTPUTS_DIR}.  "
            "Pass the run directory or log file explicitly."
        )
    candidates = sorted(
        (d for d in _OUTPUTS_DIR.iterdir() if d.is_dir() and (d / "training.log").exists()),
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directories with training.log found in {_OUTPUTS_DIR}."
        )
    return candidates[-1] / "training.log"


def list_runs(outputs_dir: Path = _OUTPUTS_DIR) -> None:
    """Print a summary of all available run directories."""
    if not outputs_dir.exists():
        print(f"No outputs directory found at {outputs_dir}")
        return
    runs = sorted(d for d in outputs_dir.iterdir() if d.is_dir())
    if not runs:
        print("No run directories found.")
        return
    for r in runs:
        log = r / "training.log"
        if log.exists():
            try:
                with open(log, encoding="utf-8", errors="replace") as f:
                    n_ep = n_val = 0
                    for line in f:
                        if _EPOCH_RE.search(line):
                            n_ep += 1
                            if "val_loss" in line:
                                n_val += 1
                status = f"{n_ep} epoch lines  ({n_val} validation)"
            except Exception:
                status = "unreadable"
        else:
            status = "(no training.log)"
        print(f"  {r.name:<45s}  {status}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color palette for up to 6 modalities (EEG / fMRI / etc.)
_COLORS = [
    "steelblue", "tomato", "forestgreen", "darkorange", "purple", "saddlebrown",
]


def _label(key: str) -> str:
    """Human-readable label from a metric key."""
    return (
        key.replace("pred_r2_h1_", "pred-R²(h=1) ")
           .replace("pred_r2_", "pred-R² ")
           .replace("r2_", "R² ")
           .replace("decorr_h1_", "decorr(h=1) ")
           .replace("decorr_", "decorr ")
           .replace("ar1_r2_h1_", "AR(1)-R²(h=1) ")
           .replace("ar1_r2_", "AR(1)-R² ")
           .replace("train_loss", "Train Loss")
           .replace("val_loss", "Val Loss")
    )


def plot_metrics(
    metrics: Dict[str, List],
    output_path: Path,
    title_prefix: str = "TwinBrain V5",
) -> None:
    """Generate a 2×2 multi-panel training-curve figure.

    Panels
    ------
    1. **Loss** — train_loss (every epoch) + val_loss (validation epochs)
    2. **Prediction R² ★** — pred_r2_* with EEG/fMRI target reference lines
    3. **Reconstruction R²** — r2_* with reconstruction target reference lines
    4. **Skill scores / baselines** — decorr_*, ar1_r2_*, pred_r2_h1_*
       (populated only when the log was written at DEBUG level)

    Parameters
    ----------
    metrics:
        Dict returned by :func:`parse_log`.
    output_path:
        Destination PNG path.
    title_prefix:
        Title text prepended to the figure title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed.  Run: pip install matplotlib",
            file=sys.stderr,
        )
        return

    epoch_all = metrics.get("epoch", [])
    epoch_val = metrics.get("val_epoch", [])

    # Classify metric keys by group
    loss_keys = [k for k in ("train_loss", "val_loss") if k in metrics]
    pred_keys = sorted(
        k for k in metrics
        if k.startswith("pred_r2_") and "_h1_" not in k
    )
    recon_keys = sorted(k for k in metrics if k.startswith("r2_"))
    skill_keys = sorted(
        k for k in metrics
        if k.startswith("decorr_") or k.startswith("ar1_r2_") or k.startswith("pred_r2_h1_")
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_flat = axes.flatten()

    def _x(key: str) -> List:
        """Return the appropriate x-axis list for a given metric key."""
        # train_loss is per-all-epochs; everything else is per-val-epoch
        if key == "train_loss":
            return epoch_all if epoch_all else list(range(1, len(metrics.get(key, [])) + 1))
        return epoch_val if epoch_val else list(range(1, len(metrics.get(key, [])) + 1))

    def _draw_lines(ax, keys, ref_lines=None, linestyles=None):
        """Plot a set of metric lines on *ax* with optional reference lines."""
        for i, key in enumerate(keys):
            vals = metrics.get(key, [])
            if not vals:
                continue
            xs = _x(key)
            if len(xs) != len(vals):
                xs = list(range(1, len(vals) + 1))
            ls = (linestyles[i] if linestyles and i < len(linestyles) else "-")
            ax.plot(
                xs, vals,
                label=_label(key),
                color=_COLORS[i % len(_COLORS)],
                linewidth=1.8,
                linestyle=ls,
                marker="o",
                markersize=3,
            )
        if ref_lines:
            for val, color, lbl in ref_lines:
                ax.axhline(
                    val, color=color, linestyle=":", linewidth=0.9, alpha=0.7, label=lbl
                )
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(True, alpha=0.3)

    # ── Panel 0: Loss ──────────────────────────────────────────────────────
    ax = axes_flat[0]
    if loss_keys:
        _draw_lines(
            ax, loss_keys,
            linestyles=["-", "--"],  # train=solid, val=dashed
        )
        ax.set_title("损失曲线 (Loss)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
    else:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center", color="gray")
        ax.set_title("损失曲线 (Loss)")

    # ── Panel 1: Prediction R² ★ ───────────────────────────────────────────
    ax = axes_flat[1]
    if pred_keys:
        _draw_lines(
            ax, pred_keys,
            ref_lines=[
                (0.20, "deepskyblue", "R²=0.20  fMRI pred target"),
                (0.10, "goldenrod",   "R²=0.10  EEG pred target"),
                (0.00, "tomato",      "R²=0  (mean baseline)"),
            ],
        )
        ax.set_title("预测 R² ★ (Prediction R²)")
        ax.set_xlabel("验证 Epoch (Val epoch)")
        ax.set_ylabel("R²")
    else:
        ax.text(0.5, 0.5, "No pred_r2 data", ha="center", va="center", color="gray")
        ax.set_title("预测 R² ★ (Prediction R²)")

    # ── Panel 2: Reconstruction R² ────────────────────────────────────────
    ax = axes_flat[2]
    if recon_keys:
        _draw_lines(
            ax, recon_keys,
            ref_lines=[
                (0.85, "forestgreen", "R²=0.85  fMRI recon target"),
                (0.80, "mediumseagreen", "R²=0.80  EEG recon target"),
                (0.30, "olive",       "R²=0.30  minimum acceptable"),
            ],
        )
        ax.set_title("重建 R² (Reconstruction R²)")
        ax.set_xlabel("验证 Epoch (Val epoch)")
        ax.set_ylabel("R²")
    else:
        ax.text(0.5, 0.5, "No r2 data", ha="center", va="center", color="gray")
        ax.set_title("重建 R² (Reconstruction R²)")

    # ── Panel 3: Skill scores / baselines ─────────────────────────────────
    ax = axes_flat[3]
    if skill_keys:
        _draw_lines(
            ax, skill_keys,
            ref_lines=[
                (0.15, "forestgreen", "decorr=0.15  (clearly beats AR(1))"),
                (0.00, "tomato",      "decorr=0  (= AR(1) baseline)"),
            ],
        )
        ax.set_title("技能分数 / 基线 (Skill scores)")
        ax.set_xlabel("验证 Epoch (Val epoch)")
        ax.set_ylabel("Score / R²")
    else:
        ax.text(
            0.5, 0.5,
            "技能分数 (decorr, ar1_r2, pred_r2_h1)\n"
            "仅在 DEBUG 日志级别下可用。\n\n"
            "在 config.yaml 中设置:\n"
            "  log_level: DEBUG\n"
            "然后重新训练。",
            ha="center", va="center", fontsize=9, color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("技能分数 / 基线 (Skill scores)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(f"{title_prefix} — 训练曲线 (Training Curves)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line interface for :mod:`plot_log`."""
    parser = argparse.ArgumentParser(
        description=(
            "TwinBrain V5 — 从 training.log 生成训练曲线图。\n"
            "Run: python plot_log.py [run_dir_or_log_path]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        metavar="RUN_DIR_OR_LOG",
        help=(
            "Run 目录（如 outputs/twinbrain_v5_xxx/）或 training.log 路径。"
            " 省略时自动使用 outputs/ 中最新的 run 目录。"
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        metavar="DIR",
        help="保存图片的目录（默认：与 training.log 同目录）。",
    )
    parser.add_argument(
        "--filename",
        default="training_curves.png",
        metavar="FILE",
        help="输出文件名（默认：training_curves.png）。",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出 outputs/ 中所有 run 目录后退出。",
    )
    args = parser.parse_args(argv)

    if args.list:
        list_runs()
        return

    try:
        log_path = find_log(args.run)
    except FileNotFoundError as exc:
        print(f"错误 (Error): {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"解析日志 (Parsing): {log_path}")
    metrics = parse_log(log_path)

    if not metrics:
        print(
            "未在日志中找到 epoch 指标。请确认训练已启动并生成了有效日志。",
            file=sys.stderr,
        )
        sys.exit(1)

    # Report what was found
    n_all  = len(metrics.get("epoch", []))
    n_val  = len(metrics.get("val_epoch", []))
    keys   = sorted(k for k in metrics if not k.startswith("_") and k not in ("epoch", "val_epoch"))
    print(f"  {n_all} 个 epoch 行，其中 {n_val} 个验证 epoch。")
    print(f"  指标: {', '.join(keys)}")

    out_dir = Path(args.output_dir) if args.output_dir else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / args.filename

    run_name = log_path.parent.name
    plot_metrics(metrics, output_path, title_prefix=f"TwinBrain V5 — {run_name}")
    print(f"完成 (Done). 图片路径: {output_path}")


if __name__ == "__main__":
    main()
