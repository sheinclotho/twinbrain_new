"""
TwinBrain — 训练可视化工具
==========================

仅包含生产训练流程所需的曲线绘图函数。

意识指标可视化（ConsciousnessVisualizer）已移至
``reference/visualization_consciousness.py``。
"""

from pathlib import Path
from typing import Dict, Optional


def plot_training_curves(
    history: Dict[str, list],
    output_dir,
    best_epoch: Optional[int] = None,
    best_r2_dict: Optional[Dict[str, float]] = None,
) -> None:
    """Save training-loss and R² curves after training completes.

    Generates two PNG files in ``output_dir``:
    - ``training_loss_curve.png``: train_loss + val_loss vs epoch
    - ``training_r2_curve.png``: val R² per modality vs validation index
      (only when R² history is present in ``history``)

    Gracefully skips if matplotlib is unavailable or if ``history`` is empty.

    Args:
        history: Trainer history dict with keys such as 'train_loss',
            'val_loss', and optionally 'val_r2_eeg', 'val_r2_fmri', etc.
        output_dir: Directory (str or Path) where PNGs are saved.
        best_epoch: Epoch index (1-based) of the best model, for annotation.
        best_r2_dict: R² values at the best epoch, for annotation.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive backend; safe on headless servers
        import matplotlib.pyplot as _plt
    except ImportError:
        return  # matplotlib unavailable — skip silently

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loss = history.get('train_loss', [])
    val_loss   = history.get('val_loss',   [])

    # ── Loss curve ─────────────────────────────────────────────────────────
    if train_loss or val_loss:
        fig, ax = _plt.subplots(figsize=(8, 4))
        if train_loss:
            ax.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss',
                    color='steelblue', linewidth=1.5)
        if val_loss:
            ax.plot(range(1, len(val_loss) + 1), val_loss, label='Val Loss',
                    color='tomato', linewidth=1.5, marker='o', markersize=3)
            best_val_idx = int(min(range(len(val_loss)), key=lambda i: val_loss[i]))
            ax.annotate(
                f'best\n{val_loss[best_val_idx]:.4f}',
                xy=(best_val_idx + 1, val_loss[best_val_idx]),
                xytext=(best_val_idx + 1 + max(1, len(val_loss) // 20), val_loss[best_val_idx]),
                fontsize=7,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                color='gray',
            )
        ax.set_xlabel('Epoch / Validation Index')
        ax.set_ylabel('Loss')
        ax.set_title('TwinBrain V5 — Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / 'training_loss_curve.png', dpi=120, bbox_inches='tight')
        _plt.close(fig)

    # ── R² curve ───────────────────────────────────────────────────────────
    # val_pred_r2_* = signal-space prediction R² (★ primary metric)
    # val_r2_*      = reconstruction R²
    pred_r2_keys  = sorted(k for k in history if k.startswith('val_pred_r2_') and history[k])
    recon_r2_keys = sorted(k for k in history if k.startswith('val_r2_') and history[k])
    r2_keys = pred_r2_keys + recon_r2_keys
    if r2_keys:
        fig, ax = _plt.subplots(figsize=(8, 4))
        colors = ['steelblue', 'tomato', 'forestgreen', 'darkorange', 'purple', 'brown', 'pink']
        for idx, key in enumerate(r2_keys):
            vals = history[key]
            is_pred = key.startswith('val_pred_r2_')
            label = key.replace('val_pred_r2_', 'Pred-R² ').replace('val_r2_', 'R² ')
            ax.plot(range(1, len(vals) + 1), vals,
                    label=label, color=colors[idx % len(colors)],
                    linewidth=2.0 if is_pred else 1.5,
                    linestyle='-' if is_pred else '--',
                    marker='o', markersize=3)
        # Reference lines — modality-specific thresholds for a fully-trained model:
        #   R²=0.30: reconstruction target (Kingma & Welling 2014)
        #   R²=0.20: fMRI prediction target (Thomas et al. 2022; Bolt et al. 2022)
        #   R²=0.10: EEG prediction target (Schirrmeister et al. 2017; Kostas et al. 2020)
        ax.axhline(0.30, color='green',       linestyle=':', linewidth=0.8, alpha=0.7,
                   label='R²=0.30 (recon / fMRI pred ideal)')
        ax.axhline(0.20, color='deepskyblue', linestyle=':', linewidth=0.8, alpha=0.6,
                   label='R²=0.20 (fMRI pred target, fully-trained)')
        ax.axhline(0.10, color='gold',        linestyle=':', linewidth=0.8, alpha=0.6,
                   label='R²=0.10 (EEG pred target, fully-trained)')
        ax.axhline(0.00, color='orange',      linestyle=':', linewidth=0.8, alpha=0.6,
                   label='R²=0 (baseline)')
        ax.set_xlabel('Validation Index')
        ax.set_ylabel('R²')
        ax.set_title(
            'TwinBrain V5 — Validation R² Curve\n'
            '(solid=prediction ★, dashed=reconstruction  |  '
            'EEG-pred target=0.10, fMRI-pred target=0.20, recon target=0.30)'
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if best_r2_dict:
            note = '  '.join(f'{k}={v:.3f}' for k, v in sorted(best_r2_dict.items()))
            ax.set_xlabel(f'Validation Index\n[Best model: {note}]', fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / 'training_r2_curve.png', dpi=120, bbox_inches='tight')
        _plt.close(fig)
