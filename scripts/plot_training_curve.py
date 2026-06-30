#!/usr/bin/env python3
"""Regenerate Fig. 2 (MAE-ST pre-training loss + LR schedule) for the paper.

Produces a paper-ready version of imgs/maest_training_curve.png:
  * x-axis cropped to the actually-trained range (no blank 134-250 region),
  * dashed line at the submitted checkpoint (epoch 130),
  * static styling (no live-dashboard title / "Current: ep" chrome),
  * "last 50 epochs" loss inset retained.

Input: a CSV with one row per epoch and columns: epoch,loss,lr
       (rename below if your log uses different headers).

Usage:
    python scripts/plot_training_curve.py path/to/training_log.csv
"""
import sys
import csv
import matplotlib.pyplot as plt

# --- config -----------------------------------------------------------------
SUBMITTED_EPOCH = 130      # dashed-line checkpoint (matches main.tex)
XPAD = 3                   # epochs of right-hand padding past the last logged epoch
INSET_LAST_N = 50          # epochs shown in the loss inset
OUT = "imgs/maest_training_curve.png"
# ---------------------------------------------------------------------------


def load(path):
    epochs, loss, lr = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(float(row["epoch"])))
            loss.append(float(row["loss"]))
            lr.append(float(row["lr"]))
    return epochs, loss, lr


def main(path):
    epochs, loss, lr = load(path)
    xmax = max(epochs) + XPAD  # crop to trained range, not the 250 schedule

    fig, (ax_loss, ax_lr) = plt.subplots(
        2, 1, figsize=(9, 6), height_ratios=[2, 1], sharex=True
    )

    # --- top: reconstruction loss ---
    ax_loss.plot(epochs, loss, color="tab:blue", lw=1.5)
    ax_loss.axvline(SUBMITTED_EPOCH, color="tab:red", ls="--", lw=1.2,
                    label=f"submitted: ep {SUBMITTED_EPOCH}")
    ax_loss.set_ylabel("Loss (epoch avg)")
    ax_loss.set_xlim(0, xmax)
    ax_loss.legend(loc="upper right")
    ax_loss.grid(alpha=0.3)

    # inset: last N epochs
    axin = ax_loss.inset_axes([0.45, 0.45, 0.5, 0.5])
    tail = [(e, l) for e, l in zip(epochs, loss) if e >= max(epochs) - INSET_LAST_N]
    axin.plot([e for e, _ in tail], [l for _, l in tail], color="tab:blue", lw=1.5)
    axin.set_title(f"Last {INSET_LAST_N} epochs", fontsize=9)
    axin.grid(alpha=0.3)

    # --- bottom: learning rate ---
    ax_lr.plot(epochs, lr, color="tab:orange", lw=1.5)
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_xlim(0, xmax)
    ax_lr.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}  (epochs {min(epochs)}-{max(epochs)}, xlim 0-{xmax})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
