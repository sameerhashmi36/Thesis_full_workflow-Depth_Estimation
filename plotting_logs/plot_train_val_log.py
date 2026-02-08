"""

Reads training/validation logs from Excel files and plots:
  - train_loss vs epoch
  - val_abs_rel vs epoch (2nd y-axis)

Creates one figure per "run_name" (experiment) and saves PNGs.

Example:
  python plot_train_val_log.py \
    --da3-xlsx /mnt/data/DA3_train_val_logs.xlsx \
    --mono-xlsx /mnt/data/Monodepth2_train_val_logs.xlsx \
    --zoe-xlsx /mnt/data/ZoeDepth_train_val_logs.xlsx \
    --out-dir ./train_val_plots
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def read_xlsx(xlsx_path: Path) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.is_file():
        raise FileNotFoundError(xlsx_path)

    xl = pd.ExcelFile(xlsx_path)
    if not xl.sheet_names:
        raise RuntimeError(f"No sheets in {xlsx_path}")

    return pd.read_excel(xlsx_path, sheet_name=xl.sheet_names[0])


def setup_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)


# ------------------------------------------------------------
# Core plotting logic
# ------------------------------------------------------------
def plot_metric(
    df: pd.DataFrame,
    model_tag: str,
    lidar_type: str,
    metric: str,
    out_png: Path,
):
    """
    metric: 'train_loss' or 'val_abs_rel'
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    # enforce order: 20%, 50%, 80%
    fractions = sorted(df["train_fraction"].unique())

    for frac in fractions:
        df_f = df[df["train_fraction"] == frac].sort_values("epoch")

        ax.plot(
            df_f["epoch"],
            df_f[metric],
            marker="o",
            label=f"{int(frac * 100)}%",
        )

    ylabel = "Train Loss" if metric == "train_loss" else "Val Abs Rel"
    title = f"{model_tag} | {lidar_type.capitalize()} | {ylabel} vs Epoch"

    setup_ax(ax, "Epoch", ylabel, title)
    ax.legend(title="Train fraction")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def process_model(model_tag: str, xlsx_path: Path, out_dir: Path):
    df = read_xlsx(xlsx_path)

    required = {
        "lidar_type",
        "train_fraction",
        "epoch",
        "train_loss",
        "val_abs_rel",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[{model_tag}] Missing columns: {missing}")

    model_out = out_dir / model_tag
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"\n[{model_tag}] Processing {xlsx_path}")
    print(f"[{model_tag}] lidar types:", df["lidar_type"].unique())

    for lidar_type in sorted(df["lidar_type"].unique()):
        df_l = df[df["lidar_type"] == lidar_type]

        # --- Training plot
        plot_metric(
            df_l,
            model_tag,
            lidar_type,
            metric="train_loss",
            out_png=model_out / f"train_loss_{lidar_type}.png",
        )

        # --- Validation plot
        plot_metric(
            df_l,
            model_tag,
            lidar_type,
            metric="val_abs_rel",
            out_png=model_out / f"val_abs_rel_{lidar_type}.png",
        )

    print(f"[{model_tag}] Saved plots to {model_out.resolve()}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--da3-xlsx", type=str, default="DA3_train_val_logs.xlsx", help="DA3_train_val_logs.xlsx path")
    ap.add_argument("--mono-xlsx", type=str, default="Monodepth2_train_val_logs.xlsx", help="Monodepth2_train_val_logs.xlsx path")
    ap.add_argument("--zoe-xlsx", type=str, default="ZoeDepth_train_val_logs.xlsx", help="ZoeDepth_train_val_logs.xlsx path")
    ap.add_argument("--out-dir", type=str, default="./train_val_plots")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    process_model("Monodepth2", Path(args.mono_xlsx), out_dir)
    process_model("ZoeDepth", Path(args.zoe_xlsx), out_dir)
    process_model("DA3", Path(args.da3_xlsx), out_dir)

    print("\n[OK] All plots generated.")


if __name__ == "__main__":
    main()
