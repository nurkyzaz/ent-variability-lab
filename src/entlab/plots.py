from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_stitched_lightcurve(df: pd.DataFrame, outpath: Path, title: str = "") -> None:
    plt.figure()
    for (survey, band), sub in df.groupby(["survey", "band"]):
        plt.scatter(sub["time"], sub["mag"], s=10, label=f"{survey} {band}")
    plt.gca().invert_yaxis()
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.title(title or "Stitched multi-survey light curve")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_clusters_2d(pca_df: pd.DataFrame, labels: pd.Series, outpath: Path, title: str) -> None:
    plt.figure()
    plt.scatter(pca_df["pc1"], pca_df["pc2"], s=12, c=labels.to_numpy())
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_top_anomalies(
    df_all: pd.DataFrame,
    scores: pd.Series,
    outpath: Path,
    top_k: int = 12
) -> None:
    """
    Makes a simple panel-like plot by overlaying top-k anomalies (not subplots to keep minimal deps).
    If you want a grid of subplots, say so and I’ll implement it.
    """
    top_ids = scores.sort_values(ascending=False).head(top_k).index.tolist()

    plt.figure()
    for oid in top_ids:
        sub = df_all[df_all["object_id"] == oid].sort_values("time")
        plt.plot(sub["time"], sub["mag"], marker="o", linestyle="-", linewidth=1, markersize=3, label=str(oid))
    plt.gca().invert_yaxis()
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.title(f"Top {top_k} anomaly light curves (overlay)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
