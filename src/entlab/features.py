from __future__ import annotations
import numpy as np
import pandas as pd

def compute_features_for_object(df_obj: pd.DataFrame) -> dict[str, float]:
    """
    df_obj: standardized photometry for ONE object across surveys/bands.
    Returns interpretable variability features.
    """
    mag = df_obj["mag"].to_numpy(dtype=float)
    t = df_obj["time"].to_numpy(dtype=float)

    n = len(mag)
    if n < 5:
        return {
            "n_obs": float(n),
            "mean_mag": np.nan,
            "std_mag": np.nan,
            "amp_p95_p05": np.nan,
            "max_delta_mag": np.nan,
            "timespan_days": np.nan,
            "slope_mag_per_day": np.nan,
            "skew_mag": np.nan,
        }

    p05, p95 = np.percentile(mag, [5, 95])
    amp = p95 - p05
    max_jump = float(np.max(np.abs(np.diff(mag)))) if n >= 2 else np.nan
    timespan = float(np.max(t) - np.min(t))

    # slope with a stable centering
    slope = np.polyfit(t - t.mean(), mag - mag.mean(), 1)[0]

    # skewness (manual to avoid extra deps)
    mu = mag.mean()
    sigma = mag.std()
    skew = float(np.mean(((mag - mu) / sigma) ** 3)) if sigma > 0 else np.nan

    return {
        "n_obs": float(n),
        "mean_mag": float(np.mean(mag)),
        "std_mag": float(np.std(mag)),
        "amp_p95_p05": float(amp),
        "max_delta_mag": float(max_jump),
        "timespan_days": float(timespan),
        "slope_mag_per_day": float(slope),
        "skew_mag": float(skew),
    }

def build_feature_table(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    df_all: standardized photometry for many objects
    Returns: feature table (rows=object_id)
    """
    rows = []
    for oid, df_obj in df_all.groupby("object_id"):
        feats = compute_features_for_object(df_obj)
        feats["object_id"] = oid
        rows.append(feats)
    return pd.DataFrame(rows).set_index("object_id").sort_index()
