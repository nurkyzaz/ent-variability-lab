from __future__ import annotations
import pandas as pd

def _guess_time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        lc = str(c).lower()
        if "mjd" in lc or lc == "jd" or "time" in lc:
            return c
    return df.columns[0]

def _guess_mag_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("mag", "magnitude", "flux") or "mag" in lc:
            return c
    return df.columns[1]

def _guess_err_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        lc = str(c).lower()
        if "err" in lc or "sigma" in lc or "unc" in lc:
            return c
    return None

def standardize_photometry(
    df: pd.DataFrame,
    survey: str,
    band: str,
    object_id: str,
) -> pd.DataFrame:
    """
    Convert raw per-survey table to a common schema:
      object_id, time, mag, mag_err, survey, band
    """
    tcol = _guess_time_col(df)
    mcol = _guess_mag_col(df)
    ecol = _guess_err_col(df)

    out = pd.DataFrame({
        "object_id": object_id,
        "time": pd.to_numeric(df[tcol], errors="coerce"),
        "mag": pd.to_numeric(df[mcol], errors="coerce"),
        "survey": survey,
        "band": band,
    })

    if ecol is not None:
        out["mag_err"] = pd.to_numeric(df[ecol], errors="coerce")
    else:
        out["mag_err"] = pd.NA

    out = out.dropna(subset=["time", "mag"]).sort_values("time")
    return out.reset_index(drop=True)

def stitch(lightcurves: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(lightcurves, ignore_index=True)
    df = df.dropna(subset=["time", "mag"]).sort_values(["object_id", "time"])
    return df.reset_index(drop=True)
