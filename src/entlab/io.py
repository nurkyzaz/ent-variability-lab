from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_dat(path: str | Path) -> pd.DataFrame:
    """
    Best-effort reader for whitespace-delimited .dat tables.
    Handles:
    - comment lines beginning with '#'
    - header or no header
    """
    path = Path(path)

    # Try with header inference
    try:
        df = pd.read_csv(path, comment="#", delim_whitespace=True)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # Fallback: no header
    df = pd.read_csv(path, comment="#", delim_whitespace=True, header=None)
    df.columns = [f"col{i}" for i in range(df.shape[1])]
    return df
