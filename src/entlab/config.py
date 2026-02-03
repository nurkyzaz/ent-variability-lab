from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path
    data_superman_phot: Path
    outputs: Path
    figures: Path
    tables: Path

def default_paths() -> Paths:
    root = Path(__file__).resolve().parents[2]  # ent-variability-lab/
    outputs = root / "outputs"
    figures = outputs / "figures"
    tables = outputs / "tables"
    return Paths(
        root=root,
        data_superman_phot=root / "data" / "superman" / "phot",
        outputs=outputs,
        figures=figures,
        tables=tables,
    )
