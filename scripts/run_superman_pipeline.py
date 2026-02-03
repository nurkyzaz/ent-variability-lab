from __future__ import annotations
from pathlib import Path

from entlab.config import default_paths
from entlab.io import read_dat
from entlab.preprocess import standardize_photometry, stitch
from entlab.features import build_feature_table
from entlab.plots import plot_stitched_lightcurve

OBJECT_ID = "J224554.84+374326.5"

def main():
    p = default_paths()
    p.outputs.mkdir(exist_ok=True)
    p.figures.mkdir(parents=True, exist_ok=True)
    p.tables.mkdir(parents=True, exist_ok=True)

    # Map filenames to (survey, band)
    files = [
        ("J224554.84+374326.5_crts.dat", "CRTS", "opt"),
        ("J224554.84+374326.5_atlas.dat", "ATLAS", "c/o"),
        ("J224554.84+374326.5_g.dat", "ZTF", "g"),
        ("J224554.84+374326.5_r.dat", "ZTF", "r"),
        ("J224554.84+374326.5_wise_w1.dat", "WISE", "W1"),
        ("J224554.84+374326.5_wise_w2.dat", "WISE", "W2"),
    ]

    curves = []
    for fname, survey, band in files:
        path = p.data_superman_phot / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        raw = read_dat(path)
        curves.append(standardize_photometry(raw, survey=survey, band=band, object_id=OBJECT_ID))

    stitched = stitch(curves)
    out_csv = p.tables / "superman_stitched.csv"
    stitched.to_csv(out_csv, index=False)

    fig_path = p.figures / "superman_stitched.png"
    plot_stitched_lightcurve(stitched, fig_path, title=f"Stitched light curve: {OBJECT_ID}")

    ft = build_feature_table(stitched)
    ft.to_csv(p.tables / "superman_features.csv")

    print("Wrote:", out_csv)
    print("Wrote:", fig_path)
    print("Wrote:", p.tables / "superman_features.csv")

if __name__ == "__main__":
    main()
