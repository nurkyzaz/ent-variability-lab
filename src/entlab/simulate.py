from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_random_walk_lightcurve(
    object_id: str,
    n_obs: int = 60,
    t_min: float = 0.0,
    t_max: float = 2000.0,
    base_mag: float = 19.0,
    noise: float = 0.05,
    walk_scale: float = 0.03,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(t_min, t_max, size=n_obs))
    steps = rng.normal(0.0, walk_scale, size=n_obs)
    mag = base_mag + np.cumsum(steps) + rng.normal(0.0, noise, size=n_obs)
    return pd.DataFrame({
        "object_id": object_id,
        "time": t,
        "mag": mag,
        "mag_err": noise,
        "survey": "SIM",
        "band": "opt",
    })

def inject_flare(
    df: pd.DataFrame,
    flare_time: float,
    flare_amp: float = -2.0,
    decay_scale: float = 400.0,
) -> pd.DataFrame:
    """
    Adds an ENT-like flare: sudden brightening (mag decreases), then slow decay.
    """
    out = df.copy()
    t = out["time"].to_numpy(dtype=float)
    # brightening then decay (in magnitudes: more negative -> brighter)
    flare = np.where(t >= flare_time, flare_amp * np.exp(-(t - flare_time) / decay_scale), 0.0)
    out["mag"] = out["mag"] + flare
    return out

def build_demo_population(
    n_objects: int = 600,
    n_flare: int = 15,
    seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_df = []

    flare_ids = set(rng.choice(np.arange(n_objects), size=n_flare, replace=False).tolist())

    for i in range(n_objects):
        oid = f"obj_{i:04d}"
        df = simulate_random_walk_lightcurve(
            object_id=oid,
            n_obs=int(rng.integers(40, 90)),
            t_min=0.0,
            t_max=3000.0,
            base_mag=float(rng.normal(19.0, 0.6)),
            noise=float(rng.uniform(0.03, 0.10)),
            walk_scale=float(rng.uniform(0.01, 0.06)),
            seed=int(rng.integers(0, 10_000_000)),
        )
        if i in flare_ids:
            flare_t = float(rng.uniform(500, 2500))
            flare_amp = float(rng.uniform(-4.0, -1.5))
            decay = float(rng.uniform(200, 900))
            df = inject_flare(df, flare_time=flare_t, flare_amp=flare_amp, decay_scale=decay)
        all_df.append(df)

    return pd.concat(all_df, ignore_index=True)
