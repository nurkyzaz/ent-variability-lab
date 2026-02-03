# Exploratory Unsupervised Analysis of Astronomical Light Curve Variability
*(with a multi-survey case study + anomaly scoring for extreme nuclear transient-like behavior)*

## What this repo does
Modern time-domain surveys generate huge volumes of light curves (brightness vs time). This repo demonstrates an interpretable unsupervised workflow to:
1) load and clean light curves,
2) stitch multi-survey photometry into a single baseline (realistic data engineering),
3) extract simple statistical variability features,
4) cluster objects into variability groups,
5) score anomalies to surface extreme candidates (ENT-like behavior).

## Data
### Option A (included / recommended): multi-survey case study
This repo can use the public photometry from the companion dataset to the paper:
“An extremely luminous flare recorded from a supermassive black hole”.
The `data/superman/phot/` folder contains multi-survey photometry (ZTF, CRTS, ATLAS, WISE).

### Option B (optional): a small population of objects
To demonstrate clustering and anomaly ranking across many objects, you can either:
- generate simulated AGN-like light curves with injected flares, or
- use a small public subset of real survey data (CSV with object_id/time/mag[/err]/band).

## Quickstart
### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
