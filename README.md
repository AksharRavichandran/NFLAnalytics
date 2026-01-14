NFLAnalytics Data Pipeline
==========================

This repo includes a lightweight pipeline to fetch NFL datasets via `nfl_data_py` for modeling and analysis. It writes structured outputs under `data/raw/` (CSV by default; Parquet also supported).

Quickstart
----------

- Create and activate a virtual env with SSL certs and dependencies:
  - `make venv`
  - `source .venv/bin/activate`

- Fetch core modeling datasets for 2019–2024 (CSV):
  - `make fetch`

- Or run with custom options:
  - `python scripts/fetch_nfl_data.py --years 2015-2024 --preset all --out data/raw --format csv --pbp-cache --pbp-cache-dir .nfl_pbp_cache`

SSL certificate note
--------------------
If you encounter SSL errors, export the certifi bundle path before running:

`export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')`

The fetch script also attempts to set this automatically if unset.

Script overview
---------------
- `scripts/fetch_nfl_data.py`: CLI to fetch datasets. Supports presets and per-dataset selection.
- `scripts/setup_venv.sh`: Creates a venv and installs requirements.
- `requirements.txt`: Python dependencies (includes `nfl_data_py`, `pandas`, `pyarrow`, `certifi`).

Modeling
--------
- Use the notebook `notebooks/02_ensemble_model.ipynb` to train and validate the ensemble model entirely in-memory (no saved joblib artifact). It will optionally build `data/processed/games_dataset.csv` from raw schedules if missing.

Datasets
--------
Presets:
- `basic`: `schedules`, `weekly`, `seasonal`, `team_desc`.
- `modeling`: Adds `pbp`, rosters, injuries, depth charts, snap counts, NGS, IDs, and PFR seasonal/weekly.
- `all`: Attempts most available datasets (adds FTN, draft/combine, lines, officials, QBR).

You can also list datasets explicitly via `--datasets` (overrides `--preset`).

Output
------
Files are saved by dataset under `data/raw/<dataset>/` as Parquet (default) or CSV.

Examples
--------
- Select datasets explicitly:
  - `python scripts/fetch_nfl_data.py --years 2021,2022,2023 --datasets pbp weekly injuries depth_charts`
- Restrict NGS types:
  - `python scripts/fetch_nfl_data.py --years 2020-2024 --datasets ngs --ngs-stats passing receiving`
- Use Parquet instead of CSV:
  - `python scripts/fetch_nfl_data.py --years 2018-2024 --preset modeling --format parquet`

Notes
-----
- Play-by-play can be cached locally via `--pbp-cache` and `--pbp-cache-dir`.
- Seasonal data supports `--season-type` = `ALL|REG|POST`.
- Some datasets (like FTN) are only available for recent seasons.
