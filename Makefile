VENV?=.venv

.PHONY: venv
venv:
	./scripts/setup_venv.sh $(VENV)

.PHONY: fetch
fetch:
	$(VENV)/bin/python scripts/fetch_nfl_data.py --years 2019-2024 --preset modeling --out data/raw --format csv --pbp-cache --pbp-cache-dir .nfl_pbp_cache
