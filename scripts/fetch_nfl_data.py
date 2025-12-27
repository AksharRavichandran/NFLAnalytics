#!/usr/bin/env python3
"""
Fetch NFL datasets via nfl_data_py for modeling.

Examples:
  # Basic modeling preset for 2019-2024 to data/raw as CSV
  python scripts/fetch_nfl_data.py --years 2019-2024 --preset modeling --out data/raw --format csv

  # All datasets, cache PBP locally
  python scripts/fetch_nfl_data.py --years 2015-2024 --preset all --pbp-cache --pbp-cache-dir .nfl_pbp_cache

  # Select datasets explicitly
  python scripts/fetch_nfl_data.py --years 2021,2022,2023 --datasets pbp weekly injuries depth_charts
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Iterable, List


def ensure_ssl_cert():
    """Ensure SSL_CERT_FILE is set using certifi to avoid SSL issues."""
    try:
        import certifi

        cert_path = certifi.where()
        # Only set if not already set; users can override via env
        os.environ.setdefault("SSL_CERT_FILE", cert_path)
    except Exception:
        # If certifi is not available or errors, continue; requests may still work
        pass


def parse_years(arg: str) -> List[int]:
    years: List[int] = []
    if "," in arg:
        years = [int(x.strip()) for x in arg.split(",") if x.strip()]
    elif "-" in arg:
        start, end = arg.split("-", 1)
        s, e = int(start.strip()), int(end.strip())
        if s > e:
            s, e = e, s
        years = list(range(s, e + 1))
    else:
        years = [int(arg)]
    return years


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_df(df, out_path: str, fmt: str) -> None:
    if df is None or getattr(df, "empty", False):
        return
    ensure_dir(os.path.dirname(out_path))
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    elif fmt == "csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_nfl_data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Fetch NFL datasets via nfl_data_py for modeling and analysis.

            Datasets:
              - pbp, weekly, seasonal, seasonal_rosters, weekly_rosters
              - schedules, injuries, depth_charts, snap_counts
              - ngs (passing/rushing/receiving), ftn
              - draft_picks, draft_values, combine, ids
              - team_desc, win_totals, sc_lines, officials
              - qbr_season, qbr_weekly
              - pfr_season_pass/rec/rush, pfr_weekly_pass/rec/rush

            Presets:
              - basic: schedules, weekly, seasonal, team_desc
              - modeling: pbp, weekly, seasonal, rosters, schedules, injuries, depth_charts, snap_counts, ngs, ids, team_desc, pfr
              - all: attempts to pull most available datasets
            """
        ),
    )

    parser.add_argument("--years", required=True, help="Years list (e.g., 2019-2024 or 2021,2022,2023)")
    parser.add_argument("--out", default="data/raw", help="Output directory root")
    parser.add_argument("--format", choices=["parquet", "csv"], default="csv", help="Output file format")
    parser.add_argument("--downcast", action="store_true", help="Downcast float64 to float32 where supported")
    parser.add_argument("--season-type", default="REG", choices=["ALL", "REG", "POST"], help="Season type for seasonal & PFR aggregations")

    parser.add_argument("--preset", choices=["basic", "modeling", "all"], help="Preset selection of datasets")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Explicit dataset names (overrides preset if provided)",
    )

    # PBP cache options
    parser.add_argument("--pbp-cache", action="store_true", help="Cache PBP locally before loading")
    parser.add_argument("--pbp-cache-dir", default=None, help="Directory for local PBP cache (optional)")

    # Combine/NGS options
    parser.add_argument(
        "--ngs-stats",
        nargs="*",
        default=["passing", "rushing", "receiving"],
        choices=["passing", "rushing", "receiving"],
        help="NGS stat types to fetch",
    )
    parser.add_argument(
        "--combine-positions",
        nargs="*",
        default=["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "CB", "S"],
        help="Combine positions to fetch",
    )

    args = parser.parse_args(argv)

    ensure_ssl_cert()

    years = parse_years(args.years)
    out_root = args.out
    fmt = args.format
    downcast = args.downcast

    # Determine dataset list
    preset = args.preset
    datasets = args.datasets

    if datasets:
        selected = set(datasets)
    else:
        if preset == "basic":
            selected = {
                "schedules",
                "weekly",
                "seasonal",
                "team_desc",
            }
        elif preset == "modeling":
            selected = {
                "pbp",
                "weekly",
                "seasonal",
                "seasonal_rosters",
                "weekly_rosters",
                "schedules",
                "injuries",
                "depth_charts",
                "snap_counts",
                "ngs",
                "ids",
                "team_desc",
                "pfr_season_pass",
                "pfr_season_rec",
                "pfr_season_rush",
                "pfr_weekly_pass",
                "pfr_weekly_rec",
                "pfr_weekly_rush",
            }
        else:  # all
            selected = {
                # core
                "pbp",
                "weekly",
                "seasonal",
                "seasonal_rosters",
                "weekly_rosters",
                "schedules",
                "injuries",
                "depth_charts",
                "snap_counts",
                "ngs",
                "ftn",
                "ids",
                "team_desc",
                # references & lines
                "win_totals",
                "sc_lines",
                "officials",
                # pfr
                "pfr_season_pass",
                "pfr_season_rec",
                "pfr_season_rush",
                "pfr_weekly_pass",
                "pfr_weekly_rec",
                "pfr_weekly_rush",
                # qbr
                "qbr_season",
                "qbr_weekly",
                # draft/combine
                "draft_picks",
                "draft_values",
                "combine",
            }

    # Import nfl_data_py lazily after SSL setup
    try:
        import nfl_data_py as nfl
        import pandas as pd  # noqa: F401
    except Exception as e:
        sys.stderr.write(
            f"Failed to import nfl_data_py or dependencies: {e}\n"
            "Did you run: source .venv/bin/activate && pip install -r requirements.txt ?\n"
        )
        return 2

    # Optional: cache PBP
    if "pbp" in selected and args.pbp_cache:
        try:
            nfl.cache_pbp(years=years, downcast=downcast, alt_path=args.pbp_cache_dir)
        except Exception as e:
            sys.stderr.write(f"Warning: PBP cache failed: {e}\n")

    def outp(name: str, filename: str) -> str:
        return os.path.join(out_root, name, filename)

    # Fetch helpers
    def _safe_fetch(name: str, func, *fargs, filename: str, **fkwargs):
        try:
            df = func(*fargs, **fkwargs)
            if df is None or getattr(df, "empty", False):
                sys.stderr.write(f"{name}: empty\n")
                return
            # Simple deterministic filename including year span
            if "{year}" in filename:
                # Split per-year saves
                for y in years:
                    try:
                        dfi = func(*fargs, **{**fkwargs, **{"years": [y]}})
                        if dfi is None or getattr(dfi, "empty", False):
                            continue
                        save_df(dfi, outp(name, filename.format(year=y)), fmt)
                    except Exception as ie:
                        sys.stderr.write(f"{name} {y}: {ie}\n")
                return
            else:
                save_df(df, outp(name, filename), fmt)
        except Exception as e:
            sys.stderr.write(f"{name}: {e}\n")

    ensure_dir(out_root)

    # Core datasets
    if "pbp" in selected:
        cols = None  # use all available unless user trims
        _safe_fetch(
            "pbp",
            nfl.import_pbp_data,
            years,
            cols,
            filename=f"pbp_{years[0]}_{years[-1]}.{fmt}",
            downcast=downcast,
            cache=bool(args.pbp_cache_dir),
            alt_path=args.pbp_cache_dir,
        )

    if "weekly" in selected:
        _safe_fetch(
            "weekly",
            nfl.import_weekly_data,
            years,
            None,
            filename=f"weekly_{years[0]}_{years[-1]}.{fmt}",
            downcast=downcast,
        )

    if "seasonal" in selected:
        _safe_fetch(
            "seasonal",
            nfl.import_seasonal_data,
            years,
            filename=f"seasonal_{args.season_type}_{years[0]}_{years[-1]}.{fmt}",
            s_type=args.season_type,
        )

    if "seasonal_rosters" in selected:
        _safe_fetch(
            "seasonal_rosters",
            nfl.import_seasonal_rosters,
            years,
            None,
            filename=f"seasonal_rosters_{years[0]}_{years[-1]}.{fmt}",
        )

    if "weekly_rosters" in selected:
        _safe_fetch(
            "weekly_rosters",
            nfl.import_weekly_rosters,
            years,
            None,
            filename=f"weekly_rosters_{years[0]}_{years[-1]}.{fmt}",
        )

    if "schedules" in selected:
        _safe_fetch(
            "schedules",
            nfl.import_schedules,
            years,
            filename=f"schedules_{years[0]}_{years[-1]}.{fmt}",
        )

    if "injuries" in selected:
        _safe_fetch(
            "injuries",
            nfl.import_injuries,
            years,
            filename=f"injuries_{years[0]}_{years[-1]}.{fmt}",
        )

    if "depth_charts" in selected:
        _safe_fetch(
            "depth_charts",
            nfl.import_depth_charts,
            years,
            filename=f"depth_charts_{years[0]}_{years[-1]}.{fmt}",
        )

    if "snap_counts" in selected:
        _safe_fetch(
            "snap_counts",
            nfl.import_snap_counts,
            years,
            filename=f"snap_counts_{years[0]}_{years[-1]}.{fmt}",
        )

    if "ngs" in selected:
        for stype in args.ngs_stats:
            _safe_fetch(
                f"ngs/{stype}",
                nfl.import_ngs_data,
                stype,
                years,
                filename=f"ngs_{stype}_{years[0]}_{years[-1]}.{fmt}",
            )

    if "ftn" in selected:
        _safe_fetch(
            "ftn",
            nfl.import_ftn_data,
            years,
            None,
            filename=f"ftn_{years[0]}_{years[-1]}.{fmt}",
            downcast=downcast,
            thread_requests=False,
        )

    if "ids" in selected:
        _safe_fetch(
            "ids",
            nfl.import_ids,
            None,
            None,
            filename=f"ids_all.{fmt}",
        )

    if "team_desc" in selected:
        _safe_fetch(
            "team_desc",
            nfl.import_team_desc,
            filename=f"team_desc.{fmt}",
        )

    if "win_totals" in selected:
        _safe_fetch(
            "win_totals",
            nfl.import_win_totals,
            years,
            filename=f"win_totals_{years[0]}_{years[-1]}.{fmt}",
        )

    if "sc_lines" in selected:
        _safe_fetch(
            "sc_lines",
            nfl.import_sc_lines,
            years,
            filename=f"sc_lines_{years[0]}_{years[-1]}.{fmt}",
        )

    if "officials" in selected:
        _safe_fetch(
            "officials",
            nfl.import_officials,
            years,
            filename=f"officials_{years[0]}_{years[-1]}.{fmt}",
        )

    if "qbr_season" in selected:
        _safe_fetch(
            "qbr/season",
            nfl.import_qbr,
            years,
            "nfl",
            "season",
            filename=f"qbr_season_{years[0]}_{years[-1]}.{fmt}",
        )

    if "qbr_weekly" in selected:
        _safe_fetch(
            "qbr/weekly",
            nfl.import_qbr,
            years,
            "nfl",
            "weekly",
            filename=f"qbr_weekly_{years[0]}_{years[-1]}.{fmt}",
        )

    # PFR (season & weekly)
    if "pfr_season_pass" in selected:
        _safe_fetch(
            "pfr/season_pass",
            nfl.import_seasonal_pfr,
            "pass",
            years,
            filename=f"pfr_season_pass_{args.season_type}_{years[0]}_{years[-1]}.{fmt}",
            s_type="pass",
            years=years,
        )
    if "pfr_season_rec" in selected:
        _safe_fetch(
            "pfr/season_rec",
            nfl.import_seasonal_pfr,
            "rec",
            years,
            filename=f"pfr_season_rec_{args.season_type}_{years[0]}_{years[-1]}.{fmt}",
            s_type="rec",
            years=years,
        )
    if "pfr_season_rush" in selected:
        _safe_fetch(
            "pfr/season_rush",
            nfl.import_seasonal_pfr,
            "rush",
            years,
            filename=f"pfr_season_rush_{args.season_type}_{years[0]}_{years[-1]}.{fmt}",
            s_type="rush",
            years=years,
        )

    if "pfr_weekly_pass" in selected:
        _safe_fetch(
            "pfr/weekly_pass",
            nfl.import_weekly_pfr,
            "pass",
            years,
            filename=f"pfr_weekly_pass_{years[0]}_{years[-1]}.{fmt}",
            s_type="pass",
            years=years,
        )
    if "pfr_weekly_rec" in selected:
        _safe_fetch(
            "pfr/weekly_rec",
            nfl.import_weekly_pfr,
            "rec",
            years,
            filename=f"pfr_weekly_rec_{years[0]}_{years[-1]}.{fmt}",
            s_type="rec",
            years=years,
        )
    if "pfr_weekly_rush" in selected:
        _safe_fetch(
            "pfr/weekly_rush",
            nfl.import_weekly_pfr,
            "rush",
            years,
            filename=f"pfr_weekly_rush_{years[0]}_{years[-1]}.{fmt}",
            s_type="rush",
            years=years,
        )

    # Draft / Combine
    if "draft_picks" in selected:
        _safe_fetch(
            "draft/picks",
            nfl.import_draft_picks,
            years,
            filename=f"draft_picks_{years[0]}_{years[-1]}.{fmt}",
        )
    if "draft_values" in selected:
        _safe_fetch("draft/values", nfl.import_draft_values, filename=f"draft_values.{fmt}")

    if "combine" in selected:
        _safe_fetch(
            "combine",
            nfl.import_combine_data,
            years,
            args.combine_positions,
            filename=f"combine_{years[0]}_{years[-1]}.{fmt}",
        )

    print(f"Done. Output written under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
