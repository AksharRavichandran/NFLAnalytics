import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
EDA_DIR = DATA_DIR / "eda"
PROCESSED_DIR = DATA_DIR / "processed"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_schedules() -> pd.DataFrame:
    sched_path = RAW_DIR / "schedules" / "schedules_2019_2024.csv"
    if not sched_path.exists():
        raise FileNotFoundError(f"Missing schedules file at {sched_path}")
    df = _read_csv(sched_path)
    # Normalize dtypes
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    # Keep regular season + playoffs for completeness; we will subset later
    return df


def load_league_trend() -> pd.DataFrame:
    lt_path = EDA_DIR / "league_trend.csv"
    if not lt_path.exists():
        # Optional; return empty if not present
        return pd.DataFrame(columns=["season", "week", "pass_rate", "epa_mean"])  # type: ignore
    df = _read_csv(lt_path)
    # Standardize column names
    df = df.rename(columns={"pass_rate": "league_pass_rate", "epa_mean": "league_epa_mean"})
    return df[["season", "week", "league_pass_rate", "league_epa_mean"]]


def _long_games(sched: pd.DataFrame) -> pd.DataFrame:
    """
    Convert schedule to team-game long form with one row per team per game.
    """
    base_cols = [
        "game_id", "season", "game_type", "week", "gameday", "weekday",
        "spread_line", "total_line", "div_game", "roof", "surface", "temp", "wind",
    ]

    # Home rows
    home = sched.assign(
        team=sched["home_team"],
        opp=sched["away_team"],
        points_for=sched["home_score"],
        points_against=sched["away_score"],
        is_home=1,
        rest=sched.get("home_rest"),
        moneyline=sched.get("home_moneyline"),
    )[base_cols + ["team", "opp", "points_for", "points_against", "is_home", "rest", "moneyline"]]

    # Away rows
    away = sched.assign(
        team=sched["away_team"],
        opp=sched["home_team"],
        points_for=sched["away_score"],
        points_against=sched["home_score"],
        is_home=0,
        rest=sched.get("away_rest"),
        moneyline=sched.get("away_moneyline"),
    )[base_cols + ["team", "opp", "points_for", "points_against", "is_home", "rest", "moneyline"]]

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["margin"] = long_df["points_for"] - long_df["points_against"]
    long_df["win"] = (long_df["margin"] > 0).astype(int)

    # Sort for rolling features
    long_df = long_df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    return long_df


def _rolling_features(long_df: pd.DataFrame) -> pd.DataFrame:
    feats = long_df.copy()
    grp = feats.groupby(["team", "season"], sort=False)

    def add_roll(col: str, name: str, window: int, func: str = "mean"):
        # Aligned series per original row order; safe to assign by position
        s = grp[col].apply(lambda x: getattr(x.shift(1).rolling(window, min_periods=1), func)())
        feats[f"{name}_{window}"] = s.to_numpy()

    # Core rolling stats up to previous game
    for w in (3, 5):
        add_roll("margin", "roll_margin_mean", w, "mean")
        add_roll("win", "roll_win_rate", w, "mean")
        add_roll("points_for", "roll_pts_for_mean", w, "mean")
        add_roll("points_against", "roll_pts_against_mean", w, "mean")

    # Season-to-date totals/means up to previous game
    feats["st_d_win_rate"] = grp["win"].apply(lambda x: x.shift(1).expanding(min_periods=1).mean()).to_numpy()
    feats["st_d_margin_mean"] = grp["margin"].apply(lambda x: x.shift(1).expanding(min_periods=1).mean()).to_numpy()

    # Carry forward rest and moneyline as numeric
    for c in ["rest", "moneyline", "temp", "wind"]:
        if c in feats.columns:
            feats[c] = pd.to_numeric(feats[c], errors="coerce")

    return feats


def _wide_game_level(sched: pd.DataFrame, team_feats: pd.DataFrame) -> pd.DataFrame:
    # Split team_feats into home/away aligned with schedule rows
    key_cols = ["game_id", "team"]
    feat_cols = [
        "roll_margin_mean_3", "roll_margin_mean_5",
        "roll_win_rate_3", "roll_win_rate_5",
        "roll_pts_for_mean_3", "roll_pts_for_mean_5",
        "roll_pts_against_mean_3", "roll_pts_against_mean_5",
        "st_d_win_rate", "st_d_margin_mean",
        "rest", "moneyline",
    ]

    home_merge = team_feats[key_cols + feat_cols + ["temp", "wind"]].rename(columns={c: f"home_{c}" for c in feat_cols + ["temp", "wind"]})
    home_merge = home_merge.rename(columns={"team": "home_team"})

    away_merge = team_feats[key_cols + feat_cols + ["temp", "wind"]].rename(columns={c: f"away_{c}" for c in feat_cols + ["temp", "wind"]})
    away_merge = away_merge.rename(columns={"team": "away_team"})

    games = sched.merge(home_merge, on=["game_id", "home_team"], how="left")
    games = games.merge(away_merge, on=["game_id", "away_team"], how="left")

    # Target: home team win
    games["home_win"] = (pd.to_numeric(games["home_score"], errors="coerce") > pd.to_numeric(games["away_score"], errors="coerce")).astype("Int64")
    return games


def build_dataset(seasons: Tuple[int, int] = (2019, 2024)) -> pd.DataFrame:
    sched = load_schedules()
    # Ensure numeric types where useful
    for c in ["week", "season", "temp", "wind", "spread_line", "total_line", "home_rest", "away_rest", "home_moneyline", "away_moneyline"]:
        if c in sched.columns:
            sched[c] = pd.to_numeric(sched[c], errors="coerce")

    # Filter specified seasons
    s0, s1 = seasons
    sched = sched[(sched["season"] >= s0) & (sched["season"] <= s1) & (sched["game_type"].isin(["REG"]))].copy()

    long_df = _long_games(sched)
    team_feats = _rolling_features(long_df)
    game_lvl = _wide_game_level(sched, team_feats)

    # Merge EDA league-level trend features
    lt = load_league_trend()
    if not lt.empty:
        game_lvl = game_lvl.merge(lt, on=["season", "week"], how="left")

    # Keep a focused set of columns
    keep_cols = [
        "game_id", "season", "week", "gameday", "weekday",
        "home_team", "away_team", "home_score", "away_score", "home_win",
        "spread_line", "total_line", "div_game", "roof", "surface",
        "home_rest", "away_rest", "home_moneyline", "away_moneyline",
        # engineered
        "home_roll_margin_mean_3", "home_roll_margin_mean_5",
        "home_roll_win_rate_3", "home_roll_win_rate_5",
        "home_roll_pts_for_mean_3", "home_roll_pts_for_mean_5",
        "home_roll_pts_against_mean_3", "home_roll_pts_against_mean_5",
        "home_st_d_win_rate", "home_st_d_margin_mean",
        "home_temp", "home_wind",
        "away_roll_margin_mean_3", "away_roll_margin_mean_5",
        "away_roll_win_rate_3", "away_roll_win_rate_5",
        "away_roll_pts_for_mean_3", "away_roll_pts_for_mean_5",
        "away_roll_pts_against_mean_3", "away_roll_pts_against_mean_5",
        "away_st_d_win_rate", "away_st_d_margin_mean",
        "away_temp", "away_wind",
        # league trend
        "league_pass_rate", "league_epa_mean",
    ]

    # Some columns might be missing depending on EDA availability
    keep_cols = [c for c in keep_cols if c in game_lvl.columns]
    out = game_lvl[keep_cols].copy()
    return out


def main():
    df = build_dataset((2019, 2024))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "games_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
