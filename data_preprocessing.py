import numpy as np
import pandas as pd

from pipeline_config import BEHAVIOR_LABEL_COLUMNS
from text_label_rules import clean_text


def clean_label_columns(df):
    df = df.copy()
    mapping = {True: 1, False: 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0}

    for col in BEHAVIOR_LABEL_COLUMNS:
        df[col] = df[col].replace(mapping)
        df[col] = pd.to_numeric(df[col], errors="coerce")

        source_col = col + "_source"
        if source_col not in df.columns:
            df[source_col] = np.where(df[col].notna(), "manual", "")
        else:
            df[source_col] = df[source_col].fillna("").astype(str).str.strip().str.lower()
            df.loc[df[col].notna() & df[source_col].eq(""), source_col] = "manual"

    return df


def set_sessions_from_open_time(df):
    df = df.copy()

    time_shifting = pd.to_datetime("2025-11-02").date()
    before_shift = df["Open Date"] < time_shifting
    after_shift = df["Open Date"] >= time_shifting

    df.loc[after_shift & (df["Open Time"].dt.hour >= 10) & (df["Open Time"].dt.hour < 13), "Session"] = "LOCKZ"
    df.loc[after_shift & (df["Open Time"].dt.hour >= 13) & (df["Open Time"].dt.hour < 16), "Session"] = "London Launch"
    df.loc[after_shift & (df["Open Time"].dt.hour >= 16) & (df["Open Time"].dt.hour < 18), "Session"] = "NYKZ"
    df.loc[after_shift & (df["Open Time"].dt.hour >= 18) & (df["Open Time"].dt.hour < 20), "Session"] = "LCKZ"

    df.loc[before_shift & (df["Open Time"].dt.hour >= 9) & (df["Open Time"].dt.hour < 12), "Session"] = "LOCKZ"
    df.loc[before_shift & (df["Open Time"].dt.hour >= 12) & (df["Open Time"].dt.hour < 15), "Session"] = "London Launch"
    df.loc[before_shift & (df["Open Time"].dt.hour >= 15) & (df["Open Time"].dt.hour < 17), "Session"] = "NYKZ"
    df.loc[before_shift & (df["Open Time"].dt.hour >= 17) & (df["Open Time"].dt.hour < 19), "Session"] = "LCKZ"

    df["Session"] = df["Session"].replace({"Interval - 2": "London Launch"})
    df["Session"] = df["Session"].fillna("Other")
    return df


def prepare_dataset_dataframe(input_df):
    df = input_df.copy()
    df["source_row"] = np.arange(2, len(df) + 2)
    df = clean_label_columns(df)

    for col in ["Open Date", "Closed Date"]:
        dt = pd.to_datetime(df[col], errors="coerce")
        df[col.replace("Date", "Time")] = dt
        df[col] = dt.dt.date

    numeric_cols = ["Open", "Closed", "TP", "SL", "Lots", "Commission", "Profit", "R/R", "%RR"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )

    df = set_sessions_from_open_time(df)

    df["Short Review"] = df["Short Review"].fillna("").astype(str)
    df["Long Review"] = df["Long Review"].fillna("").astype(str)
    df["Combined Review"] = (
        df["Short Review"].str.strip() + " " + df["Long Review"].str.strip()
    ).str.strip()

    return df


def load_dataset(dataset_path):
    df = pd.read_excel(dataset_path)
    return prepare_dataset_dataframe(df)


def add_chronological_features(df):
    df = df.copy().sort_values("Open Time").reset_index(drop=True)
    df["trade_sequence"] = np.arange(1, len(df) + 1)

    df["Consumed Time in Trade"] = (
        df["Closed Time"] - df["Open Time"]
    ).dt.total_seconds() / 60

    df["Time Since Previous Trade Minutes"] = df["Open Time"].diff().dt.total_seconds() / 60
    df.loc[df["Open Date"] != df["Open Date"].shift(1), "Time Since Previous Trade Minutes"] = np.nan

    df["Time Between Trades"] = df.groupby("Symbol")["Open Time"].diff()
    df.loc[
        df["Open Time"].dt.date != df.groupby("Symbol")["Open Time"].shift(1).dt.date,
        "Time Between Trades",
    ] = pd.NaT
    df["Time Between Minutes"] = df["Time Between Trades"].dt.total_seconds() / 60
    df["First Trade of a Day"] = df["Time Since Previous Trade Minutes"].isna()

    df["trade_count_today_before_trade"] = df.groupby("Open Date").cumcount()
    df["session_trade_count_before_trade"] = df.groupby(["Open Date", "Session"]).cumcount()

    df["cum_pnl_today_before_trade"] = 0.0
    for open_date, day_df in df.groupby("Open Date", sort=False):
        closed_times = day_df["Closed Time"]
        profits = day_df["Profit"].fillna(0.0)
        for idx, row in day_df.iterrows():
            already_closed = closed_times < row["Open Time"]
            df.loc[idx, "cum_pnl_today_before_trade"] = float(profits.loc[already_closed].sum())

    df["isWin"] = (df["Profit"] > 0).astype(int)
    df["isLose"] = (df["Profit"] < 0).astype(int)
    df["Result"] = np.select(
        [df["Profit"] > 0, df["Profit"] < 0],
        ["Win", "Lose"],
        default="Draw",
    )

    streak_id = (df["Result"] != df["Result"].shift()).cumsum()
    streak_length = df.groupby(streak_id).cumcount() + 1
    df["winStreak"] = streak_length.where(df["Result"] == "Win", 0)
    df["loseStreak"] = streak_length.where(df["Result"] == "Lose", 0)

    df["prev_win_streak"] = df["winStreak"].shift(1).fillna(0)
    df["prev_loss_streak"] = df["loseStreak"].shift(1).fillna(0)
    df.loc[df["Open Date"] != df["Open Date"].shift(1), ["prev_win_streak", "prev_loss_streak"]] = 0

    df["riskChange"] = df["%RR"] - df["%RR"].shift()
    df.loc[df["Open Date"] != df["Open Date"].shift(1), "riskChange"] = np.nan

    df["rolling_winrate_5"] = df["isWin"].rolling(5, min_periods=1).mean().shift(1)
    df["rolling_avg_interval_5"] = (
        df["Time Since Previous Trade Minutes"].rolling(5, min_periods=1).mean().shift(1)
    )
    df["rolling_avg_risk_5"] = df["%RR"].rolling(5, min_periods=1).mean().shift(1)
    df["risk_vs_rolling_avg"] = df["%RR"] / df["rolling_avg_risk_5"]

    interval_threshold = df["Time Since Previous Trade Minutes"].quantile(0.25)
    df["short_interval_flag"] = (
        df["Time Since Previous Trade Minutes"].fillna(np.inf) <= interval_threshold
    ).astype(int)

    before_trade_q75 = df["trade_count_today_before_trade"].quantile(0.75)
    df["high_frequency_before_flag"] = (
        df["trade_count_today_before_trade"] >= before_trade_q75
    ).astype(int)

    max_prev_loss = max(df["prev_loss_streak"].max(), 1)
    df["prev_loss_streak_scaled"] = df["prev_loss_streak"] / max_prev_loss
    df["negative_day_flag"] = (df["cum_pnl_today_before_trade"] < 0).astype(int)
    df["prev_trade_lose"] = df["isLose"].shift(1).fillna(0)
    df.loc[df["Open Date"] != df["Open Date"].shift(1), "prev_trade_lose"] = 0

    df["behavioral_pressure_score"] = (
        0.30 * df["short_interval_flag"].fillna(0)
        + 0.25 * df["high_frequency_before_flag"].fillna(0)
        + 0.25 * df["prev_loss_streak_scaled"].fillna(0)
        + 0.20 * df["negative_day_flag"].fillna(0)
    )

    return df


def add_label_features(df):
    df = df.copy()
    label_feature_cols = []

    for col in BEHAVIOR_LABEL_COLUMNS:
        feature_col = "label_" + clean_text(col).replace(" ", "_").replace("/", "_")
        df[feature_col] = pd.to_numeric(df[col], errors="coerce")
        label_feature_cols.append(feature_col)

    df["behavior_label_available_count"] = df[label_feature_cols].notna().sum(axis=1)
    df["behavior_label_mean"] = df[label_feature_cols].mean(axis=1, skipna=True)

    return df, label_feature_cols


def add_post_trade_features(df):
    df = df.copy()

    df["planned_move"] = (df["TP"] - df["Open"]).abs()
    df["captured_move"] = (df["Closed"] - df["Open"]).abs()
    df["captured_ratio"] = (df["captured_move"] / df["planned_move"]).replace(
        [np.inf, -np.inf], np.nan
    )

    df["tp_distance_abs"] = (df["Closed"] - df["TP"]).abs()
    df["sl_distance_abs"] = (df["Closed"] - df["SL"]).abs()
    df["close_to_tp"] = df["tp_distance_abs"] <= (df["planned_move"] * 0.05)
    df["close_to_sl"] = df["sl_distance_abs"] <= (df["planned_move"] * 0.05)
    df["manual_close_flag"] = (~df["close_to_tp"] & ~df["close_to_sl"]).astype(int)
    df["distance_to_tp_ratio"] = ((df["TP"] - df["Closed"]).abs() / df["planned_move"]).replace(
        [np.inf, -np.inf], np.nan
    )

    df["poor_rr_flag"] = (df["R/R"] < 1.5).astype(int)

    risk_q75 = df["%RR"].quantile(0.75)
    risk_q25 = df["%RR"].quantile(0.25)
    df["oversized_risk_flag"] = (df["%RR"] > risk_q75).astype(int)
    df["undersized_risk_flag"] = (df["%RR"] < risk_q25).astype(int)
    df["risk_jump_flag"] = (df["riskChange"] > 0.15).astype(int)
    df["rapid_reentry_flag"] = (
        (df["short_interval_flag"] == 1) & (~df["First Trade of a Day"])
    ).astype(int)

    short_time_threshold = df["Consumed Time in Trade"].quantile(0.33)
    df["premature_exit_flag"] = (
        (df["manual_close_flag"] == 1)
        & (df["captured_ratio"] < 0.33)
        & (df["Consumed Time in Trade"] <= short_time_threshold)
        & (df["Profit"] > 0)
    ).astype(int)

    return df
