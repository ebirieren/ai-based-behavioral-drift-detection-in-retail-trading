import io
import importlib.util
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_preprocessing import (
    add_chronological_features,
    add_label_features,
    add_post_trade_features,
    prepare_dataset_dataframe,
)
from model_training import fit_and_score_model, get_feature_sets
from recommendations import add_patterns_and_recommendations
from text_label_rules import add_rule_labels

APP_TITLE = "Behavioral Anomaly Detection for Retail Traders"

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "behavior_model.pkl"
DB_PATH = PROJECT_ROOT / "data" / "scored_trades.db"

TEMPLATE_COLUMNS = [
    "Symbol",
    "Type",
    "Open Date",
    "Open",
    "Closed Date",
    "Closed",
    "TP",
    "SL",
    "Lots",
    "Commission",
    "Profit",
    "R/R",
    "%RR",
    "Session",
    "Consumed Time in Trade",
    "Short Review",
    "Long Review",
    "Have you stick to your plan?",
    "Have you move your Stop/TP",
    "Do you think that your entry is okay?",
    "Do you think that your stop-loss is okay?",
    "Do you think that your TP is okay?",
    "Poor Risk/Reward Trade",
    "Entered Too Soon",
    "Entered Too Late",
    "Exited Too Soon",
    "Exited Too Late",
    "Traded not in Trading Plan",
    "Incorrect Stop Placement",
    "Wrong Position Size",
    "Didn't Take Planned Trade",
]

DERIVED_COLUMNS = [
    "source_row_id",
    "trade_duration_min",
    "risk_distance",
    "reward_distance",
    "realized_distance",
    "rr_planned",
    "rr_realized",
    "review_length",
    "is_win",
    "is_loss",
    "anomaly_score",
    "anomaly_label",
]

TARGET_QUESTION_COLUMNS = [
    "Have you stick to your plan?",
    "Have you move your Stop/TP",
    "Do you think that your entry is okay?",
    "Do you think that your stop-loss is okay?",
    "Do you think that your TP is okay?",
    "Poor Risk/Reward Trade",
    "Entered Too Soon",
    "Entered Too Late",
    "Exited Too Soon",
    "Exited Too Late",
    "Traded not in Trading Plan",
    "Incorrect Stop Placement",
    "Wrong Position Size",
    "Didn't Take Planned Trade",
]

TABLE_SCORE_PROXY_COLUMNS = [
    "anomaly_score",
    "anomaly_label",
    "behavioral_pressure_score",
    "trade_count_today_before_trade",
    "Time Between Minutes",
    "cum_pnl_today_before_trade",
    "loseStreak",
    "riskChange",
    "rapid_reentry_flag",
    "risk_jump_flag",
    "negative_day_flag",
]

DETAIL_SCORE_COLUMNS = [
    "anomaly_score",
    "anomaly_label",
    "post_trade_drift_score",
    "post_trade_alert_level",
    "live_drift_score",
    "live_alert_level",
    "behavioral_pressure_score",
    "dominant_pattern",
    "recommendation",
]

DETAIL_TIMING_COLUMNS = [
    "trade_duration_min",
    "Consumed Time in Trade",
    "Time Between Minutes",
    "trade_count_today_before_trade",
    "trades_today",
    "First Trade of a Day",
    "rapid_reentry_flag",
    "high_frequency_flag",
]

DETAIL_RISK_COLUMNS = [
    "risk_distance",
    "reward_distance",
    "realized_distance",
    "rr_planned",
    "rr_realized",
    "%RR",
    "R/R",
    "riskChange",
    "risk_vs_rolling_avg",
    "poor_rr_flag",
    "oversized_risk_flag",
    "undersized_risk_flag",
    "risk_jump_flag",
]

DETAIL_BEHAVIOR_COLUMNS = [
    "cum_pnl_today_before_trade",
    "winStreak",
    "loseStreak",
    "loss_streak_scaled",
    "negative_day_flag",
    "rolling_winrate_5",
    "rolling_avg_interval_5",
    "rolling_avg_risk_5",
    "manual_close_flag",
    "captured_ratio",
    "planned_move",
    "captured_move",
    "tp_distance_abs",
    "sl_distance_abs",
    "close_to_tp",
    "close_to_sl",
    "distance_to_tp_ratio",
    "premature_exit_flag",
    "review_length",
]

FEATURE_LABELS = {
    "Symbol": "Symbol",
    "Type": "Trade Direction",
    "Open": "Entry Price",
    "Closed": "Close Price",
    "TP": "Take Profit",
    "SL": "Stop Loss",
    "Lots": "Lot Size",
    "Commission": "Commission",
    "Profit": "Profit",
    "Session": "Trading Session",
    "Short Review": "Short Journal Review",
    "Long Review": "Long Journal Review",
    "source_row_id": "Source Row",
    "Open Date": "Open Date",
    "Closed Date": "Close Date",
    "Open Time": "Open Timestamp",
    "Closed Time": "Close Timestamp",
    "R/R": "Trade Result in R",
    "%RR": "Risk Percentage",
    "anomaly_score": "Behavioral Drift Score",
    "anomaly_label": "Alert Level",
    "post_trade_drift_score": "Post-Trade Drift Score",
    "post_trade_alert_level": "Post-Trade Alert Level",
    "live_drift_score": "Live Drift Score",
    "live_alert_level": "Live Alert Level",
    "behavioral_pressure_score": "Behavioral Pressure Score",
    "dominant_pattern": "Dominant Behavior Pattern",
    "recommendation": "Recommendation",
    "discipline_score": "Discipline Score",
    "execution_score": "Execution Score",
    "risk_management_score": "Risk Management Score",
    "trade_duration_min": "Calculated Trade Duration",
    "Consumed Time in Trade": "Recorded Trade Duration",
    "Time Between Minutes": "Minutes Since Previous Same-Symbol Trade",
    "trade_count_today_before_trade": "Trades Taken Before This Trade",
    "trades_today": "Total Trades That Day",
    "First Trade of a Day": "First Trade of the Day",
    "rapid_reentry_flag": "Rapid Re-entry",
    "high_frequency_flag": "High Daily Frequency",
    "risk_distance": "Stop Distance",
    "reward_distance": "Target Distance",
    "realized_distance": "Realized Price Movement",
    "rr_planned": "Planned Risk/Reward",
    "rr_realized": "Realized Risk/Reward",
    "riskChange": "Risk Change From Previous Trade",
    "risk_vs_rolling_avg": "Risk Compared With Recent Average",
    "poor_rr_flag": "Poor Risk/Reward Setup",
    "oversized_risk_flag": "Oversized Risk",
    "undersized_risk_flag": "Undersized Risk",
    "risk_jump_flag": "Sudden Risk Increase",
    "cum_pnl_today_before_trade": "Realized PnL Before This Trade",
    "winStreak": "Current Win Streak",
    "loseStreak": "Current Loss Streak",
    "loss_streak_scaled": "Loss Streak Intensity",
    "negative_day_flag": "Negative Day Before Entry",
    "rolling_winrate_5": "Recent Win Rate",
    "rolling_avg_interval_5": "Recent Average Time Between Trades",
    "rolling_avg_risk_5": "Recent Average Risk",
    "manual_close_flag": "Manual Close Away From TP/SL",
    "planned_move": "Planned Price Move",
    "captured_move": "Captured Price Move",
    "tp_distance_abs": "Distance From Take Profit",
    "sl_distance_abs": "Distance From Stop Loss",
    "close_to_tp": "Closed Near Take Profit",
    "close_to_sl": "Closed Near Stop Loss",
    "captured_ratio": "Captured Move Ratio",
    "distance_to_tp_ratio": "Distance From Target Ratio",
    "premature_exit_flag": "Possible Premature Exit",
    "review_length": "Journal Review Length",
    "Proxy": "Proxy",
    "Value": "Value",
    "How Score Is Determined": "How Score Is Determined",
    "overview_avg_anomaly_score": "Average Drift Score",
    "overview_high_risk_rate": "High Risk Rate",
    "overview_behavioral_pressure": "Behavioral Pressure",
    "overview_avg_trades_per_day": "Average Trades Per Day",
    "overview_rapid_reentry_rate": "Rapid Re-entry Rate",
    "overview_risk_jump_rate": "Risk Jump Rate",
    "overview_negative_day_entries": "Negative-Day Entries",
    "overview_max_loss_streak": "Maximum Loss Streak",
}

PROXY_EXPLANATIONS = {
    "Symbol": "Trading instrument for the trade, such as EURUSD or XAUUSD.",
    "Type": "Trade direction, usually Buy or Sell.",
    "Open Date": "Calendar date when the trade was opened.",
    "Open Time": "Full parsed timestamp when the trade was opened.",
    "Open": "Entry price of the trade.",
    "Closed Date": "Calendar date when the trade was closed.",
    "Closed Time": "Full parsed timestamp when the trade was closed.",
    "Closed": "Exit price of the trade.",
    "TP": "Planned take-profit price.",
    "SL": "Planned stop-loss price.",
    "Lots": "Position size used in the trade.",
    "Commission": "Trading commission paid for the trade.",
    "Profit": "Realized profit or loss for the trade.",
    "Session": "Trading session assigned from the trade open time.",
    "Short Review": "Short journal note written for the trade.",
    "Long Review": "Detailed journal note written for the trade.",
    "Have you stick to your plan?": "Manual journal answer about whether the trader followed the original plan.",
    "Have you move your Stop/TP": "Manual journal answer about whether stop-loss or take-profit was moved.",
    "Do you think that your entry is okay?": "Manual journal answer about entry quality.",
    "Do you think that your stop-loss is okay?": "Manual journal answer about stop-loss quality.",
    "Do you think that your TP is okay?": "Manual journal answer about take-profit quality.",
    "Poor Risk/Reward Trade": "Manual or rule-supported label showing weak risk/reward structure.",
    "Entered Too Soon": "Manual or rule-supported label showing early entry behavior.",
    "Entered Too Late": "Manual or rule-supported label showing late entry behavior.",
    "Exited Too Soon": "Manual or rule-supported label showing early exit behavior.",
    "Exited Too Late": "Manual or rule-supported label showing late exit behavior.",
    "Traded not in Trading Plan": "Manual or rule-supported label showing the trade was outside the trading plan.",
    "Incorrect Stop Placement": "Manual or rule-supported label showing possible stop-loss placement issue.",
    "Wrong Position Size": "Manual or rule-supported label showing position-size issue.",
    "Didn't Take Planned Trade": "Manual or rule-supported label showing a missed planned-trade behavior.",
    "anomaly_score": "Overall behavioral anomaly score. Higher values mean the trade looks more unusual compared with the trader's own history.",
    "anomaly_label": "Simple alert label created from the anomaly score.",
    "post_trade_drift_score": "Completed-trade anomaly score using information available after the trade is closed.",
    "post_trade_alert_level": "Post-trade alert label created from completed-trade scoring.",
    "live_drift_score": "Recommendation-side anomaly score using features available before or at trade entry.",
    "live_alert_level": "Live recommendation alert label created from pre-trade behavioral context.",
    "behavioral_pressure_score": "Combined pressure score based on fast re-entry, high frequency, loss streak, and negative daily PnL.",
    "dominant_pattern": "Most visible behavioral pattern for this trade, such as revenge-like, impulsive-like, fear-like, greed-like, or stable-like.",
    "recommendation": "Action suggestion generated from the live alert level and behavioral pattern.",
    "discipline_score": "Score created from discipline-related journal answers when they are available.",
    "execution_score": "Score created from entry/exit quality signals and premature-exit proxy.",
    "risk_management_score": "Score created from stop, target, risk/reward, stop placement, and position-size related signals.",
    "trade_duration_min": "Calculated number of minutes between open time and close time.",
    "Consumed Time in Trade": "Trade duration value recorded or recalculated for the dataset.",
    "Time Between Minutes": "Minutes between this trade and the previous trade on the same symbol.",
    "trade_count_today_before_trade": "Number of trades already opened earlier on the same day.",
    "trades_today": "Total number of trades on that trading day.",
    "First Trade of a Day": "Shows whether this trade is the first trade opened that day.",
    "rapid_reentry_flag": "Yes when the trade was opened after a very short interval, which may indicate impulsive re-entry.",
    "high_frequency_flag": "Yes when the day had unusually high trade frequency.",
    "risk_distance": "Distance between entry price and stop-loss price.",
    "reward_distance": "Distance between entry price and take-profit price.",
    "realized_distance": "Absolute price movement captured between entry and close.",
    "rr_planned": "Planned reward divided by planned risk.",
    "rr_realized": "Realized movement divided by planned risk.",
    "%RR": "Risk amount used for the trade, as recorded in the dataset.",
    "R/R": "Trade result measured in R multiple.",
    "riskChange": "Difference between current trade risk and previous trade risk.",
    "risk_vs_rolling_avg": "Current risk compared with recent average risk.",
    "poor_rr_flag": "Yes when planned risk/reward is below the selected acceptable level.",
    "oversized_risk_flag": "Yes when trade risk is higher than the trader's usual risk range.",
    "undersized_risk_flag": "Yes when trade risk is lower than the trader's usual risk range.",
    "risk_jump_flag": "Yes when risk suddenly increases compared with the previous trade.",
    "cum_pnl_today_before_trade": "Realized profit/loss from trades closed before this trade opened on the same day.",
    "winStreak": "Current consecutive winning trade count.",
    "loseStreak": "Current consecutive losing trade count.",
    "loss_streak_scaled": "Loss streak normalized between 0 and 1 for model usage.",
    "negative_day_flag": "Yes when realized PnL before this trade was negative.",
    "rolling_winrate_5": "Win rate over recent trades before this trade.",
    "rolling_avg_interval_5": "Average time interval over recent trades before this trade.",
    "rolling_avg_risk_5": "Average risk over recent trades before this trade.",
    "manual_close_flag": "Yes when the trade closed away from both TP and SL, suggesting manual intervention.",
    "captured_ratio": "Captured movement divided by planned movement.",
    "planned_move": "Expected move from entry to target.",
    "captured_move": "Move actually captured from entry to close.",
    "tp_distance_abs": "Distance between close price and take-profit price.",
    "sl_distance_abs": "Distance between close price and stop-loss price.",
    "close_to_tp": "Yes when the trade closed close to take profit.",
    "close_to_sl": "Yes when the trade closed close to stop loss.",
    "distance_to_tp_ratio": "Remaining distance to target as a ratio of planned movement.",
    "premature_exit_flag": "Yes when the trade may have been closed too early based on captured move and duration.",
    "review_length": "Number of characters in the written journal review.",
    "Proxy": "Name of the behavioral proxy being shown.",
    "Value": "Current value of the proxy for the selected trade.",
    "How Score Is Determined": "Short explanation of how this proxy contributes to the behavioral interpretation.",
    "overview_avg_anomaly_score": "Average behavioral drift score across all uploaded trades.",
    "overview_high_risk_rate": "Percentage of uploaded trades classified as High Risk.",
    "overview_behavioral_pressure": "Average pressure created by loss context, fast re-entry, high frequency, and negative daily PnL.",
    "overview_avg_trades_per_day": "Average number of trades opened per trading day.",
    "overview_rapid_reentry_rate": "Percentage of trades opened soon after a previous same-symbol trade.",
    "overview_risk_jump_rate": "Percentage of trades where risk increased suddenly compared with the previous trade.",
    "overview_negative_day_entries": "Percentage of trades opened after the trader was already negative for the day.",
    "overview_max_loss_streak": "Largest consecutive loss streak observed in the uploaded trades.",
}


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scored_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trader_id TEXT,
            trade_id TEXT,
            symbol TEXT,
            open_time TEXT,
            close_time TEXT,
            side TEXT,
            profit REAL,
            session TEXT,
            anomaly_score REAL,
            anomaly_label TEXT,
            model_version TEXT,
            created_at TEXT,
            raw_json TEXT
        )
        """
    )
    return conn


def database_is_healthy() -> tuple[bool, str]:
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True, "Database connected"
    except sqlite3.Error as exc:
        return False, f"Database error: {exc}"


def safe_inverse(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return 1 - pd.to_numeric(series, errors="coerce")


def to_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def to_bool_flag_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    mapping = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "TRUE": 1,
        "FALSE": 0,
        "Yes": 1,
        "No": 0,
        "YES": 1,
        "NO": 0,
        "yes": 1,
        "no": 0,
    }

    return pd.to_numeric(df[column].replace(mapping), errors="coerce")


def load_model():
    path = Path(MODEL_PATH)
    if path.exists():
        return joblib.load(path)
    return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "source_row_id" not in data.columns:
        data["source_row_id"] = [f"row_{index + 1}" for index in range(len(data))]

    for col in TARGET_QUESTION_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA
        data[col] = to_bool_flag_series(data, col)

    data["Open Date"] = pd.to_datetime(data["Open Date"], errors="coerce")
    data["Closed Date"] = pd.to_datetime(data["Closed Date"], errors="coerce")
    data["Open Time"] = pd.to_datetime(data["Open Date"], errors="coerce")
    data["Closed Time"] = pd.to_datetime(data["Closed Date"], errors="coerce")

    numeric_cols = [
        "Open",
        "Closed",
        "SL",
        "TP",
        "Profit",
        "Lots",
        "Commission",
        "R/R",
        "%RR",
        "Consumed Time in Trade",
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_values("Open Time").reset_index(drop=True)
    data["source_row_id"] = [f"row_{index + 1}" for index in range(len(data))]

    data["Consumed Time in Trade"] = (
        data["Closed Time"] - data["Open Time"]
    ).dt.total_seconds() / 60
    data["trade_duration_min"] = data["Consumed Time in Trade"]

    data["Time Between Trades"] = data.groupby("Symbol")["Open Time"].diff()
    prev_open_date = data.groupby("Symbol")["Open Time"].shift(1).dt.date
    data.loc[
        data["Open Time"].dt.date != prev_open_date, "Time Between Trades"
    ] = pd.NaT

    data["First Trade of a Day"] = data["Time Between Trades"].isna()
    data["trade_count_today_before_trade"] = data.groupby("Open Date").cumcount()
    data["Time Between Minutes"] = (
        data["Time Between Trades"].dt.total_seconds() / 60
    )

    data["risk_distance"] = (data["Open"] - data["SL"]).abs()
    data["reward_distance"] = (data["TP"] - data["Open"]).abs()
    data["realized_distance"] = (data["Closed"] - data["Open"]).abs()

    data["rr_planned"] = np.where(
        data["risk_distance"] > 0,
        data["reward_distance"] / data["risk_distance"],
        np.nan,
    )

    data["rr_realized"] = np.where(
        data["risk_distance"] > 0,
        data["realized_distance"] / data["risk_distance"],
        np.nan,
    )

    review_parts = []
    if "Short Review" in data.columns:
        review_parts.append(data["Short Review"].fillna("").astype(str))
    if "Long Review" in data.columns:
        review_parts.append(data["Long Review"].fillna("").astype(str))

    if review_parts:
        review_length = pd.Series(0, index=data.index, dtype=float)
        for part in review_parts:
            review_length = review_length.add(part.str.len(), fill_value=0)
        data["review_length"] = review_length
    else:
        data["review_length"] = 0

    data["is_win"] = (data["Profit"] > 0).astype(int)
    data["is_loss"] = (data["Profit"] < 0).astype(int)
    data["isWin"] = data["is_win"]
    data["isLose"] = data["is_loss"]

    data["Result"] = np.select(
        [data["Profit"] > 0, data["Profit"] == 0, data["Profit"] < 0],
        ["Win", "Draw", "Lose"],
        default="Draw",
    )

    data["streakID"] = (data["Result"] != data["Result"].shift()).cumsum()
    data["streakLength"] = data.groupby("streakID").cumcount() + 1
    data["winStreak"] = data["streakLength"].where(data["Result"] == "Win", 0)
    data["loseStreak"] = data["streakLength"].where(data["Result"] == "Lose", 0)

    data["riskChange"] = data["%RR"] - data["%RR"].shift()
    data["cum_pnl_today_before_trade"] = 0.0
    for _, day_data in data.groupby("Open Date", sort=False):
        closed_times = day_data["Closed Time"]
        profits = day_data["Profit"].fillna(0.0)
        for idx, row in day_data.iterrows():
            data.loc[idx, "cum_pnl_today_before_trade"] = float(
                profits.loc[closed_times < row["Open Time"]].sum()
            )
    data["rolling_winrate_5"] = (
        (data["Result"] == "Win").astype(int).rolling(5, min_periods=1).mean().shift()
    )
    data["rolling_avg_interval_5"] = (
        data["Time Between Minutes"].rolling(5, min_periods=1).mean().shift()
    )
    data["rolling_avg_risk_5"] = (
        data["%RR"].rolling(5, min_periods=1).mean().shift()
    )
    data["risk_vs_rolling_avg"] = data["%RR"] / data["rolling_avg_risk_5"]

    interval_threshold = data["Time Between Minutes"].quantile(0.25)
    data["short_interval_flag"] = (
        data["Time Between Minutes"].fillna(999999) <= interval_threshold
    ).astype(int)

    trades_per_day_map = data.groupby("Open Date").size().to_dict()
    data["trades_today"] = data["Open Date"].map(trades_per_day_map)
    daily_trade_q75 = data["trades_today"].dropna().quantile(0.75)
    data["high_frequency_flag"] = (data["trades_today"] >= daily_trade_q75).astype(int)

    max_lose = max(data["loseStreak"].max(), 1)
    data["loss_streak_scaled"] = data["loseStreak"] / max_lose
    data["negative_day_flag"] = (data["cum_pnl_today_before_trade"] < 0).astype(int)

    data["planned_move"] = (data["TP"] - data["Open"]).abs()
    data["captured_move"] = (data["Closed"] - data["Open"]).abs()
    data["captured_ratio"] = data["captured_move"] / data["planned_move"]
    data["captured_ratio"] = data["captured_ratio"].replace([np.inf, -np.inf], np.nan)

    data["tp_distance_abs"] = (data["Closed"] - data["TP"]).abs()
    data["sl_distance_abs"] = (data["Closed"] - data["SL"]).abs()
    data["close_to_tp"] = data["tp_distance_abs"] <= (data["planned_move"] * 0.05)
    data["close_to_sl"] = data["sl_distance_abs"] <= (data["planned_move"] * 0.05)
    data["manual_close_flag"] = (
        ~data["close_to_tp"] & ~data["close_to_sl"]
    ).astype(int)

    data["distance_to_tp_ratio"] = (
        (data["TP"] - data["Closed"]).abs() / data["planned_move"]
    )
    data["distance_to_tp_ratio"] = data["distance_to_tp_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )

    data["poor_rr_flag"] = (data["R/R"] < 1.5).astype(int)
    risk_q75 = data["%RR"].quantile(0.75)
    risk_q25 = data["%RR"].quantile(0.25)
    data["oversized_risk_flag"] = (data["%RR"] > risk_q75).astype(int)
    data["undersized_risk_flag"] = (data["%RR"] < risk_q25).astype(int)
    data["risk_jump_flag"] = (data["riskChange"] > 0.15).astype(int)
    data["rapid_reentry_flag"] = (
        (data["short_interval_flag"] == 1) & (~data["First Trade of a Day"])
    ).astype(int)

    short_time_threshold = data["Consumed Time in Trade"].quantile(0.33)
    data["premature_exit_flag"] = (
        (data["manual_close_flag"] == 1)
        & (data["captured_ratio"] < 0.33)
        & (data["Consumed Time in Trade"] <= short_time_threshold)
        & (data["Profit"] > 0)
    ).astype(int)

    discipline_parts = pd.DataFrame(
        {
            "stick_to_plan": to_numeric_series(data, "Have you stick to your plan?"),
            "not_in_plan": safe_inverse(data.get("Traded not in Trading Plan")),
            "did_take_planned": safe_inverse(data.get("Didn't Take Planned Trade")),
        }
    )
    data["discipline_score"] = discipline_parts.mean(axis=1, skipna=True)

    execution_parts = pd.DataFrame(
        {
            "entry_ok": to_numeric_series(data, "Do you think that your entry is okay?"),
            "not_entered_too_soon": safe_inverse(data.get("Entered Too Soon")),
            "not_entered_too_late": safe_inverse(data.get("Entered Too Late")),
            "not_exited_too_soon": safe_inverse(data.get("Exited Too Soon")),
            "not_exited_too_late": safe_inverse(data.get("Exited Too Late")),
            "not_premature_exit_proxy": safe_inverse(data["premature_exit_flag"]),
        }
    )
    data["execution_score"] = execution_parts.mean(axis=1, skipna=True)

    risk_parts = pd.DataFrame(
        {
            "stop_ok": to_numeric_series(data, "Do you think that your stop-loss is okay?"),
            "tp_ok": to_numeric_series(data, "Do you think that your TP is okay?"),
            "not_poor_rr": safe_inverse(data.get("Poor Risk/Reward Trade")),
            "not_incorrect_stop": safe_inverse(data.get("Incorrect Stop Placement")),
            "not_wrong_pos": safe_inverse(data.get("Wrong Position Size")),
            "not_oversized_risk_proxy": safe_inverse(data["oversized_risk_flag"]),
        }
    )
    data["risk_management_score"] = risk_parts.mean(axis=1, skipna=True)

    data["behavioral_pressure_score"] = (
        0.30 * data["short_interval_flag"].fillna(0)
        + 0.25 * data["high_frequency_flag"].fillna(0)
        + 0.25 * data["loss_streak_scaled"].fillna(0)
        + 0.20 * data["negative_day_flag"].fillna(0)
    )

    return data


def get_expected_columns() -> list[str]:
    return TEMPLATE_COLUMNS.copy()


def normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column_name).strip().lower())


def make_unique_column_names(columns) -> list[str]:
    seen_counts = {}
    unique_columns = []

    for column in columns:
        base_name = str(column).strip() or "Unnamed"
        seen_counts[base_name] = seen_counts.get(base_name, 0) + 1

        if seen_counts[base_name] == 1:
            unique_columns.append(base_name)
        else:
            unique_columns.append(f"{base_name} ({seen_counts[base_name]})")

    return unique_columns


def ensure_unique_dataframe_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    unique_columns = make_unique_column_names(df.columns)
    renamed_duplicates = [
        f"{old} -> {new}"
        for old, new in zip(df.columns, unique_columns)
        if str(old) != str(new)
    ]

    unique_df = df.copy()
    unique_df.columns = unique_columns
    return unique_df, renamed_duplicates


def align_uploaded_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    df, duplicate_header_fixes = ensure_unique_dataframe_columns(df)
    expected_columns = get_expected_columns()
    normalized_expected = {}
    for column in expected_columns:
        normalized_column = normalize_column_name(column)
        normalized_expected.setdefault(normalized_column, []).append(column)

    rename_map = {}
    for column in df.columns:
        if column in expected_columns:
            continue

        normalized_column = normalize_column_name(column)
        matching_expected_columns = normalized_expected.get(normalized_column, [])
        if len(matching_expected_columns) == 1:
            rename_map[column] = matching_expected_columns[0]

    aligned_df = df.rename(columns=rename_map).copy()
    aligned_df, normalized_duplicate_fixes = ensure_unique_dataframe_columns(aligned_df)

    missing_columns = []
    for column in expected_columns:
        if column not in aligned_df.columns:
            aligned_df[column] = pd.NA
            missing_columns.append(column)

    ordered_columns = expected_columns + [
        column for column in aligned_df.columns if column not in expected_columns
    ]

    deduplicated_order = []
    for column in ordered_columns:
        if column not in deduplicated_order:
            deduplicated_order.append(column)

    rename_map = rename_map.copy()
    if duplicate_header_fixes:
        rename_map["__duplicate_header_fixes__"] = ", ".join(duplicate_header_fixes)
    if normalized_duplicate_fixes:
        rename_map["__normalized_duplicate_fixes__"] = ", ".join(
            normalized_duplicate_fixes
        )

    return aligned_df[deduplicated_order], missing_columns, rename_map


def build_template_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    aligned_df, missing_columns, rename_map = align_uploaded_dataframe(df)
    original_columns = df.columns.tolist()
    reverse_rename_map = {target: source for source, target in rename_map.items()}

    template_columns = []
    for column in aligned_df.columns:
        if column in reverse_rename_map:
            template_columns.append(reverse_rename_map[column])
        elif column in missing_columns:
            template_columns.append(column)
        elif column in original_columns:
            template_columns.append(column)
        else:
            template_columns.append(column)

    template_df = pd.DataFrame([{column: pd.NA for column in template_columns}])

    return template_df


def heuristic_score(feature_df: pd.DataFrame) -> pd.DataFrame:
    data = feature_df.copy()

    score = np.zeros(len(data), dtype=float)

    score += np.where(data["trade_duration_min"].fillna(0) < 2, 0.20, 0)
    score += np.where(data["rr_realized"].fillna(0) < 0.5, 0.20, 0)
    score += np.where(data["Profit"].fillna(0) < 0, 0.15, 0)
    score += np.where(data["review_length"].fillna(0) == 0, 0.10, 0)
    score += np.where(data["Lots"].fillna(0) <= 0, 0.15, 0)
    score += np.where(data["risk_distance"].fillna(0) == 0, 0.20, 0)

    score = np.clip(score, 0, 1)

    data["anomaly_score"] = score
    data["anomaly_label"] = pd.cut(
        data["anomaly_score"],
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=["Normal", "Warning", "High Risk"],
    ).astype(str)

    return data


def score_trades(df: pd.DataFrame) -> pd.DataFrame:
    try:
        scored_df = prepare_dataset_dataframe(df)
        scored_df, rule_summary = add_rule_labels(scored_df)

        scored_df = add_chronological_features(scored_df)
        scored_df, label_feature_cols = add_label_features(scored_df)
        scored_df = add_post_trade_features(scored_df)

        post_numeric, live_numeric, categorical = get_feature_sets(label_feature_cols)

        scored_df, post_trade_model = fit_and_score_model(
            scored_df,
            post_numeric,
            categorical,
            "post_trade",
        )
        scored_df, live_model = fit_and_score_model(
            scored_df,
            live_numeric,
            categorical,
            "live",
        )
        scored_df = add_patterns_and_recommendations(scored_df)

        scored_df["anomaly_score"] = scored_df["post_trade_drift_score"].clip(0, 1)
        scored_df["anomaly_label"] = scored_df["post_trade_alert_level"]
        scored_df["source_row_id"] = [
            f"row_{index + 1}" for index in range(len(scored_df))
        ]

        return scored_df
    except Exception:
        feature_df = engineer_features(df)
        return heuristic_score(feature_df)


# -----------------------------
# File/template helpers
# -----------------------------
def validate_columns(df: pd.DataFrame):
    missing = [col for col in get_expected_columns() if col not in df.columns]
    return missing


def build_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": "EURUSD",
                "Type": "Buy",
                "Open Date": "2026-04-12 09:30:00",
                "Open": 1.1025,
                "Closed Date": "2026-04-12 10:05:00",
                "Closed": 1.1040,
                "TP": 1.1055,
                "SL": 1.1015,
                "Lots": 0.50,
                "Commission": -2.5,
                "Profit": 75.0,
                "R/R": 1.5,
                "%RR": 150.0,
                "Session": "LOCKZ",
                "Consumed Time in Trade": 35,
                "Short Review": "Followed plan, entry was patient, exit slightly early.",
                "Long Review": "Trade respected the setup and risk plan overall.",
                "Have you stick to your plan?": "Yes",
                "Have you move your Stop/TP": "No",
                "Do you think that your entry is okay?": "Yes",
                "Do you think that your stop-loss is okay?": "Yes",
                "Do you think that your TP is okay?": "Yes",
                "Poor Risk/Reward Trade": "No",
                "Entered Too Soon": "No",
                "Entered Too Late": "No",
                "Exited Too Soon": "Yes",
                "Exited Too Late": "No",
                "Traded not in Trading Plan": "No",
                "Incorrect Stop Placement": "No",
                "Wrong Position Size": "No",
                "Didn't Take Planned Trade": "No",
            }
        ]
    )


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()

    excel_engine = None
    if importlib.util.find_spec("openpyxl") is not None:
        excel_engine = "openpyxl"
    elif importlib.util.find_spec("xlsxwriter") is not None:
        excel_engine = "xlsxwriter"

    if excel_engine is None:
        return df.to_csv(index=False).encode("utf-8")

    with pd.ExcelWriter(output, engine=excel_engine) as writer:
        df.to_excel(writer, sheet_name="Trades", index=False)

    output.seek(0)
    return output.getvalue()


def spreadsheet_download_info(file_stem):
    if (
        importlib.util.find_spec("openpyxl") is not None
        or importlib.util.find_spec("xlsxwriter") is not None
    ):
        return (
            f"{file_stem}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    return (f"{file_stem}.csv", "text/csv")


def save_scored_trades(scored_df: pd.DataFrame, model_version: str = "v1"):
    now = datetime.utcnow().isoformat()

    rows = []

    for _, row in scored_df.iterrows():
        rows.append(
            (
                str(row.get("source_row_id", "")),
                str(row.get("source_row_id", "")),
                str(row.get("Symbol", "")),
                str(row.get("Open Date", "")),
                str(row.get("Closed Date", "")),
                str(row.get("Type", "")),
                float(row.get("Profit", 0) or 0),
                str(row.get("Session", "")),
                float(row.get("anomaly_score", 0) or 0),
                str(row.get("anomaly_label", "")),
                model_version,
                now,
                json.dumps(row.astype(str).to_dict()),
            )
        )

    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO scored_trades (
                trader_id, trade_id, symbol, open_time, close_time, side,
                profit, session, anomaly_score, anomaly_label, model_version,
                created_at, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def replace_scored_trades(scored_df: pd.DataFrame, model_version: str = "v1"):
    now = datetime.utcnow().isoformat()
    rows = []

    for _, row in scored_df.iterrows():
        rows.append(
            (
                str(row.get("source_row_id", "")),
                str(row.get("source_row_id", "")),
                str(row.get("Symbol", "")),
                str(row.get("Open Date", "")),
                str(row.get("Closed Date", "")),
                str(row.get("Type", "")),
                float(row.get("Profit", 0) or 0),
                str(row.get("Session", "")),
                float(row.get("anomaly_score", 0) or 0),
                str(row.get("anomaly_label", "")),
                model_version,
                now,
                json.dumps(row.astype(str).to_dict()),
            )
        )

    with get_connection() as conn:
        conn.execute("DELETE FROM scored_trades")
        conn.executemany(
            """
            INSERT INTO scored_trades (
                trader_id, trade_id, symbol, open_time, close_time, side,
                profit, session, anomaly_score, anomaly_label, model_version,
                created_at, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def load_all_scored_trades() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM scored_trades", conn)


def load_latest_scored_dataset() -> pd.DataFrame | None:
    db_df = load_all_scored_trades()

    if db_df.empty or "raw_json" not in db_df.columns:
        return None

    records = db_df["raw_json"].dropna().apply(json.loads).tolist()

    if not records:
        return None

    scored_df = pd.DataFrame(records)

    numeric_cols = [
        "Open",
        "Closed",
        "SL",
        "TP",
        "Profit",
        "Lots",
        "Commission",
        "R/R",
        "%RR",
        "Consumed Time in Trade",
        "trade_duration_min",
        "risk_distance",
        "reward_distance",
        "realized_distance",
        "rr_planned",
        "rr_realized",
        "review_length",
        "is_win",
        "is_loss",
        "anomaly_score",
        "post_trade_drift_score",
        "post_trade_drift_flag",
        "post_trade_raw_score",
        "live_drift_score",
        "live_drift_flag",
        "live_raw_score",
        "behavioral_pressure_score",
        "behavior_label_available_count",
        "behavior_label_mean",
        "revenge_like_score",
        "impulsive_like_score",
        "fear_like_score",
        "greed_like_score",
        "stable_like_score",
        "high_frequency_flag",
        "rapid_reentry_flag",
        "risk_jump_flag",
        "negative_day_flag",
        "loseStreak",
        "winStreak",
        "prev_loss_streak",
        "prev_win_streak",
        "prev_loss_streak_scaled",
        "cum_pnl_today_before_trade",
        "Time Since Previous Trade Minutes",
        "Time Between Minutes",
        "riskChange",
        "discipline_score",
        "execution_score",
        "risk_management_score",
        "trade_count_today_before_trade",
        "session_trade_count_before_trade",
        "short_interval_flag",
        "high_frequency_before_flag",
        "trades_today",
        "First Trade of a Day",
        "risk_vs_rolling_avg",
        "poor_rr_flag",
        "oversized_risk_flag",
        "undersized_risk_flag",
        "loss_streak_scaled",
        "rolling_winrate_5",
        "rolling_avg_interval_5",
        "rolling_avg_risk_5",
        "manual_close_flag",
        "captured_ratio",
        "planned_move",
        "captured_move",
        "tp_distance_abs",
        "sl_distance_abs",
        "close_to_tp",
        "close_to_sl",
        "distance_to_tp_ratio",
        "premature_exit_flag",
    ]

    for col in numeric_cols:
        if col in scored_df.columns:
            scored_df[col] = pd.to_numeric(scored_df[col], errors="coerce")

    for col in ["Open Date", "Closed Date"]:
        if col in scored_df.columns:
            scored_df[col] = pd.to_datetime(scored_df[col], errors="coerce")

    return scored_df


# -----------------------------
# UI navigation
# -----------------------------
def go_to_dashboard():
    st.session_state["page"] = "dashboard"


def set_dashboard_section(section_name: str):
    st.session_state["dashboard_section"] = section_name


def open_trade_detail(trade_id: str):
    st.session_state["selected_trade_id"] = trade_id
    st.session_state["dashboard_section"] = "Trade Detail"


def initialize_state():
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    if "dashboard_section" not in st.session_state:
        st.session_state["dashboard_section"] = "Overview"

    if "selected_trade_id" not in st.session_state:
        st.session_state["selected_trade_id"] = None

    if "scored_df" not in st.session_state:
        st.session_state["scored_df"] = load_latest_scored_dataset()

    if "uploaded_preview_df" not in st.session_state:
        st.session_state["uploaded_preview_df"] = None

    if "template_preview_df" not in st.session_state:
        st.session_state["template_preview_df"] = build_template()

    if "upload_feedback" not in st.session_state:
        st.session_state["upload_feedback"] = {"remapped_columns": [], "auto_added_columns": []}

def render_home():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #212121;
        }

        .home-title {
            text-align: center;
            font-size: 48px;
            font-weight: 800;
            margin-top: 120px;
            margin-bottom: 15px;
            color: white;
        }

        .home-subtitle {
            text-align: center;
            font-size: 22px;
            color: #b8b8b8;
            margin-bottom: 25px;
        }

        div.stButton {
            display: flex;
            justify-content: center;
        }

        div.stButton > button {
            width: 220px;
            height: 48px;
            border-radius: 12px;
            font-size: 17px;
            font-weight: 600;
            background-color: #1db954;
            color: white;
            border: none;
        }

        div.stButton > button:hover {
            background-color: #1ed760;
            color: white;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="home-title">{APP_TITLE}</div>
        <div class="home-subtitle">
            Now, upload your trades, and improve your trading performance
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.button("Go to Dashboard", on_click=go_to_dashboard, use_container_width=True)


def feature_label(column_name):
    if column_name in FEATURE_LABELS:
        return FEATURE_LABELS[column_name]
    return column_name.replace("_", " ").replace("/", " / ").title()


def proxy_explanation(column_name):
    if column_name in PROXY_EXPLANATIONS:
        return PROXY_EXPLANATIONS[column_name]
    if column_name in TEMPLATE_COLUMNS:
        return "Original uploaded trade or journal field."
    return "Derived feature used by the behavioral drift system."


def display_dataframe(df, use_container_width=True, hide_index=True, key=None):
    display_df = df.copy()
    column_config = {}
    rename_map = {}
    used_labels = {}

    for col in display_df.columns:
        clean_name = feature_label(col)
        if clean_name in used_labels:
            used_labels[clean_name] += 1
            clean_name = f"{clean_name} {used_labels[clean_name]}"
        else:
            used_labels[clean_name] = 1
        rename_map[col] = clean_name
        column_config[clean_name] = st.column_config.Column(
            label=clean_name,
            help=proxy_explanation(col),
        )

    display_df = display_df.rename(columns=rename_map)

    return st.dataframe(
        display_df,
        use_container_width=use_container_width,
        hide_index=hide_index,
        column_config=column_config,
        key=key,
    )


def selectable_dataframe(df, key):
    display_df = df.copy()
    column_config = {}
    rename_map = {}
    used_labels = {}

    for col in display_df.columns:
        clean_name = feature_label(col)
        if clean_name in used_labels:
            used_labels[clean_name] += 1
            clean_name = f"{clean_name} {used_labels[clean_name]}"
        else:
            used_labels[clean_name] = 1
        rename_map[col] = clean_name
        column_config[clean_name] = st.column_config.Column(
            label=clean_name,
            help=proxy_explanation(col),
        )

    display_df = display_df.rename(columns=rename_map)

    return st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )


def friendly_value(column_name, value):
    if pd.isna(value):
        return "N/A"

    flag_columns = {
        "First Trade of a Day",
        "rapid_reentry_flag",
        "high_frequency_flag",
        "poor_rr_flag",
        "oversized_risk_flag",
        "undersized_risk_flag",
        "risk_jump_flag",
        "negative_day_flag",
        "manual_close_flag",
        "close_to_tp",
        "close_to_sl",
        "premature_exit_flag",
    }

    if isinstance(value, (np.bool_, bool)):
        return "Yes" if value else "No"

    if column_name in flag_columns:
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric_value) and numeric_value in [0, 1]:
            return "Yes" if numeric_value == 1 else "No"

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if column_name in flag_columns and value in [0.0, 1.0]:
            return "Yes" if value == 1 else "No"
        return round(float(value), 4)

    return value


def render_proxy_table(selected_trade, columns):
    existing_cols = [col for col in columns if col in selected_trade.index]

    if not existing_cols:
        return

    proxy_df = pd.DataFrame(
        {
            "Feature": [feature_label(col) for col in existing_cols],
            "Value": [
                friendly_value(col, selected_trade[col])
                for col in existing_cols
            ],
            "Explanation": [proxy_explanation(col) for col in existing_cols],
        }
    )

    st.dataframe(
        proxy_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": st.column_config.Column(
                "Feature",
                help="Clear name of the trade feature or behavioral proxy.",
            ),
            "Value": st.column_config.Column(
                "Value",
                help="Value of this feature for the selected trade.",
            ),
            "Explanation": st.column_config.Column(
                "Explanation",
                help="Why this feature matters for behavioral drift detection.",
            ),
        },
    )


def selected_trade_from_state(scored_df):
    selected_id = st.session_state.get("selected_trade_id")
    if not selected_id or scored_df is None:
        return None

    if "source_row_id" in scored_df.columns:
        matched_trade = scored_df[
            scored_df["source_row_id"].astype(str) == str(selected_id)
        ]
        if not matched_trade.empty:
            return matched_trade.iloc[0]

    if str(selected_id).isdigit():
        selected_position = int(selected_id)
        if 0 <= selected_position < len(scored_df):
            return scored_df.iloc[selected_position]

    return None


def render_trade_detail(selected_trade):
    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.subheader("Trade Detail Analysis")
    with top_right:
        if st.button("Back to All Trades", use_container_width=True):
            st.session_state["dashboard_section"] = "All Trades"
            st.rerun()

    st.write("### Raw Trade Row")

    basic_cols = [col for col in TEMPLATE_COLUMNS if col in selected_trade.index]
    raw_trade = selected_trade[basic_cols].to_frame().T
    display_dataframe(
        raw_trade,
        use_container_width=True,
        hide_index=True,
    )

    detail_groups = [
        ("Score Outputs", DETAIL_SCORE_COLUMNS),
        ("Timing and Frequency Proxies", DETAIL_TIMING_COLUMNS),
        ("Risk Management Proxies", DETAIL_RISK_COLUMNS),
        ("Behavioral Context Proxies", DETAIL_BEHAVIOR_COLUMNS),
    ]

    for group_title, group_columns in detail_groups:
        existing_cols = [col for col in group_columns if col in selected_trade.index]
        if existing_cols:
            st.write(f"### {group_title}")
            render_proxy_table(selected_trade, existing_cols)

    st.write("### Proxy-Based Explanation")

    explanations = []

    if selected_trade.get("trade_duration_min", 999) < 2:
        explanations.append("Very short trade duration may indicate impulsive execution.")

    if selected_trade.get("rr_realized", 999) < 0.5:
        explanations.append(
            "Realized risk/reward is low, which may indicate poor trade management."
        )

    if selected_trade.get("Profit", 0) < 0:
        explanations.append(
            "The trade ended with a loss, increasing behavioral risk context."
        )

    if selected_trade.get("review_length", 1) == 0:
        explanations.append(
            "No journal review was written, reducing behavioral self-reflection evidence."
        )

    if selected_trade.get("risk_distance", 1) == 0:
        explanations.append(
            "Risk distance is zero or invalid, which indicates poor risk definition."
        )

    if not explanations:
        explanations.append("No strong behavioral warning proxy was detected for this trade.")

    for exp in explanations:
        st.markdown(f"- {exp}")

    proxy_data = pd.DataFrame(
        {
            "Proxy": [
                "Trade Duration",
                "Planned R/R",
                "Realized R/R",
                "Review Length",
                "Risk Distance",
                "Profit",
            ],
            "Value": [
                selected_trade.get("trade_duration_min", 0),
                selected_trade.get("rr_planned", 0),
                selected_trade.get("rr_realized", 0),
                selected_trade.get("review_length", 0),
                selected_trade.get("risk_distance", 0),
                selected_trade.get("Profit", 0),
            ],
        }
    )

    proxy_data["How Score Is Determined"] = [
        "Trades shorter than 2 minutes increase the risk score.",
        "Low planned reward versus risk can signal weak setup structure.",
        "Low realized reward versus risk increases behavioral concern.",
        "Short or missing review text lowers evidence of self-reflection.",
        "Zero or invalid stop distance indicates poor risk definition.",
        "Negative profit increases the behavioral anomaly score.",
    ]

    display_dataframe(proxy_data, use_container_width=True, hide_index=True)

    fig_proxy = px.bar(
        proxy_data,
        x="Proxy",
        y="Value",
        title="Trade Proxy Overview",
        labels={
            "Proxy": "Proxy",
            "Value": "Value",
        },
    )

    st.plotly_chart(fig_proxy, use_container_width=True)


def format_percent(value):
    if pd.isna(value):
        return "N/A"
    return f"{round(value * 100, 1)}%"


def metric_mean(df, column):
    if column not in df.columns:
        return np.nan
    return pd.to_numeric(df[column], errors="coerce").mean()


def metric_rate(df, column):
    if column not in df.columns:
        return np.nan
    return pd.to_numeric(df[column], errors="coerce").fillna(0).mean()


def build_overview_proxies(scored_df):
    proxy_rows = []

    avg_anomaly = metric_mean(scored_df, "anomaly_score")
    proxy_rows.append((
        "overview_avg_anomaly_score",
        feature_label("overview_avg_anomaly_score"),
        "N/A" if pd.isna(avg_anomaly) else round(avg_anomaly, 3),
    ))

    if "anomaly_label" in scored_df.columns:
        high_risk_rate = (scored_df["anomaly_label"].astype(str) == "High Risk").mean()
    elif "anomaly_score" in scored_df.columns:
        high_risk_rate = (pd.to_numeric(scored_df["anomaly_score"], errors="coerce") >= 0.65).mean()
    else:
        high_risk_rate = np.nan
    proxy_rows.append((
        "overview_high_risk_rate",
        feature_label("overview_high_risk_rate"),
        format_percent(high_risk_rate),
    ))

    pressure = metric_mean(scored_df, "behavioral_pressure_score")
    proxy_rows.append((
        "overview_behavioral_pressure",
        feature_label("overview_behavioral_pressure"),
        "N/A" if pd.isna(pressure) else round(pressure, 3),
    ))

    if "Open Date" in scored_df.columns:
        temp = scored_df.copy()
        temp["Open Date"] = pd.to_datetime(temp["Open Date"], errors="coerce").dt.date
        avg_trades_per_day = temp.dropna(subset=["Open Date"]).groupby("Open Date").size().mean()
        avg_trades_per_day = "N/A" if pd.isna(avg_trades_per_day) else round(avg_trades_per_day, 2)
    else:
        avg_trades_per_day = "N/A"
    proxy_rows.append((
        "overview_avg_trades_per_day",
        feature_label("overview_avg_trades_per_day"),
        avg_trades_per_day,
    ))

    rapid_reentry = metric_rate(scored_df, "rapid_reentry_flag")
    proxy_rows.append((
        "overview_rapid_reentry_rate",
        feature_label("overview_rapid_reentry_rate"),
        format_percent(rapid_reentry),
    ))

    risk_jump = metric_rate(scored_df, "risk_jump_flag")
    proxy_rows.append((
        "overview_risk_jump_rate",
        feature_label("overview_risk_jump_rate"),
        format_percent(risk_jump),
    ))

    negative_day = metric_rate(scored_df, "negative_day_flag")
    proxy_rows.append((
        "overview_negative_day_entries",
        feature_label("overview_negative_day_entries"),
        format_percent(negative_day),
    ))

    if "loseStreak" in scored_df.columns:
        max_loss_streak = pd.to_numeric(scored_df["loseStreak"], errors="coerce").max()
        max_loss_streak = "N/A" if pd.isna(max_loss_streak) else int(max_loss_streak)
    else:
        max_loss_streak = "N/A"
    proxy_rows.append((
        "overview_max_loss_streak",
        feature_label("overview_max_loss_streak"),
        max_loss_streak,
    ))

    return proxy_rows


def render_dashboard():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #212121;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #212121 0%, #161616 100%);
        }

        h1, h2, h3, p, label, div {
            color: white;
        }

        .panel-shell {
            background-color: #181818;
            border: 1px solid #2b2b2b;
            border-radius: 18px;
            padding: 1.1rem 1rem 1.25rem 1rem;
            margin-bottom: 1rem;
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .panel-copy {
            color: #b8b8b8;
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
        }

        .db-badge-wrap {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            min-height: 2.6rem;
        }

        .db-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.42rem 0.75rem;
            font-size: 0.82rem;
            font-weight: 700;
            border: 1px solid #2f2f2f;
            background: #111111;
            white-space: nowrap;
        }

        .db-badge-ok {
            color: #1ed760;
        }

        .db-badge-error {
            color: #ff6b6b;
        }

        div.stButton > button {
            border-radius: 10px;
            background-color: #1db954;
            color: white;
            border: none;
            font-weight: 600;
            width: 100%;
        }

        div.stButton > button:hover {
            background-color: #1ed760;
            color: white;
            border: none;
        }

        [data-testid="stFileUploader"] {
            background-color: #111111;
            border: 1px dashed #3a3a3a;
            border-radius: 12px;
            padding: 0.5rem;
        }

        [data-testid="stFileUploader"] button {
            background-color: #1db954;
            color: white;
            border-radius: 10px;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state["scored_df"] is not None:
        scored_df = st.session_state["scored_df"].copy()
    else:
        scored_df = None

    db_ok, db_message = database_is_healthy()
    left, right = st.columns([1.1, 5.2], gap="large")

    with left:
        st.markdown(
            """
            <div class="panel-shell">
                <div class="panel-title">Upload Excel</div>
                <div class="panel-copy">
                    Upload a new trade file. The previous dataset in the database will be removed and replaced.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload an Excel file (.xlsx)",
            type=["xlsx"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_excel(uploaded_file)
                aligned_df, auto_added_columns, rename_map = align_uploaded_dataframe(
                    uploaded_df
                )
                duplicate_header_fixes = rename_map.pop(
                    "__duplicate_header_fixes__", ""
                )
                normalized_duplicate_fixes = rename_map.pop(
                    "__normalized_duplicate_fixes__", ""
                )
                remapped_columns = [
                    f"{source} -> {target}"
                    for source, target in rename_map.items()
                    if source != target
                ]
                st.session_state["uploaded_preview_df"] = aligned_df
                st.session_state["template_preview_df"] = build_template_from_dataframe(
                    uploaded_df
                )
                st.session_state["upload_feedback"] = {
                    "file_name": uploaded_file.name,
                    "remapped_columns": remapped_columns,
                    "auto_added_columns": auto_added_columns,
                    "duplicate_header_fixes": duplicate_header_fixes,
                    "normalized_duplicate_fixes": normalized_duplicate_fixes,
                }

                if st.button("Commit Excel To Database", use_container_width=True):
                    scored_df = score_trades(aligned_df)
                    replace_scored_trades(scored_df)
                    st.session_state["scored_df"] = scored_df
                    st.session_state["selected_trade_id"] = None
                    st.session_state["dashboard_section"] = "Overview"
                    st.success("New Excel uploaded, scored, and saved. Previous data was replaced.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read the Excel file: {exc}")

        st.markdown(
            """
            <div class="panel-shell">
                <div class="panel-title">Sections</div>
                <div class="panel-copy">
                    Use the dashboard areas below to review performance, inspect trades, and download the template.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_items = ["Overview", "All Trades", "Template"]
        for item in nav_items:
            st.button(
                item,
                key=f"nav_{item}",
                on_click=set_dashboard_section,
                args=(item,),
                use_container_width=True,
            )

    with right:
        title_col, status_col = st.columns([4.8, 1.2])
        with title_col:
            st.title("Trading Behavior Dashboard")
        with status_col:
            badge_class = "db-badge-ok" if db_ok else "db-badge-error"
            status_text = "Database connected" if db_ok else "Database error"
            st.markdown(
                f"""
                <div class="db-badge-wrap">
                    <div class="db-badge {badge_class}">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not db_ok:
                st.caption(db_message)

        section = st.session_state["dashboard_section"]

        if section == "Overview":
            st.subheader("General Overview")

            if scored_df is None:
                st.info("Upload an Excel file from the left panel to populate the dashboard.")
                return

            overview_proxies = build_overview_proxies(scored_df)

            first_row = st.columns(4)
            second_row = st.columns(4)
            metric_slots = first_row + second_row

            for slot, metric_data in zip(metric_slots, overview_proxies):
                metric_key, label, value = metric_data
                slot.metric(label, value, help=proxy_explanation(metric_key))

            st.divider()

            st.subheader("Top 5 Most Problematic Trades")
            problematic = scored_df.sort_values(
                "anomaly_score", ascending=False
            ).head(5)
            display_dataframe(problematic, use_container_width=True)

            st.subheader("Most Successful Trades")
            successful = scored_df.sort_values("Profit", ascending=False).head(5)
            display_dataframe(successful, use_container_width=True)

            chart_left, chart_right = st.columns(2)

            with chart_left:
                fig_score = px.histogram(
                    scored_df,
                    x="anomaly_score",
                    nbins=20,
                    title="Anomaly Score Distribution",
                    labels={"anomaly_score": feature_label("anomaly_score")},
                )
                st.plotly_chart(fig_score, use_container_width=True)

                if "Session" in scored_df.columns:
                    fig_session = px.box(
                        scored_df,
                        x="Session",
                        y="anomaly_score",
                        title="Anomaly Score by Session",
                        labels={
                            "Session": feature_label("Session"),
                            "anomaly_score": feature_label("anomaly_score"),
                        },
                    )
                    st.plotly_chart(fig_session, use_container_width=True)

            with chart_right:
                if "Profit" in scored_df.columns:
                    fig_profit = px.scatter(
                        scored_df,
                        x="Profit",
                        y="anomaly_score",
                        color="anomaly_label",
                        hover_data=["Symbol", "Type"] if "Symbol" in scored_df.columns and "Type" in scored_df.columns else None,
                        title="Profit vs Behavioral Anomaly Score",
                        labels={
                            "Profit": feature_label("Profit"),
                            "anomaly_score": feature_label("anomaly_score"),
                            "anomaly_label": feature_label("anomaly_label"),
                            "Symbol": feature_label("Symbol"),
                            "Type": feature_label("Type"),
                        },
                    )
                    st.plotly_chart(fig_profit, use_container_width=True)

                if "Open Date" in scored_df.columns:
                    trades_per_day = scored_df.copy()
                    trades_per_day["Open Date"] = pd.to_datetime(
                        trades_per_day["Open Date"], errors="coerce"
                    ).dt.date
                    trades_per_day = (
                        trades_per_day.dropna(subset=["Open Date"])
                        .groupby("Open Date")
                        .size()
                        .reset_index(name="Trades")
                    )

                    if not trades_per_day.empty:
                        fig_daily = px.line(
                            trades_per_day,
                            x="Open Date",
                            y="Trades",
                            markers=True,
                            title="Trades Per Day",
                            labels={
                                "Open Date": feature_label("Open Date"),
                                "Trades": "Trade Count",
                            },
                        )
                        st.plotly_chart(fig_daily, use_container_width=True)

        elif section == "All Trades":
            st.subheader("All Trades")

            if scored_df is None:
                st.info("No trades in the database yet.")
                return

            upload_feedback = st.session_state.get("upload_feedback", {})
            if upload_feedback.get("file_name"):
                st.caption(f"Current uploaded file: {upload_feedback['file_name']}")

            if upload_feedback.get("remapped_columns"):
                st.info(
                    "Matched uploaded headers to template fields: "
                    + ", ".join(upload_feedback["remapped_columns"])
                )

            if upload_feedback.get("duplicate_header_fixes"):
                st.warning(
                    "Duplicate uploaded headers were renamed automatically: "
                    + upload_feedback["duplicate_header_fixes"]
                )

            if upload_feedback.get("normalized_duplicate_fixes"):
                st.warning(
                    "Some normalized headers were still duplicated and were renamed: "
                    + upload_feedback["normalized_duplicate_fixes"]
                )

            if upload_feedback.get("auto_added_columns"):
                st.warning(
                    "Some template columns were missing and were added as empty: "
                    + ", ".join(upload_feedback["auto_added_columns"])
                )

            visible_trade_columns = [
                col for col in TEMPLATE_COLUMNS if col in scored_df.columns
            ]
            score_columns = [
                col
                for col in TABLE_SCORE_PROXY_COLUMNS
                if col in scored_df.columns
            ]

            table_df = scored_df[visible_trade_columns + score_columns].copy()

            st.caption("Click a row to open its full trade detail page.")
            selected_event = selectable_dataframe(
                table_df,
                key="all_trades_selectable_table",
            )

            selected_rows = selected_event.selection.rows
            if selected_rows:
                selected_row_position = selected_rows[0]
                selected_trade = scored_df.iloc[selected_row_position]
                if "source_row_id" in selected_trade.index:
                    st.session_state["selected_trade_id"] = str(selected_trade["source_row_id"])
                else:
                    st.session_state["selected_trade_id"] = str(selected_row_position)
                st.session_state["dashboard_section"] = "Trade Detail"
                st.rerun()

            scored_file_name, scored_mime = spreadsheet_download_info("scored_trades")
            st.download_button(
                label="Download Current Scored Dataset",
                data=to_excel_bytes(scored_df),
                file_name=scored_file_name,
                mime=scored_mime,
            )

        elif section == "Trade Detail":
            selected_trade = selected_trade_from_state(scored_df)

            if selected_trade is None:
                st.info("Select a trade from All Trades to inspect its details.")
                if st.button("Go to All Trades"):
                    st.session_state["dashboard_section"] = "All Trades"
                    st.rerun()
            else:
                render_trade_detail(selected_trade)

        elif section == "Template":
            st.subheader("Excel Template")

            template_df = st.session_state.get("template_preview_df")
            if template_df is None:
                template_df = build_template()
            display_dataframe(template_df, use_container_width=True)

            template_file_name, template_mime = spreadsheet_download_info("trader_trade_template")
            st.download_button(
                label="Download Template",
                data=to_excel_bytes(template_df),
                file_name=template_file_name,
                mime=template_mime,
            )

            st.write("Required columns:")
            required_columns_df = pd.DataFrame(
                {
                    "Column": get_expected_columns(),
                    "Explanation": [
                        proxy_explanation(column)
                        for column in get_expected_columns()
                    ],
                }
            )
            display_dataframe(required_columns_df, use_container_width=True)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    initialize_state()

    if st.session_state["page"] == "home":
        render_home()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
