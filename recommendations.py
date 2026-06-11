def get_alert_level(score, drift_flag, pressure):
    if score >= 0.75 or pressure >= 0.65:
        return "High Risk"
    if score >= 0.60 or drift_flag == -1:
        return "Caution"
    return "Normal"


def add_patterns_and_recommendations(df):
    df = df.copy()

    df["post_trade_alert_level"] = df.apply(
        lambda row: get_alert_level(
            row["post_trade_drift_score"],
            row["post_trade_drift_flag"],
            row["behavioral_pressure_score"],
        ),
        axis=1,
    )

    df["live_alert_level"] = df.apply(
        lambda row: get_alert_level(
            row["live_drift_score"],
            row["live_drift_flag"],
            row["behavioral_pressure_score"],
        ),
        axis=1,
    )

    df["revenge_like_score"] = (
        0.35 * df["prev_trade_lose"].fillna(0)
        + 0.25 * df["short_interval_flag"].fillna(0)
        + 0.20 * df["prev_loss_streak_scaled"].fillna(0)
        + 0.20 * df["risk_jump_flag"].fillna(0)
    )
    df["impulsive_like_score"] = (
        0.30 * df["short_interval_flag"].fillna(0)
        + 0.25 * df["high_frequency_before_flag"].fillna(0)
        + 0.20 * df["premature_exit_flag"].fillna(0)
        + 0.15 * (df["trade_count_today_before_trade"] >= 3).astype(int)
        + 0.10 * df["risk_jump_flag"].fillna(0)
    )
    df["fear_like_score"] = (
        0.35 * df["undersized_risk_flag"].fillna(0)
        + 0.30 * df["premature_exit_flag"].fillna(0)
        + 0.20 * (df["distance_to_tp_ratio"] > 0.5).fillna(False).astype(int)
        + 0.15 * df["prev_trade_lose"].fillna(0)
    )
    df["greed_like_score"] = (
        0.30 * (df["prev_win_streak"] >= 2).astype(int)
        + 0.25 * df["risk_jump_flag"].fillna(0)
        + 0.20 * df["oversized_risk_flag"].fillna(0)
        + 0.15 * df["high_frequency_before_flag"].fillna(0)
        + 0.10 * (df["cum_pnl_today_before_trade"] > 0).astype(int)
    )
    df["stable_like_score"] = (
        0.35 * (1 - df["post_trade_drift_score"])
        + 0.30 * (1 - df["behavioral_pressure_score"].fillna(0))
        + 0.20 * (1 - df["revenge_like_score"].fillna(0))
        + 0.15 * (1 - df["impulsive_like_score"].fillna(0))
    )

    df["dominant_pattern"] = df.apply(choose_dominant_pattern, axis=1)
    df["analysis_reason"] = df.apply(build_reason, axis=1)
    df["recommendation"] = df.apply(build_recommendation, axis=1)

    return df


def choose_dominant_pattern(row):
    risk_scores = {
        "revenge-like": row["revenge_like_score"],
        "impulsive-like": row["impulsive_like_score"],
        "fear-like": row["fear_like_score"],
        "greed-like": row["greed_like_score"],
    }
    strongest_risk_pattern = max(risk_scores, key=risk_scores.get)
    strongest_risk_score = risk_scores[strongest_risk_pattern]

    if (
        row["post_trade_alert_level"] == "Normal"
        and row["live_alert_level"] == "Normal"
        and row["stable_like_score"] >= strongest_risk_score
    ):
        return "stable-like"

    return strongest_risk_pattern


def build_reason(row):
    reasons = []

    if row.get("short_interval_flag", 0) == 1:
        reasons.append("short interval between trades")
    if row.get("high_frequency_before_flag", 0) == 1:
        reasons.append("high trade frequency before this trade")
    if row.get("risk_jump_flag", 0) == 1:
        reasons.append("sudden risk increase")
    if row.get("premature_exit_flag", 0) == 1:
        reasons.append("possible premature exit")
    if row.get("poor_rr_flag", 0) == 1:
        reasons.append("low risk/reward")
    if row.get("oversized_risk_flag", 0) == 1:
        reasons.append("oversized risk")
    if row.get("undersized_risk_flag", 0) == 1:
        reasons.append("undersized risk")
    if row.get("negative_day_flag", 0) == 1:
        reasons.append("negative realized PnL before trade")
    if row.get("prev_win_streak", 0) >= 2 and row.get("risk_jump_flag", 0) == 1:
        reasons.append("risk increase after winning streak")
    if row.get("post_trade_drift_flag", 1) == -1:
        reasons.append("post-trade anomaly signal")
    if row.get("live_drift_flag", 1) == -1:
        reasons.append("live anomaly signal")

    if len(reasons) == 0:
        return "no strong anomaly signal"
    return "; ".join(reasons)


def build_recommendation(row):
    if (
        row["live_alert_level"] == "High Risk"
        and row.get("negative_day_flag", 0) == 1
        and row.get("prev_loss_streak", 0) >= 2
    ):
        return "Do not open a new trade today unless a very strong planned setup appears."

    if row["live_alert_level"] == "High Risk":
        if row.get("dominant_pattern", "") == "greed-like":
            return "Avoid increasing risk after winning trades; keep risk at the planned level."
        return "Pause before the next trade and reduce risk if trading continues."

    if row["live_alert_level"] == "Caution":
        return "Trade only planned setups and avoid increasing position size."

    if row.get("prev_win_streak", 0) >= 3 and row.get("behavioral_pressure_score", 0) < 0.35:
        return "Behavior looks stable; risk may stay at the planned level."

    return "Normal trading conditions; follow the plan."
