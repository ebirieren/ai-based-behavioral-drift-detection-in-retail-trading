import pandas as pd

from model_training import fit_and_score_model, make_isolation_forest
from pipeline_config import BEHAVIOR_LABEL_COLUMNS


def build_label_coverage(df):
    rows = []

    for col in BEHAVIOR_LABEL_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "label": col,
                "known": int(values.notna().sum()),
                "value_1": int((values == 1).sum()),
                "value_0": int((values == 0).sum()),
                "missing": int(values.isna().sum()),
                "coverage": float(values.notna().mean()),
            }
        )

    return pd.DataFrame(rows)


def build_temporal_validation(df, post_numeric, live_numeric, categorical):
    split_index = int(len(df) * 0.70)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    rows = []
    for model_name, numeric_features in [
        ("post_trade", post_numeric),
        ("live", live_numeric),
    ]:
        numeric_features = [
            col
            for col in numeric_features
            if col in train_df.columns and not train_df[col].isna().all()
        ]
        categorical_features = [col for col in categorical if col in train_df.columns]

        x_train = train_df[numeric_features + categorical_features].copy()
        x_test = test_df[numeric_features + categorical_features].copy()

        model = make_isolation_forest(numeric_features, categorical_features)
        model.fit(x_train)
        flags = model.predict(x_test)

        rows.append(
            {
                "model": model_name,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_anomaly_rate": float((flags == -1).mean()),
                "expected_contamination": 0.12,
                "note": "Temporal validation checks whether later trades are flagged at a plausible rate.",
            }
        )

    return pd.DataFrame(rows)


def build_data_quality(df, rule_summary):
    rows = [
        {"metric": "row_count", "value": len(df)},
        {"metric": "symbol_count", "value": df["Symbol"].nunique()},
        {"metric": "date_min", "value": str(df["Open Time"].min())},
        {"metric": "date_max", "value": str(df["Open Time"].max())},
        {"metric": "short_review_non_empty", "value": int(df["Short Review"].str.strip().ne("").sum())},
        {"metric": "long_review_non_empty", "value": int(df["Long Review"].str.strip().ne("").sum())},
        {"metric": "behavior_label_rows", "value": int(df[BEHAVIOR_LABEL_COLUMNS].notna().any(axis=1).sum())},
        {"metric": "duplicate_trade_keys", "value": int(df.duplicated(["Symbol", "Type", "Open Time", "Closed Time", "Profit"]).sum())},
    ]

    for index, row in rule_summary.iterrows():
        rows.append(
            {
                "metric": "rule_fills__" + row["label"],
                "value": int(row["rule_positive_fills"]),
            }
        )

    return pd.DataFrame(rows)


def build_ablation_test(df, label_feature_cols, categorical_features):
    basic_numeric = [
        "Time Since Previous Trade Minutes",
        "Time Between Minutes",
        "%RR",
        "R/R",
        "trade_count_today_before_trade",
    ]

    behavior_numeric = basic_numeric + [
        "riskChange",
        "prev_win_streak",
        "prev_loss_streak",
        "session_trade_count_before_trade",
        "cum_pnl_today_before_trade",
        "rolling_winrate_5",
        "rolling_avg_interval_5",
        "risk_vs_rolling_avg",
        "short_interval_flag",
        "high_frequency_before_flag",
        "prev_loss_streak_scaled",
        "negative_day_flag",
        "behavioral_pressure_score",
    ]

    post_trade_numeric = behavior_numeric + [
        "Consumed Time in Trade",
        "captured_ratio",
        "distance_to_tp_ratio",
        "premature_exit_flag",
        "poor_rr_flag",
        "oversized_risk_flag",
        "undersized_risk_flag",
        "risk_jump_flag",
        "rapid_reentry_flag",
        "behavior_label_available_count",
        "behavior_label_mean",
    ] + label_feature_cols

    variants = [
        ("basic_trade_context", basic_numeric),
        ("behavior_context", behavior_numeric),
        ("post_trade_full", post_trade_numeric),
    ]

    full_top = set(df.nlargest(20, "post_trade_drift_score")["trade_sequence"])
    rows = []

    for variant_name, numeric_features in variants:
        scored_df, model = fit_and_score_model(
            df,
            numeric_features,
            categorical_features,
            "ablation",
        )

        top_rows = scored_df.nlargest(20, "ablation_drift_score")
        variant_top = set(top_rows["trade_sequence"])
        overlap = len(full_top.intersection(variant_top)) / max(len(full_top), 1)
        score_corr = scored_df["ablation_drift_score"].corr(
            df["post_trade_drift_score"],
            method="spearman",
        )

        rows.append(
            {
                "model_variant": variant_name,
                "feature_count": len([col for col in numeric_features if col in df.columns]) + len(categorical_features),
                "anomaly_flag_count": int((scored_df["ablation_drift_flag"] == -1).sum()),
                "anomaly_flag_rate": float((scored_df["ablation_drift_flag"] == -1).mean()),
                "top20_overlap_with_full_model": float(overlap),
                "score_correlation_with_full_model": float(score_corr),
                "avg_behavioral_pressure_in_top20": float(top_rows["behavioral_pressure_score"].mean()),
                "purpose": get_ablation_purpose(variant_name),
            }
        )

    return pd.DataFrame(rows)


def build_contamination_sensitivity(df, post_numeric, live_numeric, categorical_features):
    contamination_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    rows = []

    model_setups = [
        ("post_trade", post_numeric, "post_trade_drift_score"),
        ("live", live_numeric, "live_drift_score"),
    ]

    for model_name, numeric_features, default_score_col in model_setups:
        default_top20 = set(df.nlargest(20, default_score_col)["trade_sequence"])

        for contamination in contamination_values:
            scored_df, model = fit_and_score_model(
                df,
                numeric_features,
                categorical_features,
                "sensitivity",
                contamination,
            )

            top20 = set(scored_df.nlargest(20, "sensitivity_drift_score")["trade_sequence"])
            top20_overlap = len(default_top20.intersection(top20)) / max(len(default_top20), 1)
            score_correlation = scored_df["sensitivity_drift_score"].corr(
                df[default_score_col],
                method="spearman",
            )

            rows.append(
                {
                    "model": model_name,
                    "contamination": contamination,
                    "anomaly_flag_count": int((scored_df["sensitivity_drift_flag"] == -1).sum()),
                    "anomaly_flag_rate": float((scored_df["sensitivity_drift_flag"] == -1).mean()),
                    "top20_overlap_with_default_012": float(top20_overlap),
                    "score_correlation_with_default_012": float(score_correlation),
                    "avg_pressure_of_flagged_trades": float(
                        scored_df.loc[
                            scored_df["sensitivity_drift_flag"] == -1,
                            "behavioral_pressure_score",
                        ].mean()
                    ),
                    "note": "Contamination changes the alert threshold; it should not be treated as true anomaly percentage.",
                }
            )

    return pd.DataFrame(rows)


def get_ablation_purpose(variant_name):
    if variant_name == "basic_trade_context":
        return "Baseline using simple trade context without richer behavioral signals."
    if variant_name == "behavior_context":
        return "Shows the effect of streak, risk change, frequency, and daily PnL context."
    return "Full post-trade model including completed-trade behavior and neutral journal label features."


def build_feature_explanation():
    rows = [
        {
            "feature_group": "Timing",
            "examples": "time since previous trade, same-symbol interval, consumed time",
            "why_it_matters": "Very short intervals can indicate impulsive re-entry or overtrading.",
        },
        {
            "feature_group": "Risk",
            "examples": "%RR, risk change, oversized risk flag, poor RR flag",
            "why_it_matters": "Risk jumps and poor reward/risk structure can indicate unstable decision making.",
        },
        {
            "feature_group": "Streak",
            "examples": "previous win streak, previous loss streak, previous trade loss",
            "why_it_matters": "Loss streaks can support revenge-like behavior; win streaks plus risk increase can support greed-like behavior.",
        },
        {
            "feature_group": "Daily Pressure",
            "examples": "trade count before trade, session count, realized PnL before trade",
            "why_it_matters": "High frequency and negative realized PnL can increase behavioral pressure.",
        },
        {
            "feature_group": "Post-Trade Execution",
            "examples": "captured ratio, distance to TP, premature exit flag, manual close flag",
            "why_it_matters": "Completed trades can be evaluated for execution quality after the trade closes.",
        },
        {
            "feature_group": "Journal Labels",
            "examples": "uploaded behavioral label columns",
            "why_it_matters": "Labels are treated as neutral signals, not forced into good or bad categories.",
        },
        {
            "feature_group": "Recommendation",
            "examples": "analysis_reason and recommendation",
            "why_it_matters": "The system is not only score-based; alerts include explanation text and an action suggestion.",
        },
    ]

    return pd.DataFrame(rows)


def export_results(
    df,
    label_coverage,
    temporal_validation,
    data_quality,
    ablation_test,
    contamination_sensitivity,
    feature_explanation,
    output_path,
):
    score_cols = [
        "trade_sequence",
        "source_row",
        "Open Date",
        "Open Time",
        "Closed Time",
        "Symbol",
        "Type",
        "Session",
        "Profit",
        "%RR",
        "R/R",
        "Consumed Time in Trade",
        "Time Since Previous Trade Minutes",
        "Time Between Minutes",
        "trade_count_today_before_trade",
        "session_trade_count_before_trade",
        "cum_pnl_today_before_trade",
        "prev_win_streak",
        "prev_loss_streak",
        "behavioral_pressure_score",
        "post_trade_drift_score",
        "post_trade_drift_flag",
        "post_trade_alert_level",
        "live_drift_score",
        "live_drift_flag",
        "live_alert_level",
        "dominant_pattern",
        "analysis_reason",
        "recommendation",
        "revenge_like_score",
        "impulsive_like_score",
        "fear_like_score",
        "greed_like_score",
        "stable_like_score",
        "behavior_label_available_count",
        "behavior_label_mean",
    ]

    score_cols = [col for col in score_cols if col in df.columns]

    with pd.ExcelWriter(output_path) as writer:
        df[score_cols].sort_values("post_trade_drift_score", ascending=False).to_excel(
            writer,
            sheet_name="Scores",
            index=False,
        )
        label_coverage.to_excel(writer, sheet_name="LabelCoverage", index=False)
        temporal_validation.to_excel(writer, sheet_name="TemporalValidation", index=False)
        ablation_test.to_excel(writer, sheet_name="AblationTest", index=False)
        contamination_sensitivity.to_excel(writer, sheet_name="ContaminationSensitivity", index=False)
        feature_explanation.to_excel(writer, sheet_name="FeatureExplanation", index=False)
        data_quality.to_excel(writer, sheet_name="DataQuality", index=False)
