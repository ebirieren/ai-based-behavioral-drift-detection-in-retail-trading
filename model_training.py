import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_sets(label_feature_cols):
    live_numeric_features = [
        "Time Since Previous Trade Minutes",
        "Time Between Minutes",
        "%RR",
        "riskChange",
        "prev_win_streak",
        "prev_loss_streak",
        "trade_count_today_before_trade",
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

    post_trade_numeric_features = live_numeric_features + [
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

    categorical_features = ["Symbol", "Type", "Session"]

    return post_trade_numeric_features, live_numeric_features, categorical_features


def make_isolation_forest(numeric_features, categorical_features, contamination=0.12):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "iso",
                IsolationForest(
                    n_estimators=300,
                    contamination=contamination,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def fit_and_score_model(df, numeric_features, categorical_features, prefix, contamination=0.12):
    df = df.copy()

    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    x = df[numeric_features + categorical_features].copy()
    model = make_isolation_forest(numeric_features, categorical_features, contamination)
    model.fit(x)

    transformed_x = model.named_steps["prep"].transform(x)
    raw_score = model.named_steps["iso"].decision_function(transformed_x)

    score_min = raw_score.min()
    score_max = raw_score.max()
    normalized_score = 1 - ((raw_score - score_min) / (score_max - score_min + 1e-9))

    df[prefix + "_raw_score"] = raw_score
    df[prefix + "_drift_flag"] = model.predict(x)
    df[prefix + "_drift_score"] = normalized_score

    return df, model
