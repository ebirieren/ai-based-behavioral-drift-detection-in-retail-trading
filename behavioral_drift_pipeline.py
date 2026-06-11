import argparse
import json
import pickle
from pathlib import Path

from data_preprocessing import (
    add_chronological_features,
    add_label_features,
    add_post_trade_features,
    load_dataset,
)
from model_training import fit_and_score_model, get_feature_sets
from pipeline_config import (
    BEHAVIOR_LABEL_COLUMNS,
    DATASET_PATH,
    LIVE_MODEL_PATH,
    MODEL_BUNDLE_PATH,
    POST_TRADE_MODEL_PATH,
    RESULTS_PATH,
)
from recommendations import add_patterns_and_recommendations
from text_label_rules import add_rule_labels
from validation_outputs import (
    build_ablation_test,
    build_contamination_sensitivity,
    build_data_quality,
    build_feature_explanation,
    build_label_coverage,
    build_temporal_validation,
    export_results,
)


def save_models(post_trade_model, live_model, post_numeric, live_numeric, categorical):
    POST_TRADE_MODEL_PATH.parent.mkdir(exist_ok=True)
    LIVE_MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_BUNDLE_PATH.parent.mkdir(exist_ok=True)

    POST_TRADE_MODEL_PATH.write_bytes(pickle.dumps(post_trade_model))
    LIVE_MODEL_PATH.write_bytes(pickle.dumps(live_model))

    MODEL_BUNDLE_PATH.write_bytes(
        pickle.dumps(
            {
                "post_trade_model": post_trade_model,
                "live_model": live_model,
                "post_trade_numeric_features": post_numeric,
                "live_numeric_features": live_numeric,
                "categorical_features": categorical,
                "behavior_label_columns": BEHAVIOR_LABEL_COLUMNS,
                "note": "Post-trade model evaluates completed trades. Live model supports recommendation using pre-trade features.",
            }
        )
    )


def run_pipeline(dataset_path, output_path):
    output_path.parent.mkdir(exist_ok=True)

    df = load_dataset(dataset_path)
    df, rule_summary = add_rule_labels(df)

    df = add_chronological_features(df)
    df, label_feature_cols = add_label_features(df)
    df = add_post_trade_features(df)

    post_numeric, live_numeric, categorical = get_feature_sets(label_feature_cols)

    df, post_trade_model = fit_and_score_model(df, post_numeric, categorical, "post_trade")
    df, live_model = fit_and_score_model(df, live_numeric, categorical, "live")
    df = add_patterns_and_recommendations(df)

    label_coverage = build_label_coverage(df)
    temporal_validation = build_temporal_validation(df, post_numeric, live_numeric, categorical)
    data_quality = build_data_quality(df, rule_summary)
    ablation_test = build_ablation_test(df, label_feature_cols, categorical)
    contamination_sensitivity = build_contamination_sensitivity(df, post_numeric, live_numeric, categorical)
    feature_explanation = build_feature_explanation()

    export_results(
        df,
        label_coverage,
        temporal_validation,
        data_quality,
        ablation_test,
        contamination_sensitivity,
        feature_explanation,
        output_path,
    )

    save_models(post_trade_model, live_model, post_numeric, live_numeric, categorical)

    summary = {
        "rows": int(len(df)),
        "output_path": str(output_path),
        "post_trade_high_risk": int((df["post_trade_alert_level"] == "High Risk").sum()),
        "live_high_risk": int((df["live_alert_level"] == "High Risk").sum()),
        "behavior_label_rows": int(df[BEHAVIOR_LABEL_COLUMNS].notna().any(axis=1).sum()),
        "temporal_validation": temporal_validation.to_dict(orient="records"),
        "ablation_test": ablation_test.to_dict(orient="records"),
        "contamination_sensitivity": contamination_sensitivity.to_dict(orient="records"),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run behavioral drift detection pipeline.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--output", default=str(RESULTS_PATH))
    args = parser.parse_args()

    summary = run_pipeline(Path(args.dataset), Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
