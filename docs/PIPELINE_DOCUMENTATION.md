# Behavioral Drift Pipeline Documentation

This document explains the current `behavioral_drift_pipeline.py` system after splitting the large file into smaller files.

The goal of the pipeline is:

1. Read the raw trade/journal dataset.
2. Clean the dataset without changing the original Excel file.
3. Produce behavioral proxy features.
4. Train two Isolation Forest models.
5. Score all trades.
6. Produce behavioral pattern labels and recommendations.
7. Export result and validation files for thesis/presentation.

The code is intentionally divided into small files so it is easier to explain in a graduation project.

## Compatibility With `userInterface.py`

The split pipeline is now connected to `userInterface.py` for uploaded trade scoring.

Important detail:

- `behavioral_drift_pipeline.py` is the batch research/scoring pipeline used to generate thesis outputs and model files.
- `userInterface.py` is the dashboard application used by the trader.
- When a user uploads an Excel file, `userInterface.py` now reuses the split pipeline functions instead of only using its older UI-side scoring logic.
- The UI produces the new split-pipeline score fields:
  - `post_trade_drift_score`
  - `post_trade_alert_level`
  - `live_drift_score`
  - `live_alert_level`
  - `dominant_pattern`
  - `recommendation`
- The UI also keeps the older display columns:
  - `anomaly_score`
  - `anomaly_label`

In the UI, `anomaly_score` is a dashboard-friendly alias for `post_trade_drift_score`, and `anomaly_label` is a dashboard-friendly alias for `post_trade_alert_level`. This keeps the previous dashboard screens working while allowing the detail page to show both the post-trade evaluation score and the live recommendation score.

The UI still keeps its older heuristic fallback. This is only used if the split pipeline cannot score an uploaded file. The main scoring path is now the split pipeline path.

The split pipeline still saves the research model files:

- `models/behavior_model_post_trade.pkl`
- `models/behavior_model_live.pkl`
- `models/behavior_model_bundle.pkl`

The current UI scores the uploaded dataset directly through the split modules. The saved model files are mainly for thesis/research reproducibility and future deployment work.

## File Structure

### `behavioral_drift_pipeline.py`

This is the main runner file.

It does not contain all cleaning/model logic anymore. It only controls the order of the pipeline.

Main flow:

```text
load dataset
add rule labels
add chronological proxies
add journal label features
add post-trade proxies
train post-trade model
train live model
create recommendations
create validation sheets
export results
save models
```

### `pipeline_config.py`

This file stores common constants.

It contains:

- input dataset path,
- output file paths,
- behavior label column names,
- text rule target names,
- keyword rules used for journal text.

This file exists so the same column lists are not repeated in many files.

### `text_label_rules.py`

This file handles journal text cleaning and rule-based label filling.

It does not train an NLP model. It only applies simple keyword/regex rules.

### `data_preprocessing.py`

This file handles:

- reading the Excel dataset,
- cleaning labels,
- parsing dates,
- recalculating sessions,
- creating chronological features,
- creating behavioral proxy features,
- creating post-trade proxy features.

### `model_training.py`

This file handles:

- selecting model features,
- creating the Isolation Forest model,
- fitting the model,
- producing normalized drift scores.

### `recommendations.py`

This file handles:

- alert levels,
- behavioral pattern scores,
- dominant pattern selection,
- explanation text,
- recommendation text.

### `validation_outputs.py`

This file creates the result workbook sheets:

- label coverage,
- temporal validation,
- data quality,
- ablation test,
- feature explanation,
- final scored trades export.

## Main File Functions

### `save_models(post_trade_model, live_model, post_numeric, live_numeric, categorical)`

Role:

Saves trained models and model metadata to `.pkl` files.

Inputs:

- `post_trade_model`: model trained with completed-trade features.
- `live_model`: model trained with live/pre-trade features.
- `post_numeric`: numeric columns used by the post-trade model.
- `live_numeric`: numeric columns used by the live model.
- `categorical`: categorical columns used by both models.

Produces:

- `models/behavior_model_post_trade.pkl`
- `models/behavior_model_live.pkl`
- `models/behavior_model_bundle.pkl`

Why it matters:

The saved files allow the system to reuse the trained models later instead of only keeping results inside memory.

### `run_pipeline(dataset_path, output_path)`

Role:

Runs the full behavioral drift pipeline from raw dataset to final output.

Inputs:

- `dataset_path`: Excel file path.
- `output_path`: result Excel file path.

Produces:

- cleaned and scored trade table,
- trained models,
- validation tables,
- Excel result file,
- summary printed to terminal.

Important steps inside:

1. `load_dataset`
2. `add_rule_labels`
3. `add_chronological_features`
4. `add_label_features`
5. `add_post_trade_features`
6. `get_feature_sets`
7. `fit_and_score_model` for post-trade model
8. `fit_and_score_model` for live model
9. `add_patterns_and_recommendations`
10. validation sheet creation
11. `export_results`
12. `save_models`

### `main()`

Role:

Allows the pipeline to be run from the terminal.

Example:

```bash
python3 behavioral_drift_pipeline.py
```

Optional example:

```bash
python3 behavioral_drift_pipeline.py --dataset data/behavioraldriftdetectiondataset.xlsx --output outputs/behavioral_drift_results.xlsx
```

## Config File

### `DATASET_PATH`

Default raw dataset:

```text
data/behavioraldriftdetectiondataset.xlsx
```

### `RESULTS_PATH`

Default output workbook:

```text
outputs/behavioral_drift_results.xlsx
```

### `POST_TRADE_MODEL_PATH`

Saved post-trade model:

```text
models/behavior_model_post_trade.pkl
```

### `LIVE_MODEL_PATH`

Saved live recommendation model:

```text
models/behavior_model_live.pkl
```

### `MODEL_BUNDLE_PATH`

Saved model bundle:

```text
models/behavior_model_bundle.pkl
```

This contains both models and the feature column names used during training.

### `BEHAVIOR_LABEL_COLUMNS`

Role:

Stores the journal/manual label columns.

Important design choice:

These labels are not separated as "good" or "bad". They are treated as neutral uploaded signals because another trader may use labels differently.

### `RULE_TARGETS`

Role:

Stores the label columns that can be supported by simple text rules.

### `TEXT_RULES`

Role:

Stores Turkish/English keyword patterns used to find behavioral clues inside journal text.

Example:

If review text contains words related to FOMO, the rule may fill `Entered Too Soon`.

## Text Rule Functions

### `clean_text(text)`

Role:

Cleans journal text before rule matching.

What it does:

- handles missing text,
- removes Turkish character accents,
- converts text to lowercase,
- removes extra spaces.

Input:

- one text value.

Output:

- cleaned text string.

### `apply_text_rules(text, patterns)`

Role:

Checks if a text matches any rule pattern.

Input:

- journal text,
- list of patterns.

Output:

- `1.0` if any pattern is found,
- `NaN` if no pattern is found.

Why `NaN`:

No match does not always mean the behavior did not happen. It only means the text rule could not detect it.

### `add_rule_labels(df)`

Role:

Applies text rules to missing behavioral label cells.

Input:

- dataframe with `Combined Review`.

Output:

- updated dataframe,
- rule summary table.

Rule summary contains:

- label name,
- how many missing rows were checked,
- how many were filled by rules,
- how many stayed missing.

## Preprocessing Functions

### `clean_label_columns(df)`

Role:

Converts label columns into numeric values.

What it does:

- converts `True`/`False` style values to `1`/`0`,
- converts labels to numeric,
- creates source columns like `Entered Too Soon_source`,
- marks existing labels as `manual`.

Output:

- dataframe with cleaned label columns.

### `set_sessions_from_open_time(df)`

Role:

Recalculates session names using trade open time.

Why:

The project assumes manually entered session values are not reliable enough. Session is therefore derived from time rules.

Output:

- dataframe with corrected `Session` values.

### `prepare_dataset_dataframe(input_df)`

Role:

Prepares an already loaded dataframe for the pipeline.

Why it exists:

The batch pipeline reads from Excel, but the UI already receives an uploaded dataframe from Streamlit. This function lets both flows use the same cleaning logic.

What it does:

- copies the uploaded/raw dataframe,
- adds `source_row`,
- cleans label columns,
- parses open/close dates,
- converts numeric trade columns,
- recalculates session,
- creates `Combined Review`.

Output:

- cleaned dataframe ready for rule labels and feature engineering.

### `load_dataset(dataset_path)`

Role:

Loads the raw Excel dataset and performs basic cleaning.

What it does:

- reads Excel,
- calls `prepare_dataset_dataframe`.

Output:

- cleaned dataframe ready for feature engineering.

### `add_chronological_features(df)`

Role:

Creates time-based and sequence-based behavioral proxies.

Important produced columns:

- `trade_sequence`
- `Consumed Time in Trade`
- `Time Since Previous Trade Minutes`
- `Time Between Minutes`
- `First Trade of a Day`
- `trade_count_today_before_trade`
- `session_trade_count_before_trade`
- `cum_pnl_today_before_trade`
- `winStreak`
- `loseStreak`
- `prev_win_streak`
- `prev_loss_streak`
- `riskChange`
- `rolling_winrate_5`
- `rolling_avg_interval_5`
- `rolling_avg_risk_5`
- `risk_vs_rolling_avg`
- `short_interval_flag`
- `high_frequency_before_flag`
- `negative_day_flag`
- `behavioral_pressure_score`

Most important correction:

`cum_pnl_today_before_trade` uses only trades that closed before the current trade opened. This avoids using future information.

### `add_label_features(df)`

Role:

Turns behavioral label columns into model features.

Important:

Labels are treated neutrally. The function does not decide that a label is good or bad.

Produced columns:

- `label_...` columns for each behavior label,
- `behavior_label_available_count`,
- `behavior_label_mean`.

Output:

- dataframe,
- list of created label feature columns.

### `add_post_trade_features(df)`

Role:

Creates features that are only available after the trade is closed.

Produced columns:

- `planned_move`
- `captured_move`
- `captured_ratio`
- `tp_distance_abs`
- `sl_distance_abs`
- `close_to_tp`
- `close_to_sl`
- `manual_close_flag`
- `distance_to_tp_ratio`
- `poor_rr_flag`
- `oversized_risk_flag`
- `undersized_risk_flag`
- `risk_jump_flag`
- `rapid_reentry_flag`
- `premature_exit_flag`

Why separate:

These features are valid for post-trade evaluation, but not for live recommendation before a trade is closed.

## Model Functions

### `get_feature_sets(label_feature_cols)`

Role:

Defines which features go into the two models.

Outputs:

- post-trade numeric feature list,
- live numeric feature list,
- categorical feature list.

Post-trade model uses:

- live features,
- completed-trade features,
- label feature columns.

Live model uses:

- features available before or at trade entry.

### `make_isolation_forest(numeric_features, categorical_features)`

Role:

Creates the Isolation Forest model pipeline.

Pipeline parts:

1. Numeric columns:
   - missing values filled with median,
   - values scaled.
2. Categorical columns:
   - missing values filled with most frequent value,
   - categories one-hot encoded.
3. Isolation Forest:
   - detects unusual trades.

Important note:

`contamination=0.12` is not ground truth. It is a model setting that tells the model roughly how many trades to treat as unusual.

### `fit_and_score_model(df, numeric_features, categorical_features, prefix)`

Role:

Trains one Isolation Forest model and adds drift score columns.

Input:

- dataframe,
- numeric feature names,
- categorical feature names,
- prefix such as `post_trade` or `live`.

Output:

- updated dataframe,
- trained model.

Produced columns:

- `{prefix}_raw_score`
- `{prefix}_drift_flag`
- `{prefix}_drift_score`

Example:

- `post_trade_drift_score`
- `live_drift_score`

## Recommendation Functions

### `get_alert_level(score, drift_flag, pressure)`

Role:

Converts model score and behavioral pressure into a simple alert label.

Output:

- `Normal`
- `Caution`
- `High Risk`

### `add_patterns_and_recommendations(df)`

Role:

Adds alert levels, behavioral pattern scores, explanations, and recommendations.

Produced columns:

- `post_trade_alert_level`
- `live_alert_level`
- `revenge_like_score`
- `impulsive_like_score`
- `fear_like_score`
- `greed_like_score`
- `stable_like_score`
- `dominant_pattern`
- `analysis_reason`
- `recommendation`

### `choose_dominant_pattern(row)`

Role:

Chooses the most important behavioral pattern for a trade.

Possible outputs:

- `revenge-like`
- `impulsive-like`
- `fear-like`
- `greed-like`
- `stable-like`

### `build_reason(row)`

Role:

Creates explanation text for why a trade was risky.

Example reasons:

- short interval between trades,
- high trade frequency,
- sudden risk increase,
- possible premature exit,
- low risk/reward,
- negative realized PnL before trade,
- post-trade anomaly signal,
- live anomaly signal.

### `build_recommendation(row)`

Role:

Creates user-facing recommendation text.

Example outputs:

- pause before the next trade,
- reduce risk,
- avoid increasing risk after winning trades,
- trade only planned setups,
- normal trading conditions.

## Validation and Output Functions

### `build_label_coverage(df)`

Role:

Shows how much behavioral label data exists.

Output sheet:

`LabelCoverage`

Columns:

- label,
- known,
- value_1,
- value_0,
- missing,
- coverage.

### `build_temporal_validation(df, post_numeric, live_numeric, categorical)`

Role:

Simulates future behavior using the same dataset.

How:

- first 70% of trades are used for training,
- last 30% are used for testing.

Important:

"Future" means later rows in the existing dataset, not new external trades.

Output sheet:

`TemporalValidation`

### `build_data_quality(df, rule_summary)`

Role:

Creates general dataset quality information.

Output sheet:

`DataQuality`

Examples:

- row count,
- symbol count,
- date range,
- journal coverage,
- label rows,
- duplicate trade keys,
- rule fill counts.

### `build_ablation_test(df, label_feature_cols, categorical_features)`

Role:

Compares different feature groups.

Variants:

1. `basic_trade_context`
2. `behavior_context`
3. `post_trade_full`

Why:

This helps show whether behavioral features change the anomaly ranking.

Output sheet:

`AblationTest`

Important columns:

- feature count,
- anomaly flag count,
- anomaly flag rate,
- top 20 overlap with full model,
- score correlation with full model,
- average behavioral pressure in top 20.

### `get_ablation_purpose(variant_name)`

Role:

Explains what each ablation variant means.

### `build_contamination_sensitivity(df, post_numeric, live_numeric, categorical_features)`

Role:

Tests different Isolation Forest contamination values.

Values tested:

- `0.05`
- `0.08`
- `0.10`
- `0.12`
- `0.15`
- `0.20`

Output sheet:

`ContaminationSensitivity`

Important meaning:

Contamination is not the true anomaly percentage. It is a threshold setting that controls how many trades the model flags as anomalous.

Important columns:

- model,
- contamination,
- anomaly flag count,
- anomaly flag rate,
- top 20 overlap with default `0.12`,
- score correlation with default `0.12`,
- average behavioral pressure of flagged trades.

Use in thesis:

This helps defend that contamination was selected as a sensitivity parameter, not as ground truth.

### `build_feature_explanation()`

Role:

Creates a thesis/presentation-friendly table that explains feature groups.

Output sheet:

`FeatureExplanation`

Feature groups:

- Timing,
- Risk,
- Streak,
- Daily Pressure,
- Post-Trade Execution,
- Journal Labels,
- Recommendation.

### `export_results(...)`

Role:

Writes final Excel workbook.

Output:

`outputs/behavioral_drift_results.xlsx`

Sheets:

- `Scores`
- `LabelCoverage`
- `TemporalValidation`
- `AblationTest`
- `ContaminationSensitivity`
- `FeatureExplanation`
- `DataQuality`

## Generated Files and Their Roles

### `outputs/behavioral_drift_results.xlsx`

This is the main result workbook.

It contains all scores, recommendations, and validation tables.

#### `Scores` sheet

Represents:

- every trade,
- post-trade drift score,
- live drift score,
- alert level,
- dominant behavior pattern,
- explanation,
- recommendation.

Main columns:

- `post_trade_drift_score`
- `post_trade_alert_level`
- `live_drift_score`
- `live_alert_level`
- `dominant_pattern`
- `analysis_reason`
- `recommendation`

#### `LabelCoverage` sheet

Represents:

How complete the uploaded behavioral labels are.

Use in thesis:

Shows why supervised accuracy/F1/AUC are not the main evaluation method.

#### `TemporalValidation` sheet

Represents:

Chronological validation using earlier trades for training and later trades for testing.

Use in thesis:

Shows that the model is evaluated in a time-aware way.

#### `AblationTest` sheet

Represents:

The effect of adding behavioral proxy groups.

Use in thesis:

Supports the claim that the system is not only profit/loss based.

#### `ContaminationSensitivity` sheet

Represents:

How results change when different contamination values are used.

Use in thesis:

Shows that contamination changes the number of flagged trades. It should be defended as a practical threshold setting, not the true anomaly rate.

#### `FeatureExplanation` sheet

Represents:

Human-readable explanation of feature groups.

Use in thesis:

Supports interpretability.

#### `DataQuality` sheet

Represents:

Basic quality checks about the dataset.

Use in thesis:

Shows dataset limitations and coverage.

### `models/behavior_model_post_trade.pkl`

This file stores the trained post-trade Isolation Forest model.

It uses completed-trade features, so it is for evaluating entered/closed trades.

### `models/behavior_model_live.pkl`

This file stores the trained live recommendation Isolation Forest model.

It uses features available before or at trade entry.

### `models/behavior_model_bundle.pkl`

This file stores:

- post-trade model,
- live model,
- post-trade feature list,
- live feature list,
- categorical feature list,
- behavior label columns,
- note explaining the model purpose.

This is useful if another part of the project needs to load both models together.

## How To Run

Run:

```bash
python3 behavioral_drift_pipeline.py
```

On this computer, the Miniforge Python environment has the needed Excel/data-science packages:

```bash
/Users/ebirieren/miniforge3/bin/python3 behavioral_drift_pipeline.py
```

If normal `python3` gives an `openpyxl` error, use the Miniforge command above.

Run with custom paths:

```bash
python3 behavioral_drift_pipeline.py --dataset data/behavioraldriftdetectiondataset.xlsx --output outputs/behavioral_drift_results.xlsx
```

Expected output files:

- `outputs/behavioral_drift_results.xlsx`
- `models/behavior_model_post_trade.pkl`
- `models/behavior_model_live.pkl`
- `models/behavior_model_bundle.pkl`

## How To Explain In Presentation

Suggested explanation:

```text
The project is not a supervised classifier because there is no complete ground-truth anomaly label.
Therefore, the model is evaluated through temporal validation, ablation testing, feature explanation,
data quality checks, and interpretable recommendation outputs.
```

Another useful explanation:

```text
The post-trade model evaluates completed trades. The live model supports recommendations using only features
available before or at trade entry. This prevents future information leakage.
```
