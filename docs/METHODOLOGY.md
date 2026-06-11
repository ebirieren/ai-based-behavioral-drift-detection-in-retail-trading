# Behavioral Drift Detection Methodology

## Project Framing

This project is a personalized anomaly detection system for retail trading behavior. It does not try to predict market direction. Instead, it evaluates whether a trader's current behavior deviates from their own historical trading and journaling patterns.

The system uses two related scoring views:

1. **Post-trade scoring** evaluates completed entered trades after all trade outcome information is known.
2. **Live recommendation scoring** estimates the behavioral risk before or at trade entry using only information that would be available at that time.

This distinction is important academically because post-trade features such as close time, captured move, and premature exit cannot be used for real-time recommendations without data leakage.

## Dataset Policy

The source workbook `data/behavioraldriftdetectiondataset.xlsx` is treated as the fixed raw dataset. The pipeline does not rewrite or restructure it. Derived files are generated separately.

Session values are deliberately recalculated from trade open time because the manually entered session values are not considered reliable enough for modeling.

## Label Treatment

All behavioral labels are treated consistently. The pipeline does not use a selected subset of labels for model-based filling. Labels may be:

- manually entered,
- rule-derived from journal text,
- missing.

Missing labels are not force-filled by an NLP classifier, because the available labeled sample is too small for reliable supervised learning.

Labels are not separated into "good" and "issue" groups in the current pipeline. This is important because future traders may upload datasets with different journaling meanings. The model therefore treats each behavioral label as a neutral numeric signal. If a label exists, it can influence the score; if it is missing, it is imputed by the model pipeline.

## Chronological Feature Policy

Features used for live recommendation must be available before the trade is entered. The most important correction is:

`cum_pnl_today_before_trade`

This is calculated as realized same-day PnL from trades that closed before the current trade opened. It does not use the current trade's profit and does not leak future trades from the same day.

## Model Definitions

Both scoring modes use Isolation Forest because the project is primarily anomaly detection and the dataset has limited reliable labels.

### Post-Trade Model

The post-trade model uses:

- timing features,
- risk features,
- previous streak and daily pressure features,
- completed-trade outcome behavior features,
- manual/rule behavioral scores when available,
- symbol, side, and session.

It outputs:

- `post_trade_drift_score`,
- `post_trade_drift_flag`,
- `post_trade_alert_level`,
- `recommendation`.

### Live Model

The live model excludes completed-trade outcome features. It uses:

- time since previous trade,
- same-symbol interval,
- planned risk,
- risk change,
- previous win/loss streak,
- trade count before this trade,
- session trade count before this trade,
- realized same-day PnL before this trade,
- rolling historical behavior,
- behavioral pressure score,
- symbol, side, and session.

It outputs:

- `live_drift_score`,
- `live_drift_flag`,
- `live_alert_level`,
- `recommendation`.

## Validation Strategy

The project does not depend on users manually reviewing trades after upload. Evaluation is derived from the uploaded dataset itself.

The project includes four validation views:

1. **Label coverage validation** shows how many reliable behavioral labels exist for each category.
2. **Temporal validation** trains on the earlier 70% of trades and tests on the later 30%, checking whether anomaly rates remain plausible over future trades.
3. **Ablation validation** compares a basic trade-context model, a behavioral-context model, and the full post-trade model. This shows whether streak, risk, timing, daily pressure, and journal-label features change the anomaly ranking.
4. **Data quality validation** reports source row count, date range, journal coverage, label coverage, and duplicate trade keys.

The phrase "future trades" in temporal validation means later trades inside the same uploaded dataset. No new external trades are required. The chronological split simulates whether a model trained on earlier trades behaves reasonably on later trades.

Current validation should be interpreted conservatively because the dataset is small and manually labeled behavior exists for only part of the trade history.

## Presentation Defense

Because the dataset does not contain complete ground-truth anomaly labels, classical supervised metrics such as accuracy, F1-score, and AUC are not the primary evaluation method. The system is evaluated through temporal validation, ablation testing, interpretability of detected anomalies, feature explanation, recommendation text, and consistency with available journal labels.
