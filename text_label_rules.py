import re
import unicodedata

import numpy as np
import pandas as pd

from pipeline_config import RULE_TARGETS, TEXT_RULES


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.translate(
        str.maketrans({"ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ö": "o", "ü": "u"})
    )
    text = re.sub(r"\s+", " ", text)
    return text


def apply_text_rules(text, patterns):
    text = clean_text(text)

    for pattern in patterns:
        if re.search(clean_text(pattern), text):
            return 1.0

    return np.nan


def add_rule_labels(df):
    df = df.copy()
    summary_rows = []

    for col in RULE_TARGETS:
        source_col = col + "_source"
        missing_mask = df[col].isna()
        rule_predictions = df.loc[missing_mask, "Combined Review"].apply(
            lambda text: apply_text_rules(text, TEXT_RULES[col])
        )

        matched_idx = rule_predictions[rule_predictions == 1].index
        df.loc[matched_idx, col] = 1.0
        df.loc[matched_idx, source_col] = "rule"

        summary_rows.append(
            {
                "label": col,
                "missing_checked": int(missing_mask.sum()),
                "rule_positive_fills": int(len(matched_idx)),
                "still_missing": int(df[col].isna().sum()),
            }
        )

    return df, pd.DataFrame(summary_rows)
