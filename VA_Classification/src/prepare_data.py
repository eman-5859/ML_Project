"""
Clean the raw longitudinal AMD file and create a patient-level Train/Test split.

Run once from the project root:
    python -m src.prepare_data
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    RAW_DATA_PATH, TRAIN_PATH, TEST_PATH, TARGET_COL, GROUP_COL,
    BIOMARKER_COLS, CATEGORICAL_COLS, ORDINAL_COLS, VA_CLASSES,
    RANDOM_STATE,
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    print(f"Raw rows: {n0:,} | patients: {df[GROUP_COL].nunique():,}")

    df = df.drop_duplicates().copy()
    print(f"After dropping exact duplicates: {len(df):,}")

    # Rows with no label cannot be used for supervised learning (never impute a label)
    df = df.dropna(subset=[TARGET_COL])
    print(f"After dropping missing {TARGET_COL}: {len(df):,}")

    # The 8 biomarkers are missing together (visit without a usable scan)
    df = df.dropna(subset=BIOMARKER_COLS)
    print(f"After dropping missing biomarkers: {len(df):,}")

    # Tiny number of rows with missing demographics
    df = df.dropna(subset=CATEGORICAL_COLS + ORDINAL_COLS)
    print(f"After dropping missing demographics: {len(df):,}")

    # Safety: only known classes
    df = df[df[TARGET_COL].isin(VA_CLASSES)]
    print(f"Final usable rows: {len(df):,} | patients: {df[GROUP_COL].nunique():,} "
          f"(kept {len(df) / n0:.1%} of raw rows)")
    return df.reset_index(drop=True)


def patient_level_split(df: pd.DataFrame):
    """~80/20 split where each patient (ID) appears in only one side,
    with class proportions kept as close as possible."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df[TARGET_COL], groups=df[GROUP_COL]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Put the raw CSV at {RAW_DATA_PATH}")

    df = clean(pd.read_csv(RAW_DATA_PATH))
    train, test = patient_level_split(df)

    # Leakage check: no patient may appear in both files
    overlap = set(train[GROUP_COL]) & set(test[GROUP_COL])
    assert not overlap, f"{len(overlap)} patients appear in both train and test!"

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print("\n=== Split summary ===")
    print(f"Train: {len(train):,} rows, {train[GROUP_COL].nunique():,} patients")
    print(f"Test : {len(test):,} rows, {test[GROUP_COL].nunique():,} patients")
    summary = pd.DataFrame({
        "train_%": train[TARGET_COL].value_counts(normalize=True).mul(100).round(1),
        "test_%": test[TARGET_COL].value_counts(normalize=True).mul(100).round(1),
    }).reindex(VA_CLASSES)
    print("\nClass distribution (%):\n", summary)
    print(f"\nSaved:\n  {TRAIN_PATH}\n  {TEST_PATH}")


if __name__ == "__main__":
    main()