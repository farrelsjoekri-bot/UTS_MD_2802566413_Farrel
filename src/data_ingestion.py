from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"

FEATURE_FILE = BASE_DIR / "A.csv"
TARGET_FILE = BASE_DIR / "A_targets.csv"

OUTPUT_FILE = INGESTED_DIR / "dataset.csv"


def data_ingestion():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Target file not found: {TARGET_FILE}")

    df = pd.read_csv(FEATURE_FILE)
    target = pd.read_csv(TARGET_FILE)

    # Validasi
    if df.empty or target.empty:
        raise ValueError("Dataset is empty")

    if len(df) != len(target):
        raise ValueError("Feature and target row mismatch")
    
    #erasing dupes
    target = target.loc[:, ~target.columns.isin(df.columns)]

    # Merge
    full_df = pd.concat([df, target], axis=1)

    #in case of dupe
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]

    # Save
    full_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Data ingested successfully: {OUTPUT_FILE}")

    return full_df


if __name__ == "__main__":
    data_ingestion()