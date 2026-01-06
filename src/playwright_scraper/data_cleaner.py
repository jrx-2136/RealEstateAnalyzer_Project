import pandas as pd
from pathlib import Path

def clean_location_csv(
    input_csv="data/outputs/magicbricks_india_properties.csv",
    output_csv="data/outputs/magicbricks_india_properties_cleaned.csv",
    drop_empty_location = False
):
    """
    Cleans the scraped CSV file.

    - Removes rows with empty / null location (optional)
    - Saves a new cleaned CSV (does NOT overwrite raw data)

    Returns:
    - Cleaned pandas DataFrame
    """

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)
    before_rows = len(df)

    if drop_empty_location:
        if "location" not in df.columns:
            raise ValueError("Column 'location' not found in CSV")

        df = df.dropna(subset=["location"])
        df = df[df["location"].str.strip() != ""]

    after_rows = len(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[Cleaner] Rows before: {before_rows}")
    print(f"[Cleaner] Rows after : {after_rows}")
    print(f"[Cleaner] Saved cleaned file → {output_csv}")

    return df
