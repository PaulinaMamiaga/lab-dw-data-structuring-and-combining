# insurance_cleaning.py
# Data cleaning utilities for the insurance dataset

import csv
import pandas as pd


def load_data(csv: str) -> pd.DataFrame:
    """Load dataset from a CSV file path."""
    return pd.read_csv(csv)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names:
    - lowercase
    - replace spaces with underscores
    - replace 'st' only when the whole column name equals 'st' -> 'state'
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    df.columns = ["state" if col == "st" else col for col in df.columns]
    return df


def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean inconsistent categorical/text values (no NaN handling here):
    - gender normalization
    - state variants to full names
    - education: 'Bachelors' -> 'Bachelor'
    - customer_lifetime_value: remove '%' symbol (keeps as string for now)
    - vehicle_class: merge some categories into 'Luxury'
    """
    df = df.copy()

    if "gender" in df.columns:
        df["gender"] = df["gender"].replace({
            "Male": "M",
            "female": "F",
            "Femal": "F"
        })

    if "state" in df.columns:
        df["state"] = df["state"].replace({
            "AZ": "Arizona",
            "Cali": "California",
            "WA": "Washington"
        })

    if "education" in df.columns:
        df["education"] = df["education"].replace({
            "Bachelors": "Bachelor"
        })

    if "customer_lifetime_value" in df.columns:
        # Remove '%' character only (NaNs stay NaN)
        df["customer_lifetime_value"] = df["customer_lifetime_value"].str.replace("%", "", regex=False)

    if "vehicle_class" in df.columns:
        df["vehicle_class"] = df["vehicle_class"].replace({
            "Sports Car": "Luxury",
            "Luxury SUV": "Luxury",
            "Luxury Car": "Luxury"
        })

    return df


def format_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix incorrect data types:
    - customer_lifetime_value -> numeric
    - number_of_open_complaints -> int (extract middle value from '1/x/00')
    """
    df = df.copy()

    if "customer_lifetime_value" in df.columns:
        df["customer_lifetime_value"] = pd.to_numeric(df["customer_lifetime_value"], errors="coerce")

    if "number_of_open_complaints" in df.columns:
        # Example: '1/5/00' -> take the middle value -> '5'
        df["number_of_open_complaints"] = (
            df["number_of_open_complaints"]
            .astype(str)
            .str.split("/")
            .str[1]
        )
        df["number_of_open_complaints"] = pd.to_numeric(df["number_of_open_complaints"], errors="coerce")

    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - numeric columns: fill with median
    - categorical columns: fill with mode
    """
    df = df.copy()

    # Numeric columns (int/float)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns (object)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    return df


def convert_numeric_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns to integers.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        # Round first to avoid truncation surprises
        df[col] = df[col].round(0).astype(int)

    return df


def handle_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """
    Drop duplicate rows and reset index.
    """
    df = df.copy()
    df = df.drop_duplicates(keep=keep).reset_index(drop=True)
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned dataframe to CSV."""
    df.to_csv(output_path, index=False)


def main(
    url: str,
    output_path: str = "combined_data.csv",
    keep_duplicates: str = "first"
) -> pd.DataFrame:
    """
    Main pipeline:
    1) load
    2) clean column names
    3) clean invalid values
    4) format data types
    5) handle nulls
    6) convert numeric columns to integers
    7) drop duplicates + reset index
    8) save
    """
    df = load_data(csv)
    df = clean_column_names(df)
    df = clean_invalid_values(df)
    df = format_data_types(df)
    df = handle_nulls(df)
    df = convert_numeric_to_int(df)
    df = handle_duplicates(df, keep=keep_duplicates)
    save_cleaned_data(df, output_path)
    return df