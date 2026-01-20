# cleaning.py
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Filter Active companies
    if "Company Status (Active/Inactive)" in df.columns:
        df = df[df["Company Status (Active/Inactive)"] == "Active"]

    # Golden columns imputation
    golden_cols = ["Revenue (USD)", "Employees Total", "IT Budget", "Year Found"]
    for col in golden_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Categorical fill
    cat_cols = ["Ownership Type", "Entity Type", "Manufacturing Status"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Outlier capping
    for col in ["Revenue (USD)", "Employees Total"]:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)

    # Normalize country names
    if "Country" in df.columns:
        df["Country"] = df["Country"].astype(str).str.strip().str.upper()

    return df

