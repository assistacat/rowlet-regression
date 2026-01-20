# feature_pipeline.py
from __future__ import annotations

import re
import json
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cleaning import clean_data

CURRENT_YEAR = 2026


TECH_SIGNAL_COLS = [
    "IT Budget",
    "IT spend",
    "No. of PC",
    "No. of Desktops",
    "No. of Laptops",
    "No. of Routers",
    "No. of Servers",
    "No. of Storage Devices",
]

LOW_CARD_COLS = ["Ownership Type", "Entity Type", "Manufacturing Status"]

RANGE_RE = re.compile(r"^\s*(\d+)\s*to\s*(\d+)\s*$", re.IGNORECASE)
PLUS_RE = re.compile(r"^\s*(\d+)\s*\+\s*$")


def _to_numeric_countish(s: pd.Series) -> pd.Series:
    """
    Converts count-like columns that may contain strings like:
      - "1 to 10" -> midpoint (5.5)
      - "1000+" -> 1000
      - "12" -> 12
    Non-parsable -> NaN
    """
    out = pd.to_numeric(s, errors="coerce")
    mask = out.isna() & s.notna()

    if mask.any():
        raw = s[mask].astype(str).str.strip()

        m = raw.str.extract(RANGE_RE)
        a = pd.to_numeric(m[0], errors="coerce")
        b = pd.to_numeric(m[1], errors="coerce")
        midpoint = (a + b) / 2.0

        p = raw.str.extract(PLUS_RE)
        xplus = pd.to_numeric(p[0], errors="coerce")

        filled = pd.Series(np.nan, index=raw.index, dtype="float64")
        filled[midpoint.notna()] = midpoint[midpoint.notna()]
        filled[(filled.isna()) & (xplus.notna())] = xplus[(filled.isna()) & (xplus.notna())]

        out.loc[mask] = filled

    return out


def add_company_age(df: pd.DataFrame, year_col: str = "Year Found") -> pd.DataFrame:
    out = df.copy()
    yf = pd.to_numeric(out.get(year_col), errors="coerce")
    age = CURRENT_YEAR - yf
    age = age.where(age.notna(), np.nan)
    median_age = float(np.nanmedian(age.values)) if np.isfinite(np.nanmedian(age.values)) else 0.0
    out["company_age"] = age.fillna(median_age).clip(lower=0)
    return out


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rev = pd.to_numeric(out.get("Revenue (USD)"), errors="coerce").fillna(0).clip(lower=0)
    emp = pd.to_numeric(out.get("Employees Total"), errors="coerce").fillna(0).clip(lower=0)

    out["log1p_revenue_usd"] = np.log1p(rev)
    out["log1p_employees_total"] = np.log1p(emp)
    return out


def add_rev_it_per_employee(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rev = pd.to_numeric(out.get("Revenue (USD)"), errors="coerce").fillna(0).clip(lower=0)
    emp = pd.to_numeric(out.get("Employees Total"), errors="coerce").fillna(0).clip(lower=0)

    out["revenue_per_employee"] = rev / (emp + 1)

    it_budget_col = "IT Budget" if "IT Budget" in out.columns else ("IT spend" if "IT spend" in out.columns else None)
    if it_budget_col is None:
        it = pd.Series(0.0, index=out.index)
    else:
        it = pd.to_numeric(out[it_budget_col], errors="coerce").fillna(0).clip(lower=0)

    out["it_spend_per_employee"] = it / (emp + 1)
    return out


def add_tech_intensity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    available = [c for c in TECH_SIGNAL_COLS if c in out.columns]
    if not available:
        out["tech_intensity"] = np.nan
        return out

    binaries = []
    for c in available:
        if c.startswith("No. of"):
            num = _to_numeric_countish(out[c])
        else:
            num = pd.to_numeric(out[c], errors="coerce")
        binaries.append((num.fillna(0) > 0).astype(int))

    out["tech_intensity"] = pd.concat(binaries, axis=1).mean(axis=1)
    return out


def add_size_bucket(df: pd.DataFrame, emp_col: str = "Employees Total") -> pd.DataFrame:
    out = df.copy()
    emp = pd.to_numeric(out.get(emp_col), errors="coerce").fillna(0).clip(lower=0)

    out["size_bucket"] = pd.cut(
        emp,
        bins=[-np.inf, 9, 49, 249, 999, np.inf],
        labels=["Micro", "Small", "Medium", "Large", "Enterprise"],
    )
    out["size_bucket"] = out["size_bucket"].astype(str).replace({"nan": "Unknown"})
    return out


def _extract_digits_prefix(series: pd.Series, n: int) -> pd.Series:
    s = series.astype(str).str.replace(r"\D+", "", regex=True)
    s = s.replace({"": np.nan})
    return s.str[:n]


def add_naics_sic_groups(df: pd.DataFrame, rare_min_count: int = 50) -> pd.DataFrame:
    out = df.copy()

    if "NAICS Code" in out.columns:
        out["naics2"] = _extract_digits_prefix(out["NAICS Code"], 2)
        out["naics3"] = _extract_digits_prefix(out["NAICS Code"], 3)
    else:
        out["naics2"] = np.nan
        out["naics3"] = np.nan

    if "SIC Code" in out.columns:
        out["sic2"] = _extract_digits_prefix(out["SIC Code"], 2)
        out["sic3"] = _extract_digits_prefix(out["SIC Code"], 3)
    else:
        out["sic2"] = np.nan
        out["sic3"] = np.nan

    for col in ["naics2", "sic2"]:
        vc = out[col].value_counts(dropna=True)
        rare = set(vc[vc < rare_min_count].index.astype(str).tolist())
        out[col] = out[col].astype(str)
        out.loc[out[col].isin(rare), col] = "Other"
        out.loc[out[col].isin(["nan", "None"]), col] = "Other"

    return out


def one_hot_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].fillna("Unknown").astype(str)
    dummies = pd.get_dummies(out[cols], prefix=[f"oh_{c}" for c in cols], drop_first=False)
    out = pd.concat([out.drop(columns=[c for c in cols if c in out.columns]), dummies], axis=1)
    return out


def standard_scale(df: pd.DataFrame, numeric_features: List[str]) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    out = df.copy()

    for f in numeric_features:
        if f not in out.columns:
            out[f] = np.nan

    X = out[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_cols = [f"scaled_{f}" for f in numeric_features]
    out[scaled_cols] = X_scaled
    return out, scaler, scaled_cols


def build_features(
    input_path: str,
    *,
    output_csv: str = "features_out.csv",
    scaler_json: str = "scaler_params.json",
) -> None:
    if input_path.lower().endswith(".csv"):
        df_raw = pd.read_csv(input_path)
    else:
        df_raw = pd.read_excel(input_path)

    df = clean_data(df_raw)

    df = add_company_age(df)
    df = add_log_features(df)
    df = add_rev_it_per_employee(df)
    df = add_tech_intensity(df)
    df = add_size_bucket(df)
    df = add_naics_sic_groups(df, rare_min_count=50)

    cols_present = [c for c in LOW_CARD_COLS if c in df.columns]
    if cols_present:
        df = one_hot_encode(df, cols_present)

    numeric_features = [
        "company_age",
        "log1p_revenue_usd",
        "log1p_employees_total",
        "revenue_per_employee",
        "it_spend_per_employee",
        "tech_intensity",
    ]
    df, scaler, scaled_cols = standard_scale(df, numeric_features)

    df.to_csv(output_csv, index=False)

    scaler_payload: Dict[str, object] = {
        "features": numeric_features,
        "scaled_features": scaled_cols,
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
    }
    with open(scaler_json, "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, indent=2)

    print("Saved:", output_csv)
    print("Saved:", scaler_json)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    build_features(
        input_path="champions_group_data.xlsx",
        output_csv="champions_features.csv",
        scaler_json="champions_scaler.json",
    )
