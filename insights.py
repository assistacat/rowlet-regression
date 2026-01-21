# insights.py
# Person 4 (Insights) — Day 2 aligned with app.py
#
# Required by app.py:
#   - ensure_derived_metrics(df)
#   - build_cluster_profile(df, cluster_col="Cluster", metrics=[...], extra_group_cols=[...])
#   - make_radar_fig(df, company_id, id_col="DUNS Number", cluster_col="Cluster",
#                    metrics=[...], company_name_col="Company Sites")
#
# Optional helpers:
#   - compute_anomaly_score(...)
#   - quick_business_take(...)

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ----------------------------
# Defaults
# ----------------------------

DEFAULT_METRICS: List[str] = [
    "Revenue (USD)",
    "Employees Total",
    "IT Spend per Employee",
    "Revenue per Employee",
    "Tech Intensity Score",
    "Company Age",
]


# ----------------------------
# Utilities
# ----------------------------

def _to_numeric(s: pd.Series) -> pd.Series:
    """Convert to numeric safely."""
    return pd.to_numeric(s, errors="coerce")


def _mode_or_nan(series: pd.Series) -> Any:
    """Return the mode (most frequent non-null value) or np.nan."""
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0]


# ----------------------------
# Derived metrics
# ----------------------------

TECH_SIGNAL_COLS = [
    "IT Budget", "IT spend",
    "No. of PC", "No. of Desktops", "No. of Laptops",
    "No. of Routers", "No. of Servers", "No. of Storage Devices",
]


def ensure_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures these derived metrics exist (using known dataset column names):
      - Company Age (median-imputed)
      - Revenue per Employee
      - IT Spend per Employee
      - Tech Intensity Score (0–1) from binary tech signals

    Assumes canonical columns:
      Revenue (USD), Employees Total, IT spend, Year Found
    """
    out = df.copy()

    # --- Company Age (median imputation) ---
    if "Company Age" not in out.columns:
        if "Year Found" in out.columns:
            year = pd.to_numeric(out["Year Found"], errors="coerce")
            age = datetime.now().year - year
            age = age.where(age >= 0, np.nan)
            out["Company Age"] = age.fillna(age.median(skipna=True))
        else:
            out["Company Age"] = np.nan

    # --- Revenue per Employee ---
    if "Revenue per Employee" not in out.columns:
        if "Revenue (USD)" in out.columns and "Employees Total" in out.columns:
            rev = pd.to_numeric(out["Revenue (USD)"], errors="coerce").fillna(0)
            emp = pd.to_numeric(out["Employees Total"], errors="coerce").fillna(0)
            out["Revenue per Employee"] = rev / (emp + 1.0)
        else:
            out["Revenue per Employee"] = np.nan

    # --- IT Spend per Employee ---
    if "IT Spend per Employee" not in out.columns:
        if "IT spend" in out.columns and "Employees Total" in out.columns:
            it = pd.to_numeric(out["IT spend"], errors="coerce").fillna(0)
            emp = pd.to_numeric(out["Employees Total"], errors="coerce").fillna(0)
            out["IT Spend per Employee"] = it / (emp + 1.0)
        else:
            out["IT Spend per Employee"] = np.nan

    # --- Tech Intensity Score (0–1) ---
    if "Tech Intensity Score" not in out.columns:
        cols = [c for c in TECH_SIGNAL_COLS if c in out.columns]
        if not cols:
            out["Tech Intensity Score"] = np.nan
        else:
            bin_signals = []
            for c in cols:
                s = out[c]
                if pd.api.types.is_bool_dtype(s):
                    b = s.fillna(False).astype(int)
                else:
                    b = pd.to_numeric(s, errors="coerce").fillna(0)
                    b = (b > 0).astype(int)
                bin_signals.append(b)

            out["Tech Intensity Score"] = pd.concat(bin_signals, axis=1).mean(axis=1)

    return out


# ----------------------------
# Required by app.py
# ----------------------------

def build_cluster_profile(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    metrics: Optional[List[str]] = None,
    extra_group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a cluster profile table for UI + report.

    Output columns:
      - cluster_col (e.g., "Cluster")
      - Cluster Size
      - <metric>_mean for each metric
      - <metric>_median for each metric (extra but useful)
      - Top <extra_group_col> for each extra group column
    """
    if metrics is None:
        metrics = DEFAULT_METRICS
    if extra_group_cols is None:
        extra_group_cols = []

    if cluster_col not in df.columns:
        raise ValueError(f"'{cluster_col}' not found in dataframe. Did you apply clustering?")

    work = df.copy()

    # Ensure metrics exist; if missing, create them as NaN so aggregation doesn't crash
    for m in metrics:
        if m not in work.columns:
            work[m] = np.nan

    grouped = work.groupby(cluster_col, dropna=False)

    # Aggregate mean/median
    agg_dict: Dict[str, List[str]] = {m: ["mean", "median"] for m in metrics}
    prof = grouped.agg(agg_dict)
    prof.columns = [f"{col}_{stat}" for col, stat in prof.columns]  # flatten MultiIndex
    prof = prof.reset_index()

    # Cluster size
    sizes = grouped.size().reset_index(name="Cluster Size")
    prof = prof.merge(sizes, on=cluster_col, how="left")

    # Extra group columns (top mode)
    for c in extra_group_cols:
        if c not in work.columns:
            prof[f"Top {c}"] = np.nan
            continue
        top_vals = grouped[c].apply(_mode_or_nan).reset_index(name=f"Top {c}")
        prof = prof.merge(top_vals, on=cluster_col, how="left")

    # Provide an alias if cluster_col isn't "Cluster"
    if cluster_col != "Cluster" and "Cluster" not in prof.columns:
        prof["Cluster"] = prof[cluster_col]

    # Sort clusters for consistent display
    try:
        prof = prof.sort_values(by=cluster_col).reset_index(drop=True)
    except Exception:
        pass

    return prof


def make_radar_fig(
    df: pd.DataFrame,
    company_id: Any,
    id_col: str = "DUNS Number",
    cluster_col: str = "Cluster",
    metrics: Optional[List[str]] = None,
    company_name_col: Optional[str] = None,
) -> go.Figure:
    """
    Create a Plotly radar chart comparing:
      - the selected company
      - its cluster average (mean)

    Returns a Plotly figure.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' not found in dataframe.")
    if cluster_col not in df.columns:
        raise ValueError(f"'{cluster_col}' not found in dataframe. Did you apply clustering?")

    work = ensure_derived_metrics(df.copy())

    # Pull selected company row
    row_df = work.loc[work[id_col] == company_id]
    if row_df.empty:
        raise ValueError(f"Company id '{company_id}' not found in column '{id_col}'.")
    row = row_df.iloc[0]

    cluster_val = row[cluster_col]
    cluster_df = work.loc[work[cluster_col] == cluster_val]

    labels: List[str] = []
    company_vals: List[float] = []
    cluster_vals: List[float] = []

    for m in metrics:
        if m not in work.columns:
            continue
        labels.append(m)

        comp_v = _to_numeric(pd.Series([row[m]])).iloc[0]
        clus_v = _to_numeric(cluster_df[m]).mean(skipna=True)

        company_vals.append(comp_v)
        cluster_vals.append(clus_v)

    if not labels:
        raise ValueError("No valid metrics available for radar chart.")

    # Normalize to 0–1 for display (per-metric, based on company vs cluster values)
    comp_norm: List[float] = []
    clus_norm: List[float] = []

    for cv, gv in zip(company_vals, cluster_vals):
        if pd.isna(cv) and pd.isna(gv):
            comp_norm.append(0.5)
            clus_norm.append(0.5)
            continue

        cv2 = 0.0 if pd.isna(cv) else float(cv)
        gv2 = 0.0 if pd.isna(gv) else float(gv)

        mn = min(cv2, gv2)
        mx = max(cv2, gv2)

        if mx - mn < 1e-9:
            comp_norm.append(0.5)
            clus_norm.append(0.5)
        else:
            comp_norm.append((cv2 - mn) / (mx - mn))
            clus_norm.append((gv2 - mn) / (mx - mn))

    # Close the loop for radar
    labels_loop = labels + [labels[0]]
    comp_loop = comp_norm + [comp_norm[0]]
    clus_loop = clus_norm + [clus_norm[0]]

    # Display name
    company_name = str(company_id)
    if company_name_col and company_name_col in work.columns:
        n = row.get(company_name_col, None)
        if pd.notna(n):
            company_name = str(n)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=clus_loop,
            theta=labels_loop,
            fill="toself",
            name="Cluster avg",
            opacity=0.55,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=comp_loop,
            theta=labels_loop,
            fill="toself",
            name=company_name,
            opacity=0.75,
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        margin=dict(l=30, r=30, t=30, b=30),
    )

    return fig


# ----------------------------
# Optional helpers
# ----------------------------

def compute_anomaly_score(
    df: pd.DataFrame,
    company_id: Any,
    id_col: str = "DUNS Number",
    cluster_col: str = "Cluster",
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Returns a small table of within-cluster z-scores for a company across metrics.
    Higher absolute z-score => more unusual within its cluster.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    work = ensure_derived_metrics(df.copy())

    if id_col not in work.columns or cluster_col not in work.columns:
        raise ValueError("Missing id_col or cluster_col.")

    row_df = work.loc[work[id_col] == company_id]
    if row_df.empty:
        raise ValueError("Company not found.")
    row = row_df.iloc[0]

    cl = row[cluster_col]
    cluster_df = work.loc[work[cluster_col] == cl]

    records = []
    for m in metrics:
        if m not in work.columns:
            continue

        vals = _to_numeric(cluster_df[m])
        mu = vals.mean(skipna=True)
        sd = vals.std(skipna=True)

        x = _to_numeric(pd.Series([row[m]])).iloc[0]
        if pd.isna(x) or pd.isna(mu) or pd.isna(sd) or sd < 1e-9:
            z = np.nan
        else:
            z = (float(x) - float(mu)) / float(sd)

        records.append({"Metric": m, "Value": x, "Cluster Mean": mu, "Z-Score": z})

    return pd.DataFrame(records).sort_values(by="Z-Score", key=lambda s: s.abs(), ascending=False)


def quick_business_take(
    df: pd.DataFrame,
    company_id: Any,
    id_col: str = "DUNS Number",
    cluster_col: str = "Cluster",
    metrics: Optional[List[str]] = None,
) -> List[str]:
    """
    Simple, readable insight bullets (good for report/demo).
    Uses deviation from cluster mean.
    """
    if metrics is None:
        metrics = [
            "Revenue per Employee",
            "IT Spend per Employee",
            "Tech Intensity Score",
            "Company Age",
        ]

    work = ensure_derived_metrics(df.copy())

    row_df = work.loc[work[id_col] == company_id]
    if row_df.empty:
        return [f"Company {company_id} not found."]
    row = row_df.iloc[0]

    if cluster_col not in work.columns:
        return ["Clustering not applied yet."]

    cl = row[cluster_col]
    cluster_df = work.loc[work[cluster_col] == cl]

    bullets = []
    for m in metrics:
        if m not in work.columns:
            continue

        x = _to_numeric(pd.Series([row[m]])).iloc[0]
        mu = _to_numeric(cluster_df[m]).mean(skipna=True)

        if pd.isna(x) or pd.isna(mu) or abs(mu) < 1e-9:
            continue

        ratio = float(x) / float(mu)
        if ratio >= 1.25:
            bullets.append(f"{m}: higher than cluster average (~{ratio:.1f}×).")
        elif ratio <= 0.75:
            bullets.append(f"{m}: lower than cluster average (~{ratio:.1f}×).")
        else:
            bullets.append(f"{m}: around cluster average.")

    if not bullets:
        bullets = ["Not enough data to generate a business take for this company."]

    return bullets
