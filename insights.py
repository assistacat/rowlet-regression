# insights.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Configs
CURRENT_YEAR = 2026

DEFAULT_METRICS = [
    "Revenue (USD)",
    "Employees Total",
    "Company Age",
    "Revenue per Employee",
    "IT Spend per Employee",
    "Tech Intensity Score",
]

# Preferred IT spend columns in this dataset
IT_SPEND_CANDIDATES = ["IT Budget", "IT spend"]

# Signals for "tech intensity"
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


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric safely; non-parsable -> NaN."""
    return pd.to_numeric(series, errors="coerce")


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _minmax_norm(s: pd.Series) -> pd.Series:
    """Min-max normalize (0..1) robustly; constant -> 0.5; all NaN -> NaN."""
    s = _safe_numeric(s)
    if s.dropna().empty:
        return s
    mn = float(s.min(skipna=True))
    mx = float(s.max(skipna=True))
    if np.isclose(mx - mn, 0.0):
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index)
    return (s - mn) / (mx - mn)


def ensure_derived_metrics(
    df: pd.DataFrame,
    *,
    current_year: int = CURRENT_YEAR,
    employees_col: str = "Employees Total",
    revenue_col: str = "Revenue (USD)",
    year_found_col: str = "Year Found",
) -> pd.DataFrame:
    """
    Ensure the dataframe has derived columns used for insights:
    - Company Age
    - Revenue per Employee
    - IT Spend per Employee
    - Tech Intensity Score

    Returns a COPY of df with added/updated derived columns.
    """
    out = df.copy()

    # Company Age
    if _col_exists(out, year_found_col):
        yf = _safe_numeric(out[year_found_col])
        out["Company Age"] = (current_year - yf).clip(lower=0)
    else:
        if "Company Age" not in out.columns:
            out["Company Age"] = np.nan

    # Revenue per Employee
    if _col_exists(out, revenue_col) and _col_exists(out, employees_col):
        rev = _safe_numeric(out[revenue_col])
        emp = _safe_numeric(out[employees_col])
        out["Revenue per Employee"] = rev / (emp.fillna(0) + 1)
    else:
        if "Revenue per Employee" not in out.columns:
            out["Revenue per Employee"] = np.nan

    # IT Spend per Employee
    it_col = _first_existing_col(out, IT_SPEND_CANDIDATES)
    if it_col is not None and _col_exists(out, employees_col):
        it = _safe_numeric(out[it_col]).fillna(0)
        emp = _safe_numeric(out[employees_col]).fillna(0)
        out["IT Spend per Employee"] = it / (emp + 1)
    else:
        if "IT Spend per Employee" not in out.columns:
            out["IT Spend per Employee"] = np.nan

    # Tech Intensity Score: average of available binary signals, normalized to 0..1
    available_signals = [c for c in TECH_SIGNAL_COLS if c in out.columns]
    if available_signals:
        binary_signals = []
        for c in available_signals:
            col = _safe_numeric(out[c]).fillna(0)
            binary_signals.append((col > 0).astype(int))
        # Average of binary flags -> 0..1
        out["Tech Intensity Score"] = pd.concat(binary_signals, axis=1).mean(axis=1)
    else:
        if "Tech Intensity Score" not in out.columns:
            out["Tech Intensity Score"] = np.nan

    return out


def build_cluster_profile(
    df: pd.DataFrame,
    *,
    cluster_col: str = "Cluster",
    metrics: Optional[List[str]] = None,
    extra_group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create a cluster profile summary table:
    - Count
    - Mean + Median of selected metrics
    Optionally adds a simple mode/top value for extra_group_cols (e.g., Country, NAICS Description).
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    if cluster_col not in df.columns:
        return pd.DataFrame({"error": [f"Missing required column: {cluster_col}"]})

    # Ensure derived metrics exist
    df2 = ensure_derived_metrics(df)

    use_metrics = [m for m in metrics if m in df2.columns]
    if not use_metrics:
        return pd.DataFrame({"error": ["No requested metrics exist in the dataframe."]})

    # Build aggregation
    agg: Dict[str, Any] = {}
    for m in use_metrics:
        agg[m] = ["mean", "median"]

    prof = df2.groupby(cluster_col).agg(agg)
    # Flatten multiindex columns
    prof.columns = [f"{col}_{stat}" for col, stat in prof.columns]
    prof.insert(0, "Cluster Size", df2.groupby(cluster_col).size())

    # Optional: add top categories for context (mode)
    if extra_group_cols:
        for col in extra_group_cols:
            if col in df2.columns:
                top = (
                    df2.groupby(cluster_col)[col]
                    .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan)
                )
                prof[f"Top {col}"] = top

    prof = prof.reset_index().sort_values("Cluster Size", ascending=False)
    return prof


def make_radar_fig(
    df: pd.DataFrame,
    *,
    company_id: Any,
    id_col: str = "DUNS Number",
    cluster_col: str = "Cluster",
    metrics: Optional[List[str]] = None,
    company_name_col: Optional[str] = None,
) -> go.Figure:
    """
    Plotly radar chart comparing a selected company vs its cluster average.
    Normalizes each metric within the (filtered) dataframe to 0..1 so the radar is interpretable.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    if id_col not in df.columns:
        return go.Figure().update_layout(title=f"Missing required column: {id_col}")

    if cluster_col not in df.columns:
        return go.Figure().update_layout(title=f"Missing required column: {cluster_col}")

    df2 = ensure_derived_metrics(df)
    use_metrics = [m for m in metrics if m in df2.columns]

    if not use_metrics:
        return go.Figure().update_layout(title="No radar metrics available in dataframe.")

    sel = df2[df2[id_col] == company_id]
    if sel.empty:
        return go.Figure().update_layout(title=f"Company not found for {id_col}={company_id}")

    sel_row = sel.iloc[0]
    cluster_val = sel_row[cluster_col]

    peer = df2[df2[cluster_col] == cluster_val]
    if peer.empty:
        return go.Figure().update_layout(title=f"No peers found for cluster={cluster_val}")

    # Normalize metrics across df2 (or you can switch to peer-only normalization if you want)
    norm_cols = {}
    for m in use_metrics:
        norm_cols[m] = _minmax_norm(df2[m])

    # Pull normalized values for company and cluster mean
    company_vals = []
    cluster_vals = []
    labels = []
    for m in use_metrics:
        labels.append(m)
        company_vals.append(float(norm_cols[m].loc[sel_row.name]) if pd.notna(norm_cols[m].loc[sel_row.name]) else np.nan)
        cluster_mean = peer[m].astype("float64", errors="ignore")
        cluster_mean_val = float(np.nanmean(_safe_numeric(cluster_mean)))
        # Convert cluster mean to normalized scale using overall min/max
        # (by normalizing series and then taking mean of normalized values in cluster)
        cluster_vals.append(float(np.nanmean(norm_cols[m].loc[peer.index])))

    # Radar needs closed loop
    labels_closed = labels + [labels[0]]
    company_closed = company_vals + [company_vals[0]]
    cluster_closed = cluster_vals + [cluster_vals[0]]

    # Title label
    if company_name_col and company_name_col in df2.columns:
        name = str(sel_row.get(company_name_col, company_id))
    else:
        name = str(company_id)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=cluster_closed,
            theta=labels_closed,
            fill="toself",
            name=f"Cluster {cluster_val} average",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=company_closed,
            theta=labels_closed,
            fill="toself",
            name=f"Company {name}",
        )
    )

    fig.update_layout(
        title=f"Company vs Cluster {cluster_val} (Normalized 0–1)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
    )
    return fig


def compute_anomaly_score(
    df: pd.DataFrame,
    *,
    company_id: Any,
    id_col: str = "DUNS Number",
    cluster_col: str = "Cluster",
    metrics: Optional[List[str]] = None,
    z_thresh: float = 2.0,
) -> Tuple[float, List[str]]:
    """
    Simple anomaly score within the company’s cluster using z-scores.
    Returns (score, flags) where:
    - score = sum of |z| beyond threshold (clipped), across metrics
    - flags = list of human-readable anomaly strings
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    if id_col not in df.columns or cluster_col not in df.columns:
        return 0.0, ["Missing required id/cluster columns."]

    df2 = ensure_derived_metrics(df)

    sel = df2[df2[id_col] == company_id]
    if sel.empty:
        return 0.0, [f"Company not found for {id_col}={company_id}"]

    sel_row = sel.iloc[0]
    cluster_val = sel_row[cluster_col]
    peer = df2[df2[cluster_col] == cluster_val]

    use_metrics = [m for m in metrics if m in df2.columns]
    if not use_metrics or peer.empty:
        return 0.0, ["No metrics or peers available for anomaly scoring."]

    score = 0.0
    flags: List[str] = []

    for m in use_metrics:
        peer_vals = _safe_numeric(peer[m])
        mu = float(peer_vals.mean(skipna=True)) if not peer_vals.dropna().empty else np.nan
        sd = float(peer_vals.std(skipna=True)) if not peer_vals.dropna().empty else np.nan
        x = _safe_numeric(pd.Series([sel_row.get(m)])).iloc[0]

        if np.isnan(mu) or np.isnan(sd) or np.isclose(sd, 0.0) or np.isnan(x):
            continue

        z = (x - mu) / sd
        if abs(z) >= z_thresh:
            direction = "above" if z > 0 else "below"
            flags.append(f"{m}: {direction} cluster avg (z={z:.2f})")
            score += min(abs(float(z)), 5.0)  # clip to avoid one metric dominating

    return float(score), flags


def quick_business_take(
    df: pd.DataFrame,
    *,
    company_id: Any,
    id_col: str = "DUNS Number",
    cluster_col: str = "Cluster",
) -> str:
    """
    A lightweight, non-LLM insight blurb (useful as fallback or for debugging).
    """
    df2 = ensure_derived_metrics(df)
    sel = df2[df2[id_col] == company_id]
    if sel.empty or cluster_col not in df2.columns:
        return "No insight available (company or cluster not found)."

    row = sel.iloc[0]
    cluster_val = row.get(cluster_col, "N/A")

    rev_pe = row.get("Revenue per Employee", np.nan)
    it_pe = row.get("IT Spend per Employee", np.nan)
    tech = row.get("Tech Intensity Score", np.nan)

    parts = [f"Cluster {cluster_val}."]
    if pd.notna(rev_pe):
        parts.append(f"Revenue/Employee ≈ {rev_pe:,.0f}.")
    if pd.notna(it_pe):
        parts.append(f"IT Spend/Employee ≈ {it_pe:,.0f}.")
    if pd.notna(tech):
        parts.append(f"Tech Intensity ≈ {tech:.2f} (0–1).")

    return " ".join(parts)
