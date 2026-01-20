from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px


# ----------------------------
# Day 2: Hard-coded feature list
# ----------------------------

SCALED_NUMERIC_COLS: List[str] = [
    "scaled_company_age",
    "scaled_log1p_revenue_usd",
    "scaled_log1p_employees_total",
    "scaled_revenue_per_employee",
    "scaled_it_spend_per_employee",
    "scaled_tech_intensity",
]

BINARY_DUMMY_COLS: List[str] = [
    "oh_Manufacturing Status_Yes",
    "oh_Ownership Type_Private",
    "oh_Ownership Type_Nonprofit",
]


# ----------------------------
# Artifacts container
# ----------------------------

@dataclass
class ClusterArtifacts:
    df_clustered: pd.DataFrame                 # original df + Cluster + ClusterName
    feature_names: List[str]                   # columns used to build X
    X: np.ndarray                              # numeric matrix used for KMeans/PCA
    elbow_inertia: Dict[int, float]            # k -> inertia
    k: int                                     # chosen k
    labels: np.ndarray                         # cluster labels
    centroids: np.ndarray                      # k x n_features
    cluster_counts: pd.Series                  # sizes
    pca_df: pd.DataFrame                       # PC1, PC2, Cluster, (optional hover cols)


# ----------------------------
# Feature building helpers
# ----------------------------

def one_hot_naics2_top_n(
    df: pd.DataFrame,
    col: str = "naics2",
    top_n: int = 6,
    other_label: str = "Other",
    prefix: str = "naics2",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode NAICS2 while limiting dimensionality:
    - keep only top_n most common categories
    - map everything else to other_label
    - returns (dummy_df, dummy_cols)
    """
    if col not in df.columns:
        empty = pd.DataFrame(index=df.index)
        return empty, []

    s = df[col].astype(str).fillna(other_label)
    s = s.replace({"nan": other_label, "None": other_label, "": other_label})

    vc = s.value_counts(dropna=False)
    top = vc.head(top_n).index.tolist()

    s_bucketed = s.where(s.isin(top), other_label)
    dummies = pd.get_dummies(s_bucketed, prefix=prefix, drop_first=False)

    # Ensure Other exists for stability
    other_col = f"{prefix}_{other_label}"
    if other_col not in dummies.columns:
        dummies[other_col] = 0

    dummy_cols = dummies.columns.tolist()
    return dummies, dummy_cols


def build_kmeans_matrix_day2(
    df_features: pd.DataFrame,
    *,
    top_n_naics2: int = 6,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Build the final numeric matrix X for KMeans using:
      - 6 scaled numeric cols (hard-coded)
      - 3 binary dummy cols (hard-coded)
      - NAICS2 one-hot limited to top N groups

    Returns:
      X: np.ndarray
      feature_names: list[str]
      X_df: pd.DataFrame (useful for debugging)
    """
    missing = [c for c in (SCALED_NUMERIC_COLS + BINARY_DUMMY_COLS) if c not in df_features.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns. "
            f"Expected these to exist from feature_pipeline: {missing}"
        )

    base = df_features[SCALED_NUMERIC_COLS + BINARY_DUMMY_COLS].copy()
    base = base.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    naics2_dummies, naics2_cols = one_hot_naics2_top_n(df_features, col="naics2", top_n=top_n_naics2)
    X_df = pd.concat([base, naics2_dummies], axis=1)

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)

    return X, feature_names, X_df


# ----------------------------
# Modeling: elbow + kmeans
# ----------------------------

def compute_elbow_inertia(
    X: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> Dict[int, float]:
    """
    Compute inertia for k in [k_min, k_max] for elbow method.
    Returns: {k: inertia}
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    inertias: Dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        km.fit(X)
        inertias[k] = float(km.inertia_)
    return inertias


def fit_kmeans(
    X: np.ndarray,
    *,
    k: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit KMeans for a chosen k.
    Returns: (labels, inertia, centroids)
    """
    X = np.asarray(X)
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    inertia = float(km.inertia_)
    centroids = km.cluster_centers_
    return labels, inertia, centroids


# ----------------------------
# PCA + plotting
# ----------------------------

def pca_2d(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    df_for_hover: Optional[pd.DataFrame] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute 2D PCA coordinates. Returns pca_df with columns:
    PC1, PC2, Cluster (+ optional hover columns if provided).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)

    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels.astype(str)

    # Add hover columns if available
    if df_for_hover is not None:
        for c in ["Company Name", "Company", "DUNS Number"]:
            if c in df_for_hover.columns:
                pca_df[c] = df_for_hover[c].astype(str)

    return pca_df


def plot_pca(
    pca_df: pd.DataFrame,
    *,
    title: str = "2D PCA projection of KMeans clusters",
) -> "px.Figure":
    """
    Build and return a Plotly PCA scatter plot.
    Expects columns: PC1, PC2, Cluster
    """
    required = {"PC1", "PC2", "Cluster"}
    missing = required - set(pca_df.columns)
    if missing:
        raise ValueError(f"pca_df missing columns: {sorted(missing)}")

    hover_cols = [c for c in ["Company Name", "Company", "DUNS Number"] if c in pca_df.columns]

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=hover_cols if hover_cols else None,
        opacity=0.75,
        title=title,
    )
    return fig

# ----------------------------
# Full Day 2 pipeline entry point
# ----------------------------

def run_day2_clustering(
    df_features: pd.DataFrame,
    *,
    top_n_naics2: int = 6,
    k_min: int = 2,
    k_max: int = 8,
    k_default: int = 5,
    chosen_k: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[ClusterArtifacts, "px.Figure"]:
    """
    End-to-end Day 2 clustering:
      1) Build X from hard-coded features + NAICS2 (top-N)
      2) Compute elbow inertias for k in [k_min, k_max]
      3) Fit KMeans with chosen_k if provided else k_default
      4) Add Cluster labels to df
      5) PCA 2D + Plotly figure
      6) Optional heuristic names (ClusterName)

    Returns: (ClusterArtifacts, plotly Figure)
    """
    X, feature_names, X_df = build_kmeans_matrix_day2(df_features, top_n_naics2=top_n_naics2)

    elbow = compute_elbow_inertia(X, k_min=k_min, k_max=k_max, random_state=random_state)

    k = int(chosen_k) if chosen_k is not None else int(k_default)
    labels, inertia, centroids = fit_kmeans(X, k=k, random_state=random_state)

    df_clustered = df_features.copy()
    df_clustered["Cluster"] = labels

    cluster_counts = pd.Series(labels).value_counts().sort_index()
    cluster_counts.index = cluster_counts.index.map(lambda i: f"Cluster {i}")
    cluster_counts.name = "count"

    pca_df = pca_2d(X, labels, df_for_hover=df_features, random_state=random_state)
    fig = plot_pca(pca_df)
    
    # insights 
    artifacts = ClusterArtifacts(
        df_clustered=df_clustered,
        feature_names=feature_names,
        X=X,
        elbow_inertia=elbow,
        k=k,
        labels=labels,
        centroids=centroids,
        cluster_counts=cluster_counts,
        pca_df=pca_df,
    )
    return artifacts, fig

# note cache clustering result requirement not yet satisfied