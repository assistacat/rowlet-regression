from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px   # plotly already in requirements.txt

@dataclass
class ClusterArtifacts:
    df_clustered: pd.DataFrame
    feature_cols: Optional[List[str]]
    X_scaled: np.ndarray
    cluster_counts: pd.Series
    pca_df: pd.DataFrame


def fit_kmeans(
    df_clean: pd.DataFrame,
    X_scaled: np.ndarray,
    *,
    feature_cols: Optional[List[str]] = None, # list of columns name if necessary
    n_clusters: int = 5,
    random_state: int = 42,
) -> ClusterArtifacts:
    X_scaled = np.asarray(X_scaled) # numpy array conversion
    if X_scaled.ndim != 2:
        raise ValueError(f"X_scaled must be 2D, got shape {X_scaled.shape}")

    # number of rows match check, clusters matched to correct companies
    if len(df_clean) != X_scaled.shape[0]:
        raise ValueError(
            f"Row mismatch: df_clean has {len(df_clean)} rows but X_scaled has {X_scaled.shape[0]}"
        )

    #Fit kmeans with clusters=5
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X_scaled)

    df_clustered = df_clean.copy()
    df_clustered["Cluster"] = labels

    cluster_counts = pd.Series(labels).value_counts().sort_index()
    cluster_counts.index = cluster_counts.index.map(lambda k: f"Cluster {k}")
    cluster_counts.name = "count"

    # PCA to 2D for visualisation
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels.astype(str)

    return ClusterArtifacts(
        df_clustered=df_clustered,
        feature_cols=feature_cols,
        X_scaled=X_scaled,
        cluster_counts=cluster_counts,
        pca_df=pca_df,
    )


def plot_pca(
    pca_df: pd.DataFrame,
    *,
    title: str = "2D PCA projection of KMeans clusters",
):
    """
    Build and return a Plotly PCA scatter plot.
    Expects columns: PC1, PC2, Cluster
    """
    required = {"PC1", "PC2", "Cluster"}
    missing = required - set(pca_df.columns)
    if missing:
        raise ValueError(f"pca_df missing columns: {sorted(missing)}")

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        opacity=0.75,
        title=title,
    )
    return fig

# call this function for clustering + plot
def cluster_and_plot(
    df_clean: pd.DataFrame,
    X_scaled: np.ndarray,
    *,
    feature_cols: Optional[List[str]] = None,
    n_clusters: int = 5,
    random_state: int = 42,
):
    """
    Convenience wrapper:
    - runs KMeans + PCA
    - returns (ClusterArtifacts, Plotly Figure)
    """
    art = fit_kmeans(
        df_clean,
        X_scaled,
        feature_cols=feature_cols,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    fig = plot_pca(art.pca_df)
    return art, fig
