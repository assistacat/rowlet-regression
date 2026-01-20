from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@dataclass
class ClusterArtifacts:
    df_clustered: pd.DataFrame
    feature_cols: Optional[List[str]]     
    X_scaled: np.ndarray
    cluster_counts: pd.Series
    pca_df: pd.DataFrame


def run_kmeans_pca(
    df_clean: pd.DataFrame,
    X_scaled: np.ndarray,
    *, # passed as keyword arg
    feature_cols: Optional[List[str]] = None, #optional list of feature names from person 2
    n_clusters: int = 5,
    random_state: int = 42,
) -> ClusterArtifacts:
    # Safety checks
    X_scaled = np.asarray(X_scaled)
    if X_scaled.ndim != 2:
        raise ValueError(f"X_scaled must be 2D, got shape {X_scaled.shape}")

    if len(df_clean) != X_scaled.shape[0]:
        raise ValueError(
            f"Row mismatch: df_clean has {len(df_clean)} rows but X_scaled has {X_scaled.shape[0]}"
        )

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X_scaled)

    df_clustered = df_clean.copy()
    df_clustered["Cluster"] = labels

    cluster_counts = pd.Series(labels).value_counts().sort_index()
    cluster_counts.index = cluster_counts.index.map(lambda k: f"Cluster {k}")
    cluster_counts.name = "count"

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