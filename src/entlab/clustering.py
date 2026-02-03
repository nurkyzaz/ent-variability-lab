from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

def prepare_X(feature_table: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    X = feature_table.copy()
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs_df = pd.DataFrame(Xs, index=X.index, columns=X.columns)
    return Xs_df, scaler

def run_kmeans(Xs: pd.DataFrame, k: int = 4, random_state: int = 42) -> pd.Series:
    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = model.fit_predict(Xs)
    return pd.Series(labels, index=Xs.index, name="cluster_kmeans")

def run_dbscan(Xs: pd.DataFrame, eps: float = 1.2, min_samples: int = 10) -> pd.Series:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(Xs)
    return pd.Series(labels, index=Xs.index, name="cluster_dbscan")

def pca_2d(Xs: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    return pd.DataFrame(Z, index=Xs.index, columns=["pc1", "pc2"])
