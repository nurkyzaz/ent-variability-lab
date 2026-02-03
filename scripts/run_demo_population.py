from __future__ import annotations
from entlab.config import default_paths
from entlab.simulate import build_demo_population
from entlab.features import build_feature_table
from entlab.clustering import prepare_X, run_kmeans, pca_2d
from entlab.anomaly import score_isolation_forest
from entlab.plots import plot_clusters_2d, plot_top_anomalies

def main():
    p = default_paths()
    p.outputs.mkdir(exist_ok=True)
    p.figures.mkdir(parents=True, exist_ok=True)
    p.tables.mkdir(parents=True, exist_ok=True)

    df = build_demo_population(n_objects=600, n_flare=15, seed=42)
    ft = build_feature_table(df)

    Xs, _ = prepare_X(ft)
    labels = run_kmeans(Xs, k=4)
    Z = pca_2d(Xs)

    scores = score_isolation_forest(ft)
    scored = ft.copy()
    scored["cluster_kmeans"] = labels
    scored["anomaly_score"] = scores
    scored.to_csv(p.tables / "demo_scored.csv")

    ft.to_csv(p.tables / "demo_feature_table.csv")

    plot_clusters_2d(Z, labels, p.figures / "demo_clusters.png", title="Demo population clusters (PCA)")
    plot_top_anomalies(df, scores, p.figures / "demo_top_anomalies.png", top_k=12)

    print("Wrote:", p.tables / "demo_feature_table.csv")
    print("Wrote:", p.tables / "demo_scored.csv")
    print("Wrote:", p.figures / "demo_clusters.png")
    print("Wrote:", p.figures / "demo_top_anomalies.png")

if __name__ == "__main__":
    main()
