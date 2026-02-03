from __future__ import annotations
import pandas as pd
from sklearn.ensemble import IsolationForest

def score_isolation_forest(
    feature_table: pd.DataFrame,
    random_state: int = 42
) -> pd.Series:
    """
    Returns anomaly score (higher = more anomalous)
    """
    X = feature_table.fillna(feature_table.median(numeric_only=True))
    model = IsolationForest(n_estimators=400, random_state=random_state, contamination="auto")
    model.fit(X)
    # decision_function higher = more normal; invert to make higher = more anomalous
    score = -model.decision_function(X)
    return pd.Series(score, index=feature_table.index, name="anomaly_score")
