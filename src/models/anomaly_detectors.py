"""
Stage 1: Unsupervised anomaly detection using pyod
- Takes the feature matrix
- Standardizes features
- Train 4 detectors: Isolation Forest, ECOD, LOF, kNN
- Each produces a raw anomaly score per customer
- Normalizes each to [0, 1] via min-max scaling
- Averages into a single unsupervised_risk_score
"""
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler

def train_anomaly_detectors(features_df):
    """
    Train multiple anomaly detectors, return averaged score
    """

    # Drop non-numeric and label columns
    drop_cols = ["label", "triggered_rules"]
    X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    detectors = {
        "iforest": IForest(n_estimators=200, contamination=0.02, random_state=42),
        "ecod": ECOD(contamination=0.02),
        "lof": LOF(n_neighbors=20, contamination=0.02),
        "knn": KNN(n_neighbors=20, contamination=0.02)
    }

    scores = pd.DataFrame(index=features_df.index)
    for name, model in detectors.items():
        model.fit(X_scaled)
        raw = model.decision_scores_ # raw anomaly scores
        scores[name] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-10) # min-max normalize to [0,1]
    scores["unsupervised_risk_score"] = scores.mean(axis=1)
    return scores
