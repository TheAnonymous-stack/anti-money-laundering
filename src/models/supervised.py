"""
Stage 3: Semi-supervised XGBoost/LightGBM
Stage 4: PU Learning
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_supervised(features_df, anomaly_scores, rule_scores):
    """
    Train XGBoost + LightGBM on the 1,000 labeled customers
    Features = engineered features + anomaly scores + rule scores
    """

    # combine all feature sources
    drop_cols = ["label", "triggered_rules"]
    X_all = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
    X_all = X_all.select_dtypes(include=[np.number]).fillna(0) # keeping only numeric columns because non-numeric columns can't be fed to XGBoost/LightGBM
    X_all = X_all.join(anomaly_scores[["unsupervised_risk_score"]])
    X_all = X_all.join(rule_scores[["rule_risk_score", "rule_count"]])

    # Split labeled vs unlabeled
    labeled_mask = features_df["label"].notna()
    X_labeled = X_all[labeled_mask]
    y_labeled = features_df.loc[labeled_mask, "label"].astype(int)

    # Train XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        scale_pos_weight=99, # 990 negatives / 10 positives
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        eval_metric="logloss"
    )
    xgb.fit(X_labeled, y_labeled)

    # Train LightGBM
    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        scale_pos_weight=99,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_labeled, y_labeled)

    # Predict probabilities for ALL customers
    xgb_proba = xgb.predict_proba(X_all)[:,1]
    lgbm_proba = lgbm.predict_proba(X_all)[:,1]
    supervised_score = (xgb_proba + lgbm_proba) / 2 # take average to reduce variance from any single model
    
    result = pd.DataFrame({
        "supervised_score": supervised_score,
    }, index=features_df.index)

    return result, xgb, lgbm, X_all

def train_pu_learning(features_df, X_all, xgb, lgbm, n_iterations=2):
    """
    PU Learning: use initial classifiers to identify 'reliable negatives' among unlabeled customers, 
    then retrain with expanded negative set
    """

    labeled_mask = features_df["label"].notna()
    unlabeled_mask = ~labeled_mask

    # Initial predictions on unlabeled data
    xgb_pred = xgb.predict_proba(X_all[unlabeled_mask])[:,1]
    lgbm_pred = lgbm.predict_proba(X_all[unlabeled_mask])[:,1]
    avg_pred = (xgb_pred + lgbm_pred) / 2

    for iteration in range(n_iterations):
        # reliable negatives: unlabeled customers scored < 0.1
        reliable_neg_mask = avg_pred < 0.1
        reliable_neg_idx = X_all[unlabeled_mask].index[reliable_neg_mask]

        # expanded training setL original labeled + reliable negatives
        expanded_idx = labeled_mask | features_df.index.isin(reliable_neg_idx)
        X_expanded = X_all[expanded_idx]
        y_expanded = features_df.loc[expanded_idx, "label"].fillna(0).astype(int)

        # Retrain
        pos_count = y_expanded.sum()
        neg_count = len(y_expanded) - pos_count
        new_weight = neg_count / max(pos_count, 1)

        xgb_pu = XGBClassifier(
            n_estimators=300, max_depth=4,
            scale_pos_weight=new_weight,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.6, random_state=42,
            eval_metric="logloss"
        )
        xgb_pu.fit(X_expanded, y_expanded)

        lgbm_pu = LGBMClassifier(
            n_estimators=300, max_depth=4,
            scale_pos_weight=new_weight,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.6, random_state=42,
            verbose=-1
        )
        lgbm_pu.fit(X_expanded, y_expanded)


        # Upgrade predictions for next iteration
        xgb_pred = xgb_pu.predict_proba(X_all[unlabeled_mask])[:, 1]
        lgbm_pred = lgbm_pu.predict_proba(X_all[unlabeled_mask])[:,1]
        avg_pred = (xgb_pred + lgbm_pred) / 2

    # Final PU score for ALL customers
    pu_score = (xgb_pu.predict_proba(X_all)[:,1] + lgbm_pu.predict_proba(X_all)[:,1]) / 2
    return pd.DataFrame({"pu_score": pu_score}, index=features_df.index)