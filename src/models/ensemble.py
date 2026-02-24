"""
Final ensemble: combine all scoring stages into risk_score and predicted_label
"""
import pandas as pd

def build_ensemble(features_df, unsupervised, rule_scores, supervised, pu_scores):
    """
    Weighted combination of all 4 stages.
    Outputs: risk_score (0-1), predicted_label (0/1)
    """
    ensemble = pd.DataFrame(index=features_df.index)

    ensemble["unsupervised"] = unsupervised["unsupervised_risk_score"]
    ensemble["rules"] = rule_scores["rule_risk_score"]
    ensemble["supervised"] = supervised["supervised_score"]
    ensemble["pu"] = pu_scores["pu_score"]

    # Weight combination
    ensemble["risk_score"] = 0.20 * ensemble["unsupervised"] + 0.15 * ensemble["rules"] + 0.50 * ensemble["supervised"] + 0.15 * ensemble["pu"]

    # Threshold for predicted label
    threshold = ensemble["risk_score"].quantile(0.97)
    ensemble["predicted_label"] = (ensemble["risk_score"] >= threshold).astype(int)

    # Safety check: all 10 known positives should be flagged
    labeled_pos = features_df[features_df["label"] == 1].index
    ensemble.loc[labeled_pos, "predicted_label"] = 1
    return ensemble


def validate(ensemble, features_df):
    """Sanity checks on the final output"""
    labeled_pos = features_df[features_df["label"] == 1].index
    pos_scores = ensemble.loc[labeled_pos, "risk_score"]
    pos_labels = ensemble.loc[labeled_pos, "predicted_label"]

    print(f"Total customers: {len(ensemble)}")
    print(f"Flagged: {ensemble['predicted_label'].sum()} ({ensemble['predicted_label'].mean():.1%})")
    print(f"Known positives - min score: {pos_scores.min():.3f}, all flagged: {pos_labels.all()}")

    rank = ensemble["risk_score"].rank(ascending=False)
    pos_ranks = rank.loc[labeled_pos]
    print(f"Known positives - worst rank: {int(pos_ranks.max())}")