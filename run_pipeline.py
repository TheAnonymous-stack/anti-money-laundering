"""End-to-end pipeline: features -> models -> outputs"""
import os
import pandas as pd
from src.feature_engineering import build_feature_matrix
from src.aml_rules import apply_rules
from src.models.anomaly_detectors import train_anomaly_detectors
from src.models.supervised import train_supervised, train_pu_learning
from src.models.ensemble import build_ensemble, validate
from src.explanations import generate_all_explanations

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Step 1: Feature engineering
    print("-" * 50)
    print("Step 1: Building feature matrix...")
    features = build_feature_matrix()
    print(f"  Feature matrix: {features.shape[0]} customers x {features.shape[1]} columns")

    # Step 2: AML rule engine
    print("-" * 50)
    print("Step 2: Applying AML rules...")
    rule_results = apply_rules(features)
    triggered_count = (rule_results["rule_count"] > 0).sum()
    print(f"  Customers with 1+ rules triggered: {triggered_count}")

    # Step 3: Unsupervised anomaly detection
    print("-" * 50)
    print("Step 3: Training anomaly detectors...")
    anomaly_scores = train_anomaly_detectors(features)
    top_100 = anomaly_scores["unsupervised_risk_score"].nlargest(100)
    print(f"  Top 100 anomaly scores: min={top_100.min():.3f}, max={top_100.max():.3f}")

    # Step 4: Semi-supervised learning
    print("-" * 50)
    print("Step 4: Training supervised models...")
    supervised_scores, xgb, lgbm, X_all = train_supervised(features, anomaly_scores, rule_results)

    # Step 5: PU Learning
    print("-" * 50)
    print("Step 5: PU Learning...")
    pu_scores = train_pu_learning(features, X_all, xgb, lgbm)

    # Step 6: Ensemble
    print("-" * 50)
    print("Step 6: Building ensemble...")
    ensemble = build_ensemble(features, anomaly_scores, rule_results, supervised_scores, pu_scores)

    # Step 7: Validation
    print("-" * 50)
    print("Step 7: Validation...")
    validate(ensemble, features)

    # Step 8: Generate explanations
    print("-" * 50)
    print("Step 8: Generating explanations...")
    explanations_df = generate_all_explanations(features, ensemble, rule_results, xgb, X_all)

    # Step 9: Save outputs
    print("-" * 50)
    print("Step 9: Saving outputs...")

    model_output = pd.DataFrame({
        "customer_id": ensemble.index,
        "predicted_label": ensemble["predicted_label"].astype(int),
        "risk_score": ensemble["risk_score"].round(6),
    })
    model_output.to_csv(os.path.join(OUTPUT_DIR, "model_output.csv"), index=False)
    print(f"  Saved model_output.csv ({len(model_output)} rows)")

    explanations_df.to_csv(os.path.join(OUTPUT_DIR, "model_output_explanations.csv"), index=False)
    print(f"  Saved model_output_explanations.csv ({len(explanations_df)} rows)")

    # Summary
    print("-" * 50)
    print("DONE")
    print(f"  Total customers: {len(model_output)}")
    print(f"  Flagged as bad actors: {model_output['predicted_label'].sum()}")
    print(f"  Flagging rate: {model_output['predicted_label'].mean():.1%}")


if __name__ == "__main__":
    main()
