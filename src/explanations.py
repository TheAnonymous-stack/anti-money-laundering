"""Generate explanations for every customer's risk assessment"""
import numpy as np
import pandas as pd
import shap


# Translate raw feature names into readable phrases
FEATURE_DESCRIPTIONS = {
    # Cross-channel
    "transaction_to_income_ratio": "transaction volume relative to declared income",
    "net_flow_to_income_ratio": "net fund flow relative to declared income",
    "total_amount_all": "total transaction volume across all channels",
    "total_transaction_count": "total number of transactions across all channels",
    "total_net_flow": "net flow of funds across all channels",
    "num_active_channels": "number of active banking channels",
    "channel_hhi": "concentration of activity across channels",
    "overall_credit_ratio": "proportion of incoming vs outgoing funds",
    "total_near_10k": "number of transactions near the $10,000 reporting threshold",
    "overall_near_10k_ratio": "proportion of near-threshold transactions",
    "total_round_count": "number of round-amount transactions",
    "overall_round_ratio": "proportion of round-amount transactions",
    "total_credit_all": "total incoming funds across all channels",
    "total_debit_all": "total outgoing funds across all channels",
    # Scores from other stages
    "unsupervised_risk_score": "statistical anomaly score",
    "rule_risk_score": "domain rule-based risk assessment",
    "rule_count": "number of AML rules triggered",
    # KYC individual
    "age": "customer age",
    "is_minor": "minor (under 18) status",
    "is_young_adult": "young adult (18-24) status",
    "is_student": "student occupation status",
    "is_unemployed": "unemployed status",
    "is_retired": "retired status",
    "income": "declared income",
    "log_income": "declared income level",
    "income_missing": "missing income declaration",
    "tenure_days": "length of customer relationship",
    "is_new_customer": "recently onboarded customer",
    # KYC business
    "industry_is_holding": "holding company classification",
    "industry_is_real_estate": "real estate industry classification",
    "industry_is_legal": "legal services industry classification",
    "industry_is_financial": "financial services industry classification",
    "industry_is_msb": "money services business classification",
    "is_newly_established": "recently established business",
    "zero_sales": "zero declared sales",
    "sales_missing": "missing sales declaration",
    "zero_employees": "zero reported employees",
    "employee_count_missing": "missing employee count",
    "sales_per_employee": "sales per employee ratio",
    "business_age_days": "age of business",
    # EMT
    "emt_credit_count": "number of incoming e-transfers received",
    "emt_transaction_count": "total e-transfer activity",
    "emt_total_credit_amount": "total incoming e-transfer volume",
    "emt_total_amount": "total e-transfer volume",
    # Wire
    "wire_total_credit_amount": "incoming wire transfer volume",
    "wire_transaction_count": "wire transfer activity",
    "wire_total_amount": "total wire transfer volume",
    # Western Union
    "westernunion_total_credit_amount": "incoming Western Union transfer volume",
    "westernunion_transaction_count": "Western Union activity",
    # ABM
    "abm_cash_count": "number of cash transactions at ABMs",
    "abm_cash_ratio": "proportion of ABM transactions involving cash",
    "abm_foreign_ratio": "proportion of ABM transactions at foreign locations",
    "abm_total_amount": "total ABM transaction volume",
    "abm_transaction_count": "total ABM activity",
    # Card
    "card_foreign_ratio": "proportion of foreign card transactions",
    "card_ecommerce_ratio": "proportion of e-commerce card transactions",
    "card_total_amount": "total card transaction volume",
    "card_transaction_count": "total card activity",
    # EFT
    "eft_total_amount": "total electronic funds transfer volume",
    "eft_transaction_count": "electronic funds transfer activity",
}


def get_shap_explanations(xgb_model, X_all, top_n=5):
    """
    Compute SHAP values from XGBoost and return
    top_n contributing features per customer.
    """
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_all)

    top_features = []
    for i in range(len(X_all)):
        abs_shap = np.abs(shap_values[i])
        top_idx = np.argsort(abs_shap)[-top_n:][::-1]
        feats = []
        for idx in top_idx:
            fname = X_all.columns[idx]
            feats.append((fname, shap_values[i][idx], X_all.iloc[i, idx]))
        top_features.append(feats)

    return top_features


def generate_explanation(row, risk_score, triggered_rules, top_shap_features):
    """
    Generate a single customer's explanation string.
    High-risk and low-risk customers get different templates.
    """
    is_business = row.get("is_small_business", 0) == 1
    ctype = "small business" if is_business else "individual"

    if risk_score >= 0.5:
        lines = [f"This {ctype} customer has been assigned an elevated risk score of {risk_score:.2f}."]

        if triggered_rules:
            lines.append("The following AML indicators were identified:")
            for desc in triggered_rules:
                lines.append(f"- {desc}")

        if top_shap_features:
            lines.append("Key contributing factors from model analysis:")
            for fname, shap_val, fval in top_shap_features:
                readable = FEATURE_DESCRIPTIONS.get(fname, fname.replace("_", " "))
                direction = "elevated" if shap_val > 0 else "reduced"
                lines.append(f"- {readable.capitalize()} ({direction} risk)")

        lines.append(
            "These patterns are consistent with indicators identified by "
            "FINTRAC and FATF for potential money laundering activity."
        )

    elif risk_score >= 0.2:
        lines = [f"This {ctype} customer has been assigned a moderate risk score of {risk_score:.2f}."]

        if triggered_rules:
            lines.append("The following indicators were noted:")
            for desc in triggered_rules:
                lines.append(f"- {desc}")
        else:
            lines.append("While no specific AML rules were triggered, the customer's statistical profile shows some deviation from typical patterns.")

        if top_shap_features:
            fname, shap_val, fval = top_shap_features[0]
            readable = FEATURE_DESCRIPTIONS.get(fname, fname.replace("_", " "))
            lines.append(f"Primary contributing factor: {readable}.")

        lines.append("Continued monitoring is recommended.")

    else:
        lines = [f"This {ctype} customer has been assessed as low risk (score: {risk_score:.2f})."]
        lines.append("Transaction activity is consistent with the customer's profile.")

        if top_shap_features:
            fname, shap_val, fval = top_shap_features[0]
            readable = FEATURE_DESCRIPTIONS.get(fname, fname.replace("_", " "))
            lines.append(f"Primary assessment factor: {readable}.")

        lines.append("No significant AML red flags were identified during this assessment period.")

    explanation = " ".join(lines)
    return explanation[:2000]


def generate_all_explanations(features_df, ensemble, rule_results, xgb_model, X_all):
    """
    Generate explanations for all 61,410 customers.
    Returns a DataFrame with customer_id and explanation.
    """
    top_shap = get_shap_explanations(xgb_model, X_all, top_n=5)

    explanations = []
    for i, (idx, row) in enumerate(features_df.iterrows()):
        risk_score = ensemble.loc[idx, "risk_score"]
        triggered = rule_results.loc[idx, "triggered_rules"]
        shap_feats = top_shap[i]
        explanation = generate_explanation(row, risk_score, triggered, shap_feats)
        explanations.append({"customer_id": idx, "explanation": explanation})

    return pd.DataFrame(explanations)
