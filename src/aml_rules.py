"""
Domain-driven AML rule engine.
Each rule checks feature conditions from the Knowledge Library,
returns a binary trigger and a description.
The rule engine scores every customer and produces a rule_risk_score.
"""
import pandas as pd

def _rule_money_mule(row):
    """RF001, RF002: Young/student/unemployed with high EMT inflows"""
    if row["is_individual"] != 1:
        return False, ""
    is_young = row["is_minor"] == 1 or row["is_young_adult"] == 1
    is_vulnerable = row["is_student"] == 1 or row["is_unemployed"] == 1
    high_emt = row.get("emt_credit_count", 0) > 20
    if (is_young or is_vulnerable) and high_emt:
        return True, "Young or student/unemployed individual with unusually high incoming e-transfers"
    return False, ""


def _rule_pass_through(row):
    """RF003: Rapid pass-through with extreme credit ratio and multi-channel"""
    credit_ratio = row.get("overall_credit_ratio", 0.5)
    extreme_ratio = credit_ratio > 0.85 or credit_ratio < 0.15
    multi_channel = row.get("num_active_channels", 0) >= 4
    high_volume = row.get("total_transaction_count", 0) > 50
    if extreme_ratio and multi_channel and high_volume:
        return True, "Account shows rapid pass-through activity with extreme credit/debit imbalance across multiple channels"
    return False, ""


def _rule_structuring(row):
    """RF004, RF005, RF006: Near-10K transactions, round amounts, cross-channel structuring"""
    near_10k = row.get("total_near_10k", 0)
    round_ratio = row.get("overall_round_ratio", 0)
    channels = row.get("num_active_channels", 0)

    if near_10k > 3:
        return True, f"Multiple transactions ({int(near_10k)}) just below the $10,000 CAD reporting threshold"
    if near_10k > 1 and channels >= 3:
        return True, "Near-threshold transactions spread across multiple channels suggesting structured activity"
    if round_ratio > 0.5 and row.get("total_round_count", 0) > 10:
        return True, "Unusually high proportion of round-amount transactions suggesting deliberate structuring"
    return False, ""


def _rule_shell_company(row):
    """RF007, RF009: Holding company or zero-sales entity"""
    if row["is_small_business"] != 1:
        return False, ""
    is_holding = row.get("industry_is_holding", 0) == 1
    no_sales = row.get("zero_sales", 0) == 1 or row.get("sales_missing", 0) == 1
    no_employees = row.get("zero_employees", 0) == 1 or row.get("employee_count_missing", 0) == 1

    if is_holding and (no_sales or no_employees):
        return True, "Holding company with zero sales or no employees — consistent with shell entity indicators"
    if no_sales and row.get("total_amount_all", 0) > 10000:
        return True, "Business with zero declared sales yet significant transaction volume — possible pass-through entity"
    return False, ""


def _rule_new_business_high_activity(row):
    """RF008: Newly established business with immediate high volume"""
    if row["is_small_business"] != 1:
        return False, ""
    if row.get("is_newly_established", 0) == 1:
        high_count = row.get("total_transaction_count", 0) > 100
        high_amount = row.get("total_amount_all", 0) > 50000
        if high_count or high_amount:
            return True, "Newly established business with disproportionately high transaction volume"
    return False, ""


def _rule_layering(row):
    """RF010, RF011: Multi-channel diversification + burst activity"""
    channels = row.get("num_active_channels", 0)
    hhi = row.get("channel_hhi", 1)
    total_transactions = row.get("total_transaction_count", 0)

    if channels >= 5 and hhi < 0.3 and total_transactions > 50:
        return True, "Transaction activity spread across 5+ channels with low concentration — consistent with layering behaviour"

    for ch in ["abm", "card", "cheque", "eft", "emt", "westernunion", "wire"]:
        burst = row.get(f"{ch}_burst_score", 0)
        if burst > 10 and row.get(f"{ch}_transaction_count", 0) > 20:
            return True, f"Unusually high burst activity in {ch} channel suggesting coordinated fund movements"
    return False, ""


def _rule_wire_layering(row):
    """RF012: Large wire/WU inflows + multi-channel presence"""
    wire_credit = row.get("wire_total_credit_amount", 0)
    wu_credit = row.get("westernunion_total_credit_amount", 0)
    channels = row.get("num_active_channels", 0)

    if (wire_credit > 5000 or wu_credit > 5000) and channels >= 3:
        return True, "Large incoming wire or Western Union transfers combined with activity across multiple other channels"
    return False, ""


def _rule_income_mismatch(row):
    """RF013, RF016: Transaction or net flow grossly exceeding income"""
    # Skip if income is missing/zero — shell_company rule handles those
    if row.get("income_missing", 0) == 1:
        return False, ""

    transaction_ratio = row.get("transaction_to_income_ratio", 0)
    net_flow_ratio = row.get("net_flow_to_income_ratio", 0)

    if transaction_ratio > 100:
        return True, f"Total transaction volume exceeds declared income by {transaction_ratio:.0f}x"
    if net_flow_ratio > 100:
        return True, f"Net fund flow exceeds declared income by {net_flow_ratio:.0f}x"
    if transaction_ratio > 50:
        return True, f"Transaction volume exceeds declared income by {transaction_ratio:.0f}x"
    return False, ""


def _rule_minor_large_transfers(row):
    """RF014: Minor receiving large wire or EMT transfers"""
    if row["is_individual"] != 1:
        return False, ""
    if row.get("is_minor", 0) != 1:
        return False, ""
    wire_credit = row.get("wire_total_credit_amount", 0)
    emt_credit = row.get("emt_total_credit_amount", 0)

    if wire_credit > 1000 or emt_credit > 1000:
        return True, "Individual under 18 receiving large wire or e-transfer amounts inconsistent with age"
    return False, ""


def _rule_new_customer_high_activity(row):
    """RF015: Recently onboarded customer with high activity relative to income"""
    if row.get("is_new_customer", 0) != 1:
        return False, ""
    transaction_ratio = row.get("transaction_to_income_ratio", 0)
    if transaction_ratio > 5:
        return True, "Recently onboarded customer with transaction volume significantly exceeding declared income"
    return False, ""


def _rule_cash_intensive(row):
    """RF017, RF027: High ABM cash activity or EMT-to-cash pattern"""
    cash_count = row.get("abm_cash_count", 0)
    cash_ratio = row.get("abm_cash_ratio", 0)
    abm_amount = row.get("abm_total_amount", 0)

    if cash_count > 20 and abm_amount > 10000:
        return True, "Frequent large cash transactions at ABMs exceeding typical retail banking patterns"
    if cash_ratio > 0.5 and cash_count > 10:
        return True, "Majority of ABM transactions involve physical cash — consistent with cash-intensive laundering"

    emt_credit = row.get("emt_credit_count", 0)
    abm_debit = row.get("abm_debit_count", 0)
    if emt_credit > 20 and abm_debit > 10:
        return True, "High volume of incoming e-transfers followed by frequent ABM withdrawals"
    return False, ""


def _rule_foreign_abm(row):
    """RF018: Foreign ABM usage inconsistent with profile"""
    if row["is_individual"] != 1:
        return False, ""
    foreign_ratio = row.get("abm_foreign_ratio", 0)
    is_vulnerable = row.get("is_student", 0) == 1 or row.get("is_unemployed", 0) == 1

    if foreign_ratio > 0.3 and is_vulnerable:
        return True, "Student or unemployed individual with significant foreign ABM usage inconsistent with profile"
    return False, ""


def _rule_high_risk_industry(row):
    """RF019-RF022: Business in a high-risk industry"""
    if row["is_small_business"] != 1:
        return False, ""
    if row.get("industry_is_holding", 0) == 1:
        return True, "Business operates as a holding company — a structure frequently associated with concealed ownership"
    if row.get("industry_is_real_estate", 0) == 1:
        return True, "Business in real estate sector — identified as highly vulnerable to money laundering by the Cullen Commission"
    if row.get("industry_is_legal", 0) == 1:
        return True, "Legal services firm — FATF identifies legal professionals as commonly exploited to facilitate laundering"
    if row.get("industry_is_financial", 0) == 1 or row.get("industry_is_msb", 0) == 1:
        return True, "Financial services or MSB — commonly exploited by professional money launderers per FINTRAC"
    return False, ""


def _rule_card_suspicious(row):
    """RF029: High foreign/ecommerce card ratio with high amounts"""
    foreign = row.get("card_foreign_ratio", 0)
    ecommerce = row.get("card_ecommerce_ratio", 0)
    card_amount = row.get("card_total_amount", 0)

    if card_amount > 10000 and (foreign > 0.3 or ecommerce > 0.7):
        return True, "High-value card activity with elevated foreign or e-commerce transaction ratio"
    return False, ""


def _rule_night_weekend(row):
    """RF030: Disproportionate off-hours activity"""
    for ch in ["abm", "card", "cheque", "eft", "emt", "westernunion", "wire"]:
        count = row.get(f"{ch}_transaction_count", 0)
        if count < 5:
            continue
        weekend = row.get(f"{ch}_weekend_ratio", 0)
        night = row.get(f"{ch}_night_ratio", 0)
        if weekend > 0.7 or night > 0.5:
            return True, f"Disproportionate weekend or nighttime activity in {ch} channel suggesting avoidance of business-hour scrutiny"
    return False, ""


# Orchestrator
SEVERITY_WEIGHTS = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25}

def apply_rules(features_df):
    """
    Run all rules on the feature matrix.
    Returns a DataFrame with:
      - One column per rule (0/1 trigger)
      - 'triggered_rules': list of description strings
      - 'rule_count': number of rules triggered
      - 'rule_risk_score': weighted score normalized to [0, 1]
    """
    rules = [
        ("money_mule", _rule_money_mule, "HIGH"),
        ("pass_through", _rule_pass_through, "HIGH"),
        ("structuring", _rule_structuring, "HIGH"),
        ("shell_company", _rule_shell_company, "HIGH"),
        ("new_biz_high_activity", _rule_new_business_high_activity,"HIGH"),
        ("layering", _rule_layering, "HIGH"),
        ("wire_layering", _rule_wire_layering, "HIGH"),
        ("income_mismatch", _rule_income_mismatch, "HIGH"),
        ("minor_large_transfers", _rule_minor_large_transfers, "HIGH"),
        ("new_customer_high_activity", _rule_new_customer_high_activity, "MEDIUM"),
        ("cash_intensive", _rule_cash_intensive, "HIGH"),
        ("foreign_abm", _rule_foreign_abm, "MEDIUM"),
        ("high_risk_industry", _rule_high_risk_industry, "MEDIUM"),
        ("card_suspicious", _rule_card_suspicious, "MEDIUM"),
        ("night_weekend", _rule_night_weekend, "LOW"),
    ]

    results = pd.DataFrame(index=features_df.index)
    all_descriptions = []

    for rule_name, rule_fn, severity in rules:
        triggers = []
        descriptions = []
        for idx, row in features_df.iterrows():
            triggered, desc = rule_fn(row)
            triggers.append(int(triggered))
            descriptions.append(desc)
        results[f"rule_{rule_name}"] = triggers
        all_descriptions.append(descriptions)

    # Count triggered rules per customer
    rule_cols = [c for c in results.columns if c.startswith("rule_")]
    results["rule_count"] = results[rule_cols].sum(axis=1)

    # Weighted score: sum(triggered * severity_weight) / max_possible
    max_score = sum(SEVERITY_WEIGHTS[sev] for _, _, sev in rules)
    weighted = sum(
        results[f"rule_{name}"] * SEVERITY_WEIGHTS[sev]
        for name, _, sev in rules
    )
    results["rule_risk_score"] = (weighted / max_score).clip(0, 1)

    # Collect triggered descriptions per customer
    triggered_descs = []
    for i in range(len(features_df)):
        descs = [all_descriptions[j][i] for j in range(len(rules))
                 if all_descriptions[j][i] != ""]
        triggered_descs.append(descs)
    results["triggered_rules"] = triggered_descs

    return results
