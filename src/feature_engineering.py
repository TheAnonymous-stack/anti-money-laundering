"""Transform raw CSVs into a customer-level feature matrix"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from src.data_loader import (load_kyc_individual, load_kyc_smallbusiness, load_labels, load_transactions, CHANNELS, DATA_DIR)

REF_DATE = pd.Timestamp("2025-01-31") # Reference date: end of the trasaction data window

def build_individual_features(df):
    """
    Derive features from individual KYC profiles.
    Key AML insight: the positives included studetns with $8 income receiving thousands in wire transfers
    and an unemployed person with $112k income.
    The individual features should capture such mismatches
    """

    feature = pd.DataFrame(index=df["customer_id"])
    feature["is_individual"] = 1
    feature["is_small_business"] = 0

    # Business-specific features don't apply so set to sentinel -1
    # Tree models will learn to ignore these when is_individual = 1
    business_features = ["business_age_days", "is_newly_established", "industry_is_holding", "industry_is_real_estate", "industry_is_legal", "industry_is_financial", "industry_is_msb", "industry_is_other", "employee_count", "employee_count_missing", "zero_employees", "sales", "sales_missing", "zero_sales", "log_sales", "sales_per_employee"]
    for col in business_features:
        feature[col] = -1

    feature["age"] = (REF_DATE - df["birth_date"]).dt.days.values / 365
    feature["is_minor"] = (feature["age"] < 18).astype(int)
    feature["is_young_adult"] = ((feature["age"] >= 18) & (feature["age"] < 25)).astype(int)

    # Tenure: how long the individual has been a customer
    feature["tenure_days"] = (REF_DATE - df["onboard_date"]).dt.days.values
    feature["is_new_customer"] = (feature["tenure_days"] < 365).astype(int)

    # Occupation flags: occupation not listed in the kyc_occupation_codes.csv
    occupation = df["occupation_code"].fillna("UNKNOWN").values
    feature["is_student"] = (occupation == "STUDENT").astype(int)
    feature["is_unemployed"] = (occupation == "UNEMPLOYED").astype(int)
    feature["is_retired"] = (occupation == "RETIRED").astype(int)
    feature["is_self_employed"] = (occupation == "SELF_EMPLOYED").astype(int)
    feature["is_other_occupation"] = (occupation == "OTHER").astype(int)
    feature["occupation_missing"] = (occupation == "UNKNOWN").astype(int)

    special_occupations = ["STUDENT", "UNEMPLOYED", "RETIRED", "SELF_EMPLOYED", "OTHER", "UNKNOWN"]
    feature["has_coded_occupation"] = (~np.isin(occupation, special_occupations)).astype(int)

    # Income
    feature["income"] = df["income"].values
    feature["income_missing"] = feature["income"].isna().astype(int)
    feature["income"] = feature["income"].fillna(0)
    feature["log_income"] = np.log1p(feature["income"]) # log-transform income to handle the wide range of income

    # Gender encoding
    gender_map = {"MALE": 0, "FEMALE": 1}
    feature["gender"] = df["gender"].map(gender_map).values
    feature["gender_missing"] = feature["gender"].isna().astype(int)

    # Marital status
    marital_map = {"Single": 0, "Married": 1, "Widowed": 2, "Divorced": 3}
    feature["marital_status"] = df["marital_status"].map(marital_map).values
    feature["marital_missing"] = feature["marital_status"].isna().astype(int)

    # Province (frequency encoding: common provinces get lower values)
    province_freq = df["province"].value_counts(normalize=True)
    feature["province_freq"] = df["province"].map(province_freq).values
    feature["province_missing"] = feature["province_freq"].isna().astype(int)

    # City is "other": privacy-masking flag indicating smaller/rural location
    feature["city_is_other"] = (df["city"].str.lower() == "other").astype(int).values

    # Fill NaN features with -1, a sentinel value tree models can split on
    feature = feature.fillna(-1)
    return feature

def build_business_features(df):
    """
    Derive features from small business KYC profiles.
    Key AML insight: positives included holding companies, a brand-new law firm with $0 sales, and real estate operators
    """

    feature = pd.DataFrame(index=df["customer_id"])
    feature["is_individual"] = 0
    feature["is_small_business"] = 1

    # Individual-specific features don't apply so set to sentinel -1
    # Tree models will learn to ignore these when is_small_business = 1
    individual_features = ["age", "is_minor", "is_young_adult", "is_student", "is_unemployed", "is_retired", "is_self_employed", "is_other_occupation", "occupation_missing", "has_coded_occupation", "gender", "gender_missing", "marital_status", "marital_missing"]
    for col in individual_features:
        feature[col] = -1
    
    # Tenure
    feature["tenure_days"] = (REF_DATE - df["onboard_date"]).dt.days.values
    feature["is_new_customer"] = (feature["tenure_days"] < 365).astype(int)

    # Business age: how long the business has existed
    # A brand-new business with high transaction volume is sus
    business_age = (REF_DATE - df["established_date"]).dt.days.values
    feature["business_age_days"] = business_age
    feature["is_newly_established"] = (business_age < 180).astype(int)

    # high-risk industry flags
    # these industries are commonly exploited for money laundering per FINTRAC/FATF
    code = df["industry_code"].astype(str).values
    feature["industry_is_holding"] = np.isin(code, ["7215"]).astype(int) # Shell companies
    feature["industry_is_real_estate"] = np.isin(code, ["7511", "7512", "7599", "4491"]).astype(int)
    feature["industry_is_legal"] = np.isin(code, ["7761"]).astype(int)
    feature["industry_is_financial"] = np.isin(code, ["7214", "7292", "7421", "7499"]).astype(int)
    feature["industry_is_msb"] = np.isin(code, ["4842"]).astype(int) # courier service
    feature["industry_is_other"] = np.isin(code, ["Other"]).astype(int)

    # Employee count
    feature["employee_count"] = df["employee_count"].values
    feature["employee_count_missing"] = feature["employee_count"].isna().astype(int)
    feature["zero_employees"] = (df["employee_count"].fillna(-1) == 0).astype(int).values # distinguish between 0 employees and NaN value

    # Sales
    feature["sales"] = df["sales"].values
    feature["sales_missing"] = feature["sales"].isna().astype(int)
    feature["zero_sales"] = (df["sales"].fillna(-1) == 0).astype(int).values
    feature["log_sales"] = np.log1p(df["sales"].fillna(0).values)

    # Sales/employee: shell companies often have extreme ratios
    employee_count = df["employee_count"].fillna(0).values
    sales_vals = df["sales"].fillna(0).values
    feature["sales_per_employee"] = np.where(employee_count > 0, sales_vals / employee_count, 0)

    # Use sales as the business income so that cross-channel features such as transaction to income ratio work for both individual and business
    feature["income"] = sales_vals
    feature["income_missing"] = df["sales"].isna().astype(int).values
    feature["log_income"] = np.log1p(feature["income"])

    # Province
    province_freq = df["province"].value_counts(normalize=True)
    feature["province_freq"] = df["province"].map(province_freq).values
    feature["province_missing"] = df["province"].isna().astype(int).values

    # City
    feature["city_is_other"] = (df["city"].str.lower() == "other").astype(int).values

    feature = feature.fillna(-1)
    return feature

def _agg_single_df(df, channel):
    """
    Core aggregation: takes a transaction DataFrame, returns per-customer stats
    Works for any channel. Channel-specific extras are added at the end
    """
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")

    df["is_credit"] = (df["debit_credit"] == "C").astype(int) # Credit means money coming in account
    df["is_debit"] = (df["debit_credit"] == "D").astype(int) # Debit means money coming out of account

    # AML structuring: splitting transactions to avoid $10K reporting threshold
    df["is_round_100"] = (df["amount_cad"] % 100 == 0).astype(int)
    df["is_near_10k"] = ((df["amount_cad"] >= 9000) & (df["amount_cad"] < 10000)).astype(int)

    # Temporal features (some channels such as cheque/wire only have dates, no times)
    has_time = df["transaction_datetime"].notna()
    df["is_weekend"] = 0
    df["is_night"] = 0
    if has_time.any():
        df.loc[has_time, "is_weekend"] = (df.loc[has_time, "transaction_datetime"].dt.dayofweek.isin([5, 6]).astype(int)) # Saturday and Sunday are encoded as 5 and 6 respectively
        df.loc[has_time, "is_night"] = (df.loc[has_time, "transaction_datetime"].dt.hour.isin(list(range(0, 6)) + [22, 23]).astype(int)) # 10 PM to 5 AM is considered night
    
    group = df.groupby("customer_id")
    agg = pd.DataFrame({
        # Volume
        "transaction_count": group["amount_cad"].count(),
        "credit_count": group["is_credit"].sum(),
        "debit_count": group["is_debit"].sum(),

        # Amounts
        "total_amount": group["amount_cad"].sum(),
        "sum_sq_amount": group["amount_cad"].apply(lambda x: (x ** 2).sum()),
        "mean_amount": group["amount_cad"].mean(),
        "std_amount": group["amount_cad"].std(),
        "max_amount": group["amount_cad"].max(),
        "min_amount": group["amount_cad"].min(),
        "median_amount": group["amount_cad"].median(),

        # AML signals
        "round_count": group["is_round_100"].sum(),
        "near_10k_count": group["is_near_10k"].sum(),

        # Temporal
        "weekend_count": group["is_weekend"].sum(),
        "night_count": group["is_night"].sum()
    })

    # Get credit/debit amounts
    credit_mask = df["is_credit"] == 1
    debit_mask = df["is_debit"] == 1
    agg["total_credit_amount"] = df[credit_mask].groupby("customer_id")["amount_cad"].sum()
    agg["total_debit_amount"] = df[debit_mask].groupby("customer_id")["amount_cad"].sum()

    # in case a customer only has either credit or debit exclusively
    agg["total_credit_amount"] = agg["total_credit_amount"].fillna(0)
    agg["total_debit_amount"] = agg["total_debit_amount"].fillna(0)


    # Derived ratios
    agg["credit_ratio"] = agg["credit_count"] / agg["transaction_count"]
    agg["net_flow"] = agg["total_credit_amount"] - agg["total_debit_amount"]
    agg["round_ratio"] = agg["round_count"] / agg["transaction_count"]
    agg["weekend_ratio"] = agg["weekend_count"] / agg["transaction_count"]
    agg["night_ratio"] = agg["night_count"] / agg["transaction_count"]

    # Active days and burst score
    if df["transaction_datetime"].notna().any():
        date_series = df.loc[has_time, "transaction_datetime"].dt.date
        agg["active_days"] = df.loc[has_time].groupby("customer_id")["transaction_datetime"].apply(lambda x: x.dt.date.nunique()) # count unique days each customer made transactions
        daily_counts = (df.loc[has_time]
                        .assign(date_only=date_series.values)
                        .groupby(["customer_id", "date_only"])
                        .size())
        agg["max_daily_count"] = daily_counts.groupby(level=0).max()
        avg_daily_transactions = agg["transaction_count"] / agg["active_days"].clip(lower=1)
        agg["burst_score"] = agg["max_daily_count"] / avg_daily_transactions

    else:
        agg["active_days"] = 1
        agg["max_daily_count"] = agg["transaction_count"]
        agg["burst_score"] = 1

    # Channel-specific extras
    if channel == "abm" and "cash_indicator" in df.columns:
        df["cash_indicator"] = pd.to_numeric(df["cash_indicator"], errors="coerce").fillna(0)
        agg["cash_count"] = df.groupby("customer_id")["cash_indicator"].sum() # counting number of transactions involving physical cash at ABM
        agg["cash_ratio"] = agg["cash_count"] / agg["transaction_count"]
        if "country" in df.columns:
            df["is_foreign"] = (df["country"] != "CA").astype(int)
            agg["foreign_ratio"] = df.groupby("customer_id")["is_foreign"].mean() # ratio of ABM transactions occurred at foreign ABMs/ all ABM transactions
    
    if channel == "card":
        if "ecommerce_ind" in df.columns:
            df["ecommerce_ind"] = pd.to_numeric(df["ecommerce_ind"], errors="coerce").fillna(0)
            agg["ecommerce_ratio"] = df.groupby("customer_id")["ecommerce_ind"].mean() # ratio of card transactions for ecommerce / all card transactions
            agg["ecommerce_count"] = df.groupby("customer_id")["ecommerce_ind"].sum()
            agg["unique_mcc"] = df.groupby("customer_id")["merchant_category"].nunique()
        if "country" in df.columns:
            df["is_foreign"] = (df["country"] != "CA").astype(int)
            agg["foreign_ratio"] = df.groupby("customer_id")["is_foreign"].mean() # ratio of card transactions made at foreign countries / all card transactions
            agg["foreign_count"] = df.groupby("customer_id")["is_foreign"].sum()
    
    return agg


def _combine_chunks(partial_list):
    """
    Recombine partial aggregates from chunks

    Problem: a customer might have too many card transactions such that
    the transactions are split across chunks. Each chunk computed mean, std and other stats independently.
    It is necessary to combine the partial results correctly

    Strategy: for additive stats (count, sum), just sum
    For min/max, take min/max. Recompute derived stats from the sums
    """

    combined = pd.concat(partial_list)

    # Define how to combine each column
    additive = ["transaction_count", "credit_count", "debit_count", "total_amount", "sum_sq_amount",
                "total_credit_amount", "total_debit_amount", "round_count", "near_10k_count", "weekend_count", 
                "night_count", "active_days", "max_daily_count"]
    take_max = ["max_amount"]
    take_min = ["min_amount"]

    agg_dict = {}
    for col in combined.columns:
        if col in take_max:
            agg_dict[col] = "max"
        elif col in take_min:
            agg_dict[col] = "min"
        elif col in additive:
            agg_dict[col] = "sum"
        
        # Skip derived columns to recompute them later
    result = combined.groupby(combined.index).agg(agg_dict)

    # Recompute derived columns from the correct totals
    result["mean_amount"] = result["total_amount"] / result["transaction_count"]
    variance = (result["sum_sq_amount"] / result["transaction_count"]) - result["mean_amount"] ** 2 # var = E[X^2] - E[X]^2
    result["std_amount"] = np.sqrt(variance.clip(lower=0))
    result.drop(columns=["sum_sq_amount"], inplace=True)

    result["median_amount"] = np.nan # can't recover median from chunks because finding median requires knowledge of all data points 
    result["credit_ratio"] = result["credit_count"] / result["transaction_count"]
    result["net_flow"] = result["total_credit_amount"] - result["total_debit_amount"]
    result["round_ratio"] = result["round_count"] / result["transaction_count"]
    result["weekend_ratio"] = result["weekend_count"] / result["transaction_count"]
    result["night_ratio"] = result["night_count"] / result["transaction_count"]

    avg_daily_transactions = result["transaction_count"] / result["active_days"].clip(lower=1)
    result["burst_score"] = result["max_daily_count"] / avg_daily_transactions


    # Channel-specific columns that are additive
    for col in ["cash_count", "ecommerce_count", "unique_mcc", "foreign_count"]:
        if col in combined.columns:
            result[col] = combined.groupby(combined.index)[col].sum()

    if "cash_count" in result.columns and "transaction_count" in result.columns:
        result["cash_ratio"] = result["cash_count"] / result["transaction_count"]
    
    if "ecommerce_count" in result.columns:
        result["ecommerce_ratio"] = result["ecommerce_count"] / result["transaction_count"]
    
    if "foreign_count" in result.columns:
        result["foreign_ratio"] = result["foreign_count"] / result["transaction_count"]
        
    return result


def aggregate_channel_transactions(channel):
    """
    Load and aggregate a channel's transactions into per-customer features.
    Uses chunked reading for card.csv (3.55M rows)
    Prefixes all columns with channel name
    """
    filepath = os.path.join(DATA_DIR, f"{channel}.csv")
    use_chunks = os.path.getsize(filepath) > 50_000_000 # >50MB
    if use_chunks:
        chunks = load_transactions(channel, chunksize=500_000)
        partial = []
        for chunk in tqdm(chunks, desc=f"Aggregating {channel}"): # show live progress bar for visual illustration
            partial.append(_agg_single_df(chunk, channel))
        result = _combine_chunks(partial)
    
    else:
        df = load_transactions(channel)
        result = _agg_single_df(df, channel)
    
    # Prefix columns to distinguish channels and to avoid colliding when merging all channels into a DataFrame for each customer
    result.columns = [f"{channel}_{c}" for c in result.columns]
    return result


def build_cross_channel_features(all_channels_df, kyc_features):
    """
    Compute features that span across multiple channels
    Such features capture layering behavior and income mismatches
    """

    cross = pd.DataFrame(index=all_channels_df.index)

    # For each customer, track the number of channels used
    # A channel is active if its transaction count is positive
    transaction_counts = {}
    for channel in CHANNELS:
        col = f"{channel}_transaction_count"
        if col in all_channels_df.columns:
            transaction_counts[channel] = all_channels_df[col]

    transaction_count_df = pd.DataFrame(transaction_counts, index=all_channels_df.index).fillna(0)
    cross["num_active_channels"] = (transaction_count_df > 0).sum(axis=1)

    # Totals across all channels
    cross["total_transaction_count"] = transaction_count_df.sum(axis=1)

    total_amounts = []
    net_flows = []
    near_10k_counts = []
    round_counts = []
    for channel in CHANNELS:
        amount_col = f"{channel}_total_amount"
        net_flow_col = f"{channel}_net_flow"
        near_10k_col = f"{channel}_near_10k_count"
        round_col = f"{channel}_round_count"

        if amount_col in all_channels_df.columns:
            total_amounts.append(all_channels_df[amount_col])
        
        if net_flow_col in all_channels_df.columns:
            net_flows.append(all_channels_df[net_flow_col])
        
        if near_10k_col in all_channels_df.columns:
            near_10k_counts.append(all_channels_df[near_10k_col])
        
        if round_col in all_channels_df.columns:
            round_counts.append(all_channels_df[round_col])
    cross["total_amount_all"] = pd.concat(total_amounts, axis=1).sum(axis=1)
    cross["total_net_flow"] = pd.concat(net_flows, axis=1).sum(axis=1)

    # Sum near-threshold and round-amount counts across ALL channels
    # A customer spreading across multiple channels is sus
    cross["total_near_10k"] = pd.concat(near_10k_counts, axis=1).sum(axis=1)
    cross["total_round_count"] = pd.concat(round_counts, axis=1).sum(axis=1)
    cross["overall_round_ratio"] = cross["total_round_count"] / cross["total_transaction_count"].clip(lower=1)
    cross["overall_near_10k_ratio"] = cross["total_near_10k"] / cross["total_transaction_count"].clip(lower=1)

    # Channel concentration (Herfindahl-Hirschman Index)
    # HHI = sum of (channel_share)^2
    # high HHI suggests high concentration in a small number of channels
    # low HHI suggest low concentrations across a high number of channels
    shares = transaction_count_df.div(cross["total_transaction_count"].clip(lower=1), axis=0)
    cross["channel_hhi"] = (shares ** 2).sum(axis=1)

    # Income mismatch 
    income = kyc_features["income"].reindex(all_channels_df.index).fillna(0).clip(lower=1)
    cross["transaction_to_income_ratio"] = cross["total_amount_all"] / income
    cross["net_flow_to_income_ratio"] = cross["total_net_flow"].abs() / income

    # Credit-heavy behaviour across channels
    # Launderers receive more than they spend => credit >> debit
    total_credit = []
    total_debit = []
    for channel in CHANNELS:
        credit_col = f"{channel}_total_credit_amount"
        debit_col = f"{channel}_total_debit_amount"
        if credit_col in all_channels_df.columns:
            total_credit.append(all_channels_df[credit_col])
        if debit_col in all_channels_df.columns:
            total_debit.append(all_channels_df[debit_col])

    cross["total_credit_all"] = pd.concat(total_credit, axis=1).sum(axis=1)
    cross["total_debit_all"] = pd.concat(total_debit, axis=1).sum(axis=1)
    cross["overall_credit_ratio"] = cross["total_credit_all"] / (cross["total_credit_all"] + cross["total_debit_all"]).clip(lower=1)

    return cross

def build_feature_matrix():
    """
    Master function: loads all data, engineers features, 
    and returns a single DataFrame with 200 features for all customers
    """

    # load KYC data and build features
    individual = load_kyc_individual()
    individual_features = build_individual_features(individual)

    business = load_kyc_smallbusiness()
    business_features = build_business_features(business)

    # Stack individual and business into a DataFrame
    kyc_features = pd.concat([individual_features, business_features])

    # Aggregate each channel's transactions into per-customer stats
    channel_dfs = {}
    for channel in CHANNELS:
        channel_dfs[channel] = aggregate_channel_transactions(channel)
    

    # Join all channels horizontally onto the KYC features
    # Reindex to ensure every customer gets a row
    features = kyc_features.copy()
    for channel in CHANNELS:
        channel_df = channel_dfs[channel].reindex(features.index).fillna(0)
        features = features.join(channel_df)
    

    cross = build_cross_channel_features(features, kyc_features)
    features = features.join(cross)


    # Attach labels
    labels = load_labels().set_index("customer_id")
    features = features.join(labels, how="left") # keep all customers so unlabeled customers get NaN

    return features



