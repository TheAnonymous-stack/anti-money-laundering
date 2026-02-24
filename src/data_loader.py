"""This module centralizes all CSV loading"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data') # path pointer to data directory to avoid hardcoding absolute paths

CHANNELS = ["abm", "card", "cheque", "eft", "emt", "westernunion", "wire"]

def load_kyc_individual():
    df = pd.read_csv(os.path.join(DATA_DIR, "kyc_individual.csv"))
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce") # convert date from strings to datetime objects or Not a Time object if a row misses the value for birth_date
    df["onboard_date"] = pd.to_datetime(df["onboard_date"], errors="coerce")
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    return df

def load_kyc_smallbusiness():
    df = pd.read_csv(os.path.join(DATA_DIR, "kyc_smallbusiness.csv"))
    df["established_date"] = pd.to_datetime(df["established_date"], errors="coerce")
    df["onboard_date"] = pd.to_datetime(df["onboard_date"], errors="coerce")
    df["employee_count"] = pd.to_numeric(df["employee_count"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    return df

def load_labels():
    return pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))

def load_occupation_codes():
    return pd.read_csv(os.path.join(DATA_DIR, "kyc_occupation_codes.csv")).dropna(subset=["occupation_code"]) # drop rows that lack a value for occupation_code which is used to look up occupation title given occupation_code in kyc_individual.csv

def load_industry_codes():
    return pd.read_csv(os.path.join(DATA_DIR, "kyc_industry_codes.csv")).dropna(subset=["industry_code"])

def load_transactions(channel, chunksize=None):
    """
    Load a transaction file. Pass chunksize for large files like card.csv
    """
    path = os.path.join(DATA_DIR, f"{channel}.csv")
    if chunksize:
        return pd.read_csv(path, chunksize=chunksize)
    return pd.read_csv(path)

def load_all_customer_ids():
    """
    Union of all customer IDs from both KYC files
    """
    individual = load_kyc_individual()["customer_id"]
    business = load_kyc_smallbusiness()["customer_id"]
    return pd.concat([individual, business]).reset_index(drop=True) # append the business rows below the individual rows and discard the old indices of business rows to avoid index duplicates

