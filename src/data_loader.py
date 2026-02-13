"""
Data loading and preparation module.
Uses the freMTPL2freq and freMTPL2sev datasets from sklearn (OpenML) —
standard actuarial datasets for motor third-party liability insurance in France.
"""

import pandas as pd
import numpy as np
import ssl
import certifi
import os

# Fix SSL certificate issue on macOS
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_openml


def load_data() -> pd.DataFrame:
    """Load and prepare the freMTPL2freq dataset with severity for pure premium."""

    # -------------------------------------------------------------------
    # Fetch frequency data from OpenML
    # -------------------------------------------------------------------
    freq = fetch_openml(data_id=41214, as_frame=True, parser="auto")
    df = freq.frame.copy()

    # -------------------------------------------------------------------
    # Fetch severity data from OpenML
    # -------------------------------------------------------------------
    try:
        sev = fetch_openml(data_id=41215, as_frame=True, parser="auto")
        df_sev = sev.frame.copy()
        df_sev["IDpol"] = pd.to_numeric(df_sev["IDpol"], errors="coerce")
        df_sev["ClaimAmount"] = pd.to_numeric(df_sev["ClaimAmount"], errors="coerce")

        # Aggregate severity per policy: total cost and average cost per claim
        sev_agg = df_sev.groupby("IDpol").agg(
            TotalClaimAmount=("ClaimAmount", "sum"),
            AvgClaimAmount=("ClaimAmount", "mean"),
            NbClaims_sev=("ClaimAmount", "count")
        ).reset_index()

        has_severity = True
    except Exception:
        has_severity = False

    # -------------------------------------------------------------------
    # Basic type conversions
    # -------------------------------------------------------------------
    numeric_cols = ["IDpol", "Exposure", "ClaimNb", "VehPower", "VehAge",
                    "DrivAge", "BonusMalus", "Density"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Area"] = df["Area"].astype(str).str.strip("'\"")
    df["VehBrand"] = df["VehBrand"].astype(str).str.strip("'\"")
    df["VehGas"] = df["VehGas"].astype(str).str.strip("'\"")
    df["Region"] = df["Region"].astype(str).str.strip("'\"")

    # -------------------------------------------------------------------
    # Merge severity data
    # -------------------------------------------------------------------
    if has_severity:
        df = df.merge(sev_agg, on="IDpol", how="left")
        df["TotalClaimAmount"] = df["TotalClaimAmount"].fillna(0)
        df["AvgClaimAmount"] = df["AvgClaimAmount"].fillna(0)
    else:
        # Simulate severity if dataset unavailable
        np.random.seed(42)
        mask = df["ClaimNb"] > 0
        df["TotalClaimAmount"] = 0.0
        df.loc[mask, "TotalClaimAmount"] = np.random.lognormal(
            mean=7.5, sigma=1.0, size=mask.sum()
        ) * df.loc[mask, "ClaimNb"]
        df["AvgClaimAmount"] = 0.0
        df.loc[mask, "AvgClaimAmount"] = (
            df.loc[mask, "TotalClaimAmount"] / df.loc[mask, "ClaimNb"]
        )

    # -------------------------------------------------------------------
    # Filters (standard actuarial cleaning)
    # -------------------------------------------------------------------
    df = df[df["Exposure"].between(0.01, 1.0)].copy()
    df = df[df["BonusMalus"].between(50, 230)].copy()

    # Cap severity outliers (above 99.5th percentile of non-zero claims)
    non_zero_mask = df["TotalClaimAmount"] > 0
    if non_zero_mask.any():
        sev_threshold = df.loc[non_zero_mask, "TotalClaimAmount"].quantile(0.995)
        df["TotalClaimAmount"] = df["TotalClaimAmount"].clip(upper=sev_threshold)
        df.loc[df["ClaimNb"] > 0, "AvgClaimAmount"] = (
            df.loc[df["ClaimNb"] > 0, "TotalClaimAmount"]
            / df.loc[df["ClaimNb"] > 0, "ClaimNb"]
        )

    # -------------------------------------------------------------------
    # Feature engineering
    # -------------------------------------------------------------------
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["PurePremium"] = df["TotalClaimAmount"] / df["Exposure"]
    df["Severity"] = np.where(
        df["ClaimNb"] > 0,
        df["TotalClaimAmount"] / df["ClaimNb"],
        0
    )

    # Binned driver age
    bins_age = [17, 25, 35, 45, 55, 65, 100]
    labels_age = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    df["DrivAge_bin"] = pd.cut(df["DrivAge"], bins=bins_age, labels=labels_age)

    # Binned vehicle age
    bins_veh = [-1, 1, 5, 10, 15, 50]
    labels_veh = ["0-1", "2-5", "6-10", "11-15", "15+"]
    df["VehAge_bin"] = pd.cut(df["VehAge"], bins=bins_veh, labels=labels_veh)

    # Binned vehicle power
    bins_pow = [0, 5, 7, 9, 12, 20]
    labels_pow = ["≤5", "6-7", "8-9", "10-12", "13+"]
    df["VehPower_bin"] = pd.cut(df["VehPower"], bins=bins_pow, labels=labels_pow)

    # Binned BonusMalus
    bins_bm = [49, 60, 80, 100, 120, 230]
    labels_bm = ["50-60", "61-80", "81-100", "101-120", "121+"]
    df["BonusMalus_bin"] = pd.cut(df["BonusMalus"], bins=bins_bm, labels=labels_bm)

    # Log density
    df["LogDensity"] = np.log1p(df["Density"])

    df.reset_index(drop=True, inplace=True)

    return df


def get_modeling_data(df: pd.DataFrame):
    """
    Prepare X, y, weights for frequency modeling.
    Returns X (DataFrame), y (Series), w (Series/array), claim_count, df_model.
    """
    features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity",
                "Area", "VehGas", "VehBrand", "Region"]

    bin_cols = ["DrivAge_bin", "VehAge_bin", "VehPower_bin", "BonusMalus_bin"]
    cols_needed = features + ["ClaimNb", "Exposure", "Frequency",
                               "TotalClaimAmount", "Severity", "PurePremium"] + \
                  [c for c in bin_cols if c in df.columns]
    df_model = df[cols_needed].dropna(subset=features).copy()

    # One-hot encode categoricals
    cat_cols = ["Area", "VehGas", "VehBrand", "Region"]
    df_encoded = pd.get_dummies(df_model[features], columns=cat_cols, drop_first=True,
                                 dtype=float)

    X = df_encoded
    y = df_model["Frequency"].values
    w = df_model["Exposure"].values
    claim_count = df_model["ClaimNb"].values

    return X, y, w, claim_count, df_model
