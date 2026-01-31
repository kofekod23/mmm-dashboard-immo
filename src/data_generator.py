"""
Synthetic data generator for MMM Real Estate POC.

Generates weekly data (2020-W01 to 2025-W52) across 13 French regions
with realistic media spend, macro variables, and lead/download targets.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import (
    REGIONS,
    MEDIA_CHANNELS,
    ADSTOCK_DECAY,
    SATURATION_ALPHA,
    geometric_adstock,
    hill_saturation,
)

np.random.seed(42)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"


def generate_date_index() -> pd.DatetimeIndex:
    """Weekly dates from 2020-W01 to 2025-W52."""
    return pd.date_range("2020-01-06", "2025-12-29", freq="W-MON")


def yearly_transactions_baseline(year: int) -> float:
    """Approximate annual real estate transactions in France (DVF-inspired)."""
    mapping = {
        2020: 1_020_000,  # COVID dip
        2021: 1_170_000,  # rebound
        2022: 1_130_000,
        2023: 870_000,    # rate shock
        2024: 920_000,    # slow recovery
        2025: 980_000,
    }
    return mapping.get(year, 1_000_000)


def seasonality_factor(week: int, month: int) -> float:
    """Seasonal multiplier: spring/summer peak, winter trough, Aug dip."""
    base = 1.0 + 0.15 * np.sin(2 * np.pi * (week - 13) / 52)
    if month in (7, 8):
        base *= 0.90
    if month == 12:
        base *= 0.92
    return base


def interest_rate(date: pd.Timestamp) -> float:
    """20-year mortgage rate approximation."""
    year_frac = date.year + date.month / 12
    if year_frac < 2022.0:
        return 1.2 + 0.1 * (year_frac - 2020)
    if year_frac < 2023.5:
        return 1.4 + 1.8 * (year_frac - 2022.0)
    if year_frac < 2024.5:
        return 4.0 - 0.3 * (year_frac - 2023.5)
    return 3.5 - 0.1 * (year_frac - 2024.5)


def usury_rate(date: pd.Timestamp) -> float:
    """Usury rate (taux d'usure) approximation — quarterly steps."""
    ir = interest_rate(date)
    return ir + 1.0 + 0.1 * np.sin(date.quarter)


def pinel_coefficient(date: pd.Timestamp) -> float:
    """Pinel tax incentive coefficient."""
    if date.year <= 2022:
        return 1.0
    if date.year == 2023:
        return 0.7
    if date.year == 2024:
        return 0.3
    return 0.0


def covid_impact(date: pd.Timestamp) -> float:
    """COVID lockdown impact multiplier (1 = no impact, <1 = negative)."""
    y, m = date.year, date.month
    if y == 2020 and m in (3, 4, 5):
        return 0.40 if m == 4 else 0.60
    if y == 2020 and m in (11, 12):
        return 0.75
    if y == 2021 and m in (1, 2, 3, 4, 5):
        return 0.85
    return 1.0


def generate_media_spend(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate weekly media spend per channel."""
    n = len(dates)
    spend = pd.DataFrame(index=range(n))

    for i, d in enumerate(dates):
        week, month, year = d.isocalendar()[1], d.month, d.year
        year_factor = 1.0 + 0.03 * (year - 2020)  # slight YoY increase

        # TV (BFM etc): 4 bursts of 1 month (Jan, Apr, Sep, Nov) ~200k€/burst → ~800k€/year
        if month in (1, 4, 9, 11):
            spend.loc[i, "spend_tv"] = np.random.uniform(45_000, 55_000) * year_factor
        else:
            spend.loc[i, "spend_tv"] = np.random.uniform(0, 2_000) * year_factor

        # Radio: 6 bursts of 1 month (Jan, Mar, May, Jul, Sep, Nov) → ~600k€/year
        if month in (1, 3, 5, 7, 9, 11):
            spend.loc[i, "spend_radio"] = np.random.uniform(20_000, 28_000) * year_factor
        else:
            spend.loc[i, "spend_radio"] = np.random.uniform(0, 1_500) * year_factor

        # RATP/Bus: 4 bursts of 1 month (Feb, May, Sep, Nov) → ~400k€/year
        if month in (2, 5, 9, 11):
            spend.loc[i, "spend_ratp_display"] = np.random.uniform(20_000, 28_000) * year_factor
        else:
            spend.loc[i, "spend_ratp_display"] = np.random.uniform(0, 1_000) * year_factor

        # Google Ads: always-on, effet instantané, coupe = stop → ~2M€/year
        base_google = np.random.uniform(30_000, 45_000) * year_factor
        # boost spring
        if month in (3, 4, 5, 6):
            base_google *= 1.3
        # summer: mild dip
        if month in (7, 8):
            base_google *= 0.85
        # Q4 push
        if month in (10, 11):
            base_google *= 1.2
        spend.loc[i, "spend_google_ads"] = base_google

    return spend.astype(float)


def generate_national_data() -> pd.DataFrame:
    """Generate national-level weekly dataset."""
    dates = generate_date_index()
    n = len(dates)

    df = pd.DataFrame({"date": dates})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    # Macro variables
    df["interest_rate_20y"] = df["date"].apply(interest_rate)
    df["usury_rate"] = df["date"].apply(usury_rate)
    df["pinel"] = df["date"].apply(pinel_coefficient)
    df["covid_impact"] = df["date"].apply(covid_impact)

    # Seasonality
    df["seasonality"] = df.apply(lambda r: seasonality_factor(r["week"], r["month"]), axis=1)

    # Baseline weekly transactions (national)
    df["baseline_transactions"] = df.apply(
        lambda r: yearly_transactions_baseline(r["year"]) / 52 * r["seasonality"] * r["covid_impact"],
        axis=1,
    )

    # Media spend
    spend_df = generate_media_spend(dates)
    for col in spend_df.columns:
        df[col] = spend_df[col].values

    # Adstock transformations
    for ch in MEDIA_CHANNELS:
        col_spend = f"spend_{ch}"
        col_adstock = f"adstock_{ch}"
        df[col_adstock] = geometric_adstock(df[col_spend].values, ADSTOCK_DECAY[ch])

    # Saturated media effects
    for ch in MEDIA_CHANNELS:
        col_adstock = f"adstock_{ch}"
        col_sat = f"saturated_{ch}"
        df[col_sat] = hill_saturation(df[col_adstock].values, SATURATION_ALPHA[ch])

    # Channel contribution coefficients (leads per unit of saturated media)
    # Target CPAs: Google Ads ~€35 < RATP ~€55 < Radio ~€80 < TV ~€130
    coefficients = {
        "tv": 150,
        "radio": 1600,
        "ratp_display": 500,
        "google_ads": 30000,
    }

    # Interest rate effect: higher rates = fewer leads
    rate_effect = -800 * (df["interest_rate_20y"] - 2.0)

    # Pinel effect
    pinel_effect = 600 * df["pinel"]

    # Conversion rate from transactions to leads (~0.8%)
    conversion_rate = 0.008

    # Compute leads
    df["leads"] = (
        df["baseline_transactions"] * conversion_rate
        + df["saturated_tv"] * coefficients["tv"]
        + df["saturated_radio"] * coefficients["radio"]
        + df["saturated_ratp_display"] * coefficients["ratp_display"]
        + df["saturated_google_ads"] * coefficients["google_ads"]
        + rate_effect
        + pinel_effect
        + np.random.normal(0, 300, n)
    ).clip(lower=100)

    # App downloads: correlated with leads + extra digital boost
    df["app_downloads"] = (
        df["leads"] * np.random.uniform(1.8, 2.5, n)
        + df["saturated_google_ads"] * 2000
        + np.random.normal(0, 500, n)
    ).clip(lower=50).astype(int)

    df["leads"] = df["leads"].round().astype(int)

    return df


def generate_regional_data(national_df: pd.DataFrame) -> pd.DataFrame:
    """Split national data into 13 regions with appropriate weights."""
    rows = []
    for _, row in national_df.iterrows():
        for region, weight in REGIONS.items():
            r = row.copy()
            r["region"] = region
            noise = np.random.uniform(0.9, 1.1)
            r["leads"] = max(1, int(row["leads"] * weight * noise))
            r["app_downloads"] = max(1, int(row["app_downloads"] * weight * noise))

            # RATP only impacts IDF
            if region != "Île-de-France":
                r["spend_ratp_display"] = 0.0
                r["adstock_ratp_display"] = 0.0
                r["saturated_ratp_display"] = 0.0
            else:
                # Scale spend to regional level
                pass

            # Scale spend by region weight (except RATP handled above)
            for ch in ["tv", "radio", "google_ads"]:
                r[f"spend_{ch}"] = row[f"spend_{ch}"] * weight

            rows.append(r)

    regional_df = pd.DataFrame(rows)
    return regional_df


def main():
    """Generate and save datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating national data...")
    national_df = generate_national_data()
    national_path = OUTPUT_DIR / "national_weekly.csv"
    national_df.to_csv(national_path, index=False)
    print(f"  Saved {len(national_df)} rows to {national_path}")

    print("Generating regional data...")
    regional_df = generate_regional_data(national_df)
    regional_path = OUTPUT_DIR / "regional_weekly.csv"
    regional_df.to_csv(regional_path, index=False)
    print(f"  Saved {len(regional_df)} rows to {regional_path}")

    # Summary
    print("\n--- Summary ---")
    print(f"Date range: {national_df['date'].min()} to {national_df['date'].max()}")
    print(f"Weeks: {len(national_df)}")
    print(f"Regions: {regional_df['region'].nunique()}")
    print(f"Total leads (national): {national_df['leads'].sum():,}")
    print(f"Total app downloads (national): {national_df['app_downloads'].sum():,}")
    for ch in MEDIA_CHANNELS:
        total = national_df[f"spend_{ch}"].sum()
        print(f"  Total spend {ch}: {total:,.0f}€")


if __name__ == "__main__":
    main()
