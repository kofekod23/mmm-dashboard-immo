"""
Synthetic data generator for MMM Real Estate POC.

Generates weekly data (2020-W01 to 2025-W52) across 13 French regions
with media spend calibrated to SL's publicly available benchmarks:
  - Annual budget ~20M EUR
  - TV & Offline: 50% (10M), Google Ads SEA: 25% (5M),
    Meta: 10% (2M), Google Play: ~3% (600K), Apple Search Ads: ~2% (400K)
  - 48M leads/year, ~920K leads/week
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
    """Seasonal multiplier aligned with SL rhythm:
    Mar-Jun VERY HIGH, Sep-Oct HIGH, Jul-Aug LOW, Nov-Dec MEDIUM, Jan-Feb LOW.
    """
    base = 1.0 + 0.15 * np.sin(2 * np.pi * (week - 13) / 52)
    if month in (7, 8):
        base *= 0.85
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
    """Generate weekly media spend per channel — SL benchmark scale."""
    n = len(dates)
    spend = pd.DataFrame(index=range(n))

    for i, d in enumerate(dates):
        week, month, year = d.isocalendar()[1], d.month, d.year
        year_factor = 1.0 + 0.03 * (year - 2020)  # slight YoY increase

        # ── TV: 4 waves/year (Mar, May, Sep, Nov), ~2.5M per wave = 10M/year
        # Each wave = 4 weeks at ~625K/week
        if month in (3, 5, 9, 11):
            spend.loc[i, "spend_tv"] = np.random.uniform(450_000, 580_000) * year_factor
        else:
            # Minimal maintenance / sponsoring
            spend.loc[i, "spend_tv"] = np.random.uniform(5_000, 20_000) * year_factor

        # ── Google Ads SEA: always-on, ~5M/year = ~96K/week average
        base_google = np.random.uniform(80_000, 115_000) * year_factor
        # Spring peak (families preparing moves)
        if month in (3, 4, 5, 6):
            base_google *= 1.30
        # Summer dip
        if month in (7, 8):
            base_google *= 0.70
        # Back-to-school + Q4
        if month in (9, 10):
            base_google *= 1.20
        spend.loc[i, "spend_google_ads"] = base_google

        # ── Meta (Facebook/Instagram): ~2M/year = ~38K/week average
        # Seasonal: stronger in spring & back-to-school
        base_meta = np.random.uniform(30_000, 48_000) * year_factor
        if month in (3, 4, 5, 6):
            base_meta *= 1.25
        if month in (7, 8):
            base_meta *= 0.65
        if month in (9, 10):
            base_meta *= 1.15
        spend.loc[i, "spend_meta"] = base_meta

        # ── Google Play (App Install campaigns): ~600K/year = ~11.5K/week average
        # Peaks aligned with app-push moments (summer rentals, spring)
        base_gplay = np.random.uniform(8_000, 15_000) * year_factor
        if month in (4, 5, 6):
            base_gplay *= 1.30
        if month in (7, 8):
            # Students looking for rentals = app push
            base_gplay *= 1.40
        spend.loc[i, "spend_google_play"] = base_gplay

        # ── Apple Search Ads: ~400K/year = ~7.7K/week average
        # Similar seasonal pattern to Google Play
        base_apple = np.random.uniform(5_500, 10_000) * year_factor
        if month in (4, 5, 6):
            base_apple *= 1.30
        if month in (7, 8):
            base_apple *= 1.40
        spend.loc[i, "spend_apple_search_ads"] = base_apple

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

    # Baseline weekly leads (organic/SEO/brand = ~80% of total)
    # SL: 48M leads/year → ~920K/week total, baseline ~750K/week
    df["baseline_leads"] = df.apply(
        lambda r: (yearly_transactions_baseline(r["year"]) / 52)
        * r["seasonality"] * r["covid_impact"] * 38,  # scale factor for organic baseline
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

    # Channel contribution coefficients
    # Target CPLs: Google Ads ~10€, Meta ~15€, TV ~50€, Google Play ~6€, Apple ~7€
    coefficients = {
        "tv": 10_000,           # TV: high spend, moderate CPL (~50€)
        "google_ads": 250_000,  # Google Ads: best CPL (~10€)
        "meta": 80_000,         # Meta: moderate CPL (~15€)
        "google_play": 8_000,   # Google Play: small channel, drives downloads more than leads
        "apple_search_ads": 5_000,  # Apple: smallest channel
    }

    # Interest rate effect: higher rates = fewer leads
    rate_effect = -30_000 * (df["interest_rate_20y"] - 2.0)

    # Pinel effect
    pinel_effect = 20_000 * df["pinel"]

    # Compute leads
    df["leads"] = (
        df["baseline_leads"]
        + df["saturated_tv"] * coefficients["tv"]
        + df["saturated_google_ads"] * coefficients["google_ads"]
        + df["saturated_meta"] * coefficients["meta"]
        + df["saturated_google_play"] * coefficients["google_play"]
        + df["saturated_apple_search_ads"] * coefficients["apple_search_ads"]
        + rate_effect
        + pinel_effect
        + np.random.normal(0, 10_000, n)
    ).clip(lower=1_000)

    # App downloads: correlated with leads + strong digital channel boost
    # SL app is a major driver — about 1 download per 3-4 leads
    df["app_downloads"] = (
        df["leads"] * np.random.uniform(0.25, 0.35, n)
        + df["saturated_google_play"] * 15_000
        + df["saturated_apple_search_ads"] * 8_000
        + np.random.normal(0, 5_000, n)
    ).clip(lower=500).astype(int)

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

            # Scale spend by region weight for all channels
            for ch in MEDIA_CHANNELS:
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
    print(f"Avg weekly leads: {national_df['leads'].mean():,.0f}")
    print(f"Total app downloads (national): {national_df['app_downloads'].sum():,}")
    for ch in MEDIA_CHANNELS:
        total = national_df[f"spend_{ch}"].sum()
        print(f"  Total spend {ch}: {total:,.0f}€")


if __name__ == "__main__":
    main()
