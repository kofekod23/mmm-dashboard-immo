"""
MMM Model using PyMC-Marketing for Real Estate leads.

Fits a Media Mix Model on the national weekly dataset,
then provides channel contributions, ROAS, and budget optimization.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "model"


def load_data() -> pd.DataFrame:
    """Load national weekly data."""
    df = pd.read_csv(DATA_DIR / "national_weekly.csv", parse_dates=["date"])
    return df


def prepare_model_data(df: pd.DataFrame) -> dict:
    """Prepare data arrays for the MMM."""
    spend_cols = ["spend_tv", "spend_radio", "spend_ratp_display", "spend_google_ads"]
    control_cols = ["interest_rate_20y", "pinel", "covid_impact"]

    X_media = df[spend_cols].values
    X_control = df[control_cols].values
    y = df["leads"].values.astype(float)
    date_col = df["date"].values

    return {
        "X_media": X_media,
        "X_control": X_control,
        "y": y,
        "dates": date_col,
        "spend_cols": spend_cols,
        "control_cols": control_cols,
    }


def fit_mmm(df: pd.DataFrame, samples: int = 1000, tune: int = 500):
    """
    Fit MMM using PyMC-Marketing.

    Returns the fitted MMM object, or a fallback dict if pymc-marketing
    is not installed.
    """
    try:
        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

        channel_columns = ["spend_tv", "spend_radio", "spend_ratp_display", "spend_google_ads"]
        control_columns = ["interest_rate_20y", "pinel", "covid_impact"]
        date_column = "date"
        target_column = "leads"

        mmm = MMM(
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            date_column=date_column,
            channel_columns=channel_columns,
            control_columns=control_columns,
        )

        model_df = df[[date_column] + channel_columns + control_columns + [target_column]].copy()
        model_df = model_df.dropna()

        X = model_df.drop(columns=[target_column])
        y = model_df[target_column].values.astype(float)

        mmm.fit(X=X, y=y, target_accept=0.85, chains=2, draws=samples, tune=tune)
        return mmm

    except ImportError:
        print("pymc-marketing not installed. Using fallback analytical model.")
        return _fit_fallback(df)


def _fit_fallback(df: pd.DataFrame) -> dict:
    """
    Lightweight OLS-based fallback when PyMC-Marketing is unavailable.
    Returns a dict with model results.
    """
    from sklearn.linear_model import Ridge
    from src.utils import MEDIA_CHANNELS, geometric_adstock, hill_saturation, ADSTOCK_DECAY, SATURATION_ALPHA

    spend_cols = [f"spend_{ch}" for ch in MEDIA_CHANNELS]
    control_cols = ["interest_rate_20y", "pinel", "covid_impact"]

    # Build feature matrix with adstock + saturation
    features = {}
    for ch in MEDIA_CHANNELS:
        raw = df[f"spend_{ch}"].values
        adstocked = geometric_adstock(raw, ADSTOCK_DECAY[ch])
        saturated = hill_saturation(adstocked, SATURATION_ALPHA[ch])
        features[f"sat_{ch}"] = saturated

    for col in control_cols:
        features[col] = df[col].values

    X = pd.DataFrame(features)
    y = df["leads"].values.astype(float)

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    y_pred = model.predict(X)

    # Channel contributions
    contributions = {}
    for ch in MEDIA_CHANNELS:
        coef = model.coef_[list(X.columns).index(f"sat_{ch}")]
        contrib = coef * X[f"sat_{ch}"].values
        contributions[ch] = contrib

    # Total spend per channel
    total_spend = {ch: df[f"spend_{ch}"].sum() for ch in MEDIA_CHANNELS}
    total_contrib = {ch: contributions[ch].sum() for ch in MEDIA_CHANNELS}
    roas = {ch: total_contrib[ch] / max(total_spend[ch], 1) for ch in MEDIA_CHANNELS}

    results = {
        "type": "fallback",
        "model": model,
        "y_pred": y_pred,
        "y_actual": y,
        "contributions": contributions,
        "total_spend": total_spend,
        "total_contributions": total_contrib,
        "roas": roas,
        "coefficients": dict(zip(X.columns, model.coef_)),
        "intercept": model.intercept_,
        "r2": model.score(X, y),
        "dates": df["date"].values,
    }

    return results


def predict_leads(model_results: dict, spend_scenario: dict) -> float:
    """
    Predict leads for a given weekly spend scenario using the fallback model.

    spend_scenario: dict like {"tv": 200000, "radio": 100000, ...}
    Returns predicted weekly leads (national).
    """
    from src.utils import MEDIA_CHANNELS, ADSTOCK_DECAY, SATURATION_ALPHA

    if isinstance(model_results, dict) and model_results.get("type") == "fallback":
        model = model_results["model"]
        coefficients = model_results["coefficients"]
        intercept = model_results["intercept"]

        predicted = intercept
        for ch in MEDIA_CHANNELS:
            spend = spend_scenario.get(ch, 0)
            # Simple single-week approximation
            adstocked = spend  # single week, no carry-over
            # Normalize using max from training data
            sat_val = (spend / max(model_results["total_spend"][ch] / 313, 1)) ** SATURATION_ALPHA[ch]
            sat_val = sat_val / (1 + sat_val)
            predicted += coefficients.get(f"sat_{ch}", 0) * sat_val

        # Add average control effect
        avg_controls = {
            "interest_rate_20y": 3.0,
            "pinel": 0.4,
            "covid_impact": 1.0,
        }
        for ctrl, val in avg_controls.items():
            predicted += coefficients.get(ctrl, 0) * val

        return max(0, predicted)

    raise ValueError("PyMC-Marketing model prediction not implemented in this fallback path.")


def save_results(results: dict, output_dir: Path = MODEL_DIR):
    """Save model results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = {
        "type": results["type"],
        "r2": float(results["r2"]),
        "intercept": float(results["intercept"]),
        "coefficients": {k: float(v) for k, v in results["coefficients"].items()},
        "total_spend": {k: float(v) for k, v in results["total_spend"].items()},
        "total_contributions": {k: float(v) for k, v in results["total_contributions"].items()},
        "roas": {k: float(v) for k, v in results["roas"].items()},
    }

    with open(output_dir / "model_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        "date": results["dates"],
        "y_actual": results["y_actual"],
        "y_pred": results["y_pred"],
    })
    for ch, contrib in results["contributions"].items():
        pred_df[f"contrib_{ch}"] = contrib
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    print(f"Results saved to {output_dir}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows loaded.")

    print("Fitting model...")
    results = fit_mmm(df)

    if isinstance(results, dict):
        print(f"\n--- Model Results (R²={results['r2']:.3f}) ---")
        for ch in ["tv", "radio", "ratp_display", "google_ads"]:
            spend = results["total_spend"][ch]
            contrib = results["total_contributions"][ch]
            roas = results["roas"][ch]
            print(f"  {ch:20s}: spend={spend:>12,.0f}€  contrib={contrib:>10,.0f} leads  ROAS={roas:.4f}")

        print("\nSaving results...")
        save_results(results)

        # Test prediction
        test_scenario = {"tv": 200_000, "radio": 150_000, "ratp_display": 100_000, "google_ads": 120_000}
        pred = predict_leads(results, test_scenario)
        print(f"\nTest prediction (weekly): {pred:,.0f} leads")
    else:
        print("PyMC-Marketing model fitted. Use mmm object methods for analysis.")


if __name__ == "__main__":
    main()
