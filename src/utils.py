"""Utility functions for MMM Real Estate project."""

import numpy as np
import pandas as pd


# 13 French regions with transaction share weights
REGIONS = {
    "Île-de-France": 0.25,
    "Auvergne-Rhône-Alpes": 0.12,
    "Nouvelle-Aquitaine": 0.08,
    "Occitanie": 0.09,
    "Hauts-de-France": 0.07,
    "Grand Est": 0.06,
    "Provence-Alpes-Côte d'Azur": 0.10,
    "Pays de la Loire": 0.05,
    "Bretagne": 0.05,
    "Normandie": 0.04,
    "Bourgogne-Franche-Comté": 0.03,
    "Centre-Val de Loire": 0.03,
    "Corse": 0.01,
}

MEDIA_CHANNELS = ["tv", "google_ads", "meta", "google_play", "apple_search_ads"]

CHANNEL_LABELS = {
    "tv": "TV",
    "google_ads": "Google Ads (SEA)",
    "meta": "Meta (FB/Insta)",
    "google_play": "Google Play",
    "apple_search_ads": "Apple Search Ads",
}

# Adstock decay rates (rémanence)
# TV: forte mémoire (brand awareness, on repense à la pub des mois après)
# Google Ads / App Install: effet instantané, coupe = stop
# Meta: léger effet mémoire (retargeting, social)
ADSTOCK_DECAY = {
    "tv": 0.75,
    "google_ads": 0.05,
    "meta": 0.25,
    "google_play": 0.10,
    "apple_search_ads": 0.10,
}

# Logistic saturation alpha
SATURATION_ALPHA = {
    "tv": 0.85,
    "google_ads": 0.4,
    "meta": 0.5,
    "google_play": 0.6,
    "apple_search_ads": 0.6,
}


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock transformation."""
    result = np.zeros_like(x, dtype=float)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = x[i] + decay * result[i - 1]
    return result


def hill_saturation(x: np.ndarray, alpha: float, lam: float = 1.0) -> np.ndarray:
    """Hill function saturation: x^alpha / (lam^alpha + x^alpha)."""
    x_norm = x / (x.max() + 1e-8)
    return x_norm**alpha / (lam**alpha + x_norm**alpha)


def format_euros(value: float) -> str:
    """Format a number as euros."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M€"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k€"
    return f"{value:.0f}€"
