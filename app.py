"""
Streamlit Dashboard for MMM Real Estate POC.

Pages:
1. Overview — KPIs, temporal evolution, regional breakdown
2. Channel Contributions — waterfall, contribution %, ROAS
3. Response Curves — saturation curves per channel
4. Budget Simulator — sliders, real-time prediction, CPA, optimization
5. Regional Analysis — heatmap, regional performance
"""

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import urllib.request

from src.utils import (
    REGIONS,
    MEDIA_CHANNELS,
    CHANNEL_LABELS,
    ADSTOCK_DECAY,
    SATURATION_ALPHA,
    geometric_adstock,
    hill_saturation,
    format_euros,
)

DATA_DIR = Path("data/generated")
MODEL_DIR = Path("data/model")

st.set_page_config(page_title="MMM Real Estate Dashboard", layout="wide")

# ── Floating popup navigation ──────────────────────────────────────────────────
st.markdown("""
<style>
.fab-toggle {
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    width: 56px; height: 56px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff; border: none; cursor: pointer;
    box-shadow: 0 4px 16px rgba(99,102,241,.45);
    font-size: 24px; display: flex; align-items: center; justify-content: center;
    transition: transform .2s;
}
.fab-toggle:hover { transform: scale(1.1); }
.fab-popup {
    display: none; position: fixed; bottom: 92px; right: 24px; z-index: 9998;
    background: #1e1e2e; border: 1px solid #444; border-radius: 14px;
    padding: 20px 24px; min-width: 260px;
    box-shadow: 0 8px 32px rgba(0,0,0,.45);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.fab-popup.show { display: block; animation: fadeUp .2s ease-out; }
@keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
.fab-popup h4 { margin: 0 0 14px 0; color: #e0e0e0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
.fab-popup a {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; margin: 6px 0; border-radius: 8px;
    text-decoration: none; color: #e0e0e0; font-weight: 500; font-size: 15px;
    transition: background .15s;
}
.fab-popup a:hover { background: rgba(255,255,255,.08); }
.fab-popup a .icon { font-size: 20px; }
</style>
<div class="fab-popup" id="fabPopup">
    <h4>Navigate</h4>
    <a href="https://mmm-realestate-550505238976.europe-west1.run.app" target="_blank">
        <span class="icon">📊</span> Immo MMM Dashboard
    </a>
    <a href="http://34.22.238.118:8000" target="_blank">
        <span class="icon">🏠</span> HomeVision Search Engine
    </a>
</div>
<button class="fab-toggle" id="fabToggle" onclick="document.getElementById('fabPopup').classList.toggle('show')">⚡</button>
""", unsafe_allow_html=True)


# ── Visitor logging ────────────────────────────────────────────────────────────

ADMIN_PASSWORD = "ROY"
LOG_FILE = Path("data/visitor_log.jsonl")


def get_geolocation(ip: str) -> dict | None:
    """Get geolocation via ip-api.com."""
    if ip in ("127.0.0.1", "localhost", "::1") or ip.startswith(("10.", "192.168.", "172.")):
        return None
    try:
        url = f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,region,regionName,city,zip,lat,lon,isp"
        req = urllib.request.Request(url, headers={"User-Agent": "MMM-Dashboard"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "success":
                return {k: data.get(k) for k in ("country", "countryCode", "region", "regionName", "city", "zip", "lat", "lon", "isp")}
    except Exception:
        pass
    return None


def log_visit(page: str):
    """Log a page visit with IP + geolocation."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Get IP from Streamlit headers (Cloud Run sets X-Forwarded-For)
    ip = "unknown"
    try:
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For", headers.get("X-Real-Ip", "unknown"))
        if "," in ip:
            ip = ip.split(",")[0].strip()
    except Exception:
        pass

    # Deduplicate: don't re-log same IP+page within the same session
    session_key = f"logged_{page}_{ip}"
    if st.session_state.get(session_key):
        return
    st.session_state[session_key] = True

    geo = get_geolocation(ip)
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "page": page,
        "ip": ip,
    }
    if geo:
        entry["geolocation"] = geo

    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def load_visitor_logs(limit: int = 200) -> list:
    """Read visitor logs."""
    if not LOG_FILE.exists():
        return []
    logs = []
    with open(LOG_FILE) as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    logs.reverse()
    return logs[:limit]


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_national():
    return pd.read_csv(DATA_DIR / "national_weekly.csv", parse_dates=["date"])


@st.cache_data
def load_regional():
    df = pd.read_csv(DATA_DIR / "regional_weekly.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_model_results():
    path = MODEL_DIR / "model_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_predictions():
    path = MODEL_DIR / "predictions.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


@st.cache_data
def load_bayesian_posteriors():
    """Load Bayesian posteriors JSON. Returns None if absent (Ridge fallback)."""
    path = MODEL_DIR / "bayesian_posteriors.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _mc_simulate_leads(bayesian, spend_inputs, nat, n_draws=1000):
    """Monte Carlo simulation of predicted leads from Bayesian posteriors.

    Uses total_contribution per channel scaled by spend ratio vs historical average,
    with saturation curve from adstock_alpha posteriors.

    Returns dict with keys: leads_mean, leads_hdi_3, leads_hdi_97,
    channel_contribs (dict of ch -> {mean, hdi_3, hdi_97}).
    """
    rng = np.random.default_rng(42)
    channels = MEDIA_CHANNELS
    ch_data = bayesian["channels"]
    n_weeks = 313  # number of weeks in training data

    # Baseline: use Ridge intercept + controls (Bayesian intercept is normalized)
    results_ridge = load_model_results()
    baseline = results_ridge["intercept"] if results_ridge else 5000.0
    for ctrl in ["interest_rate_20y", "pinel", "covid_impact"]:
        ctrl_coef = results_ridge["coefficients"].get(ctrl, 0) if results_ridge else 0
        baseline += ctrl_coef * nat[ctrl].mean()

    baseline_draws = rng.normal(baseline, abs(baseline) * 0.02, size=n_draws)

    # Draw channel contributions
    total_leads = baseline_draws.copy()
    ch_contrib_draws = {}

    for ch in channels:
        ch_info = ch_data.get(ch, {})
        tc = ch_info.get("total_contribution", {})

        spend = spend_inputs[ch]
        avg_weekly = nat[f"spend_{ch}"].mean()
        x_norm = spend / max(avg_weekly, 1)

        # Draw saturation alpha from posteriors
        sat_alpha_post = ch_info.get("adstock_alpha", {})
        alpha_vals = rng.normal(
            sat_alpha_post.get("mean", SATURATION_ALPHA[ch]),
            sat_alpha_post.get("std", 0.05),
            size=n_draws,
        )
        alpha_vals = np.clip(alpha_vals, 0.01, 5.0)

        # Saturation at current spend vs at average spend (ratio=1)
        sat_at_spend = x_norm ** alpha_vals / (1 + x_norm ** alpha_vals)
        sat_at_avg = 1.0 ** alpha_vals / (1 + 1.0 ** alpha_vals)  # = 0.5 for Hill

        # Scale: weekly contribution at average = total_contribution / n_weeks
        avg_weekly_contrib = tc.get("mean", 0) / n_weeks
        avg_weekly_std = tc.get("std", 0) / n_weeks

        # Draw base weekly contribution, then scale by saturation ratio
        base_contrib = rng.normal(avg_weekly_contrib, avg_weekly_std, size=n_draws)
        ratio = sat_at_spend / np.clip(sat_at_avg, 0.001, None)
        contrib = base_contrib * ratio

        ch_contrib_draws[ch] = contrib
        total_leads += contrib

    total_leads = np.clip(total_leads, 0, None)

    result = {
        "leads_mean": float(np.mean(total_leads)),
        "leads_hdi_3": float(np.percentile(total_leads, 3)),
        "leads_hdi_97": float(np.percentile(total_leads, 97)),
        "channel_contribs": {},
    }
    for ch in channels:
        d = ch_contrib_draws[ch]
        result["channel_contribs"][ch] = {
            "mean": float(np.mean(d)),
            "hdi_3": float(np.percentile(d, 3)),
            "hdi_97": float(np.percentile(d, 97)),
        }
    return result


# ── Sidebar ────────────────────────────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Channel Contributions", "Response Curves", "Budget Simulator", "Forecast", "Goal Planner", "Model Validation", "Fixed Costs", "SL Benchmark", "Regional Analysis", "FAQ", "FAQ (FR)", "Admin"],
)

# ── Page: Overview ─────────────────────────────────────────────────────────────

if page == "Admin":
    st.title("Admin — Visitor Logs")

    pwd = st.text_input("Password", type="password", key="admin_pwd")

    if pwd == ADMIN_PASSWORD:
        st.success("Authenticated.")

        logs = load_visitor_logs(limit=200)
        st.metric("Total visits logged", len(logs))

        if logs:
            # Summary table
            log_df = pd.DataFrame(logs)
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])

            # Extract geo fields
            log_df["country"] = log_df.get("geolocation", pd.Series([None] * len(log_df))).apply(
                lambda g: g.get("country", "") if isinstance(g, dict) else ""
            )
            log_df["city"] = log_df.get("geolocation", pd.Series([None] * len(log_df))).apply(
                lambda g: g.get("city", "") if isinstance(g, dict) else ""
            )
            log_df["isp"] = log_df.get("geolocation", pd.Series([None] * len(log_df))).apply(
                lambda g: g.get("isp", "") if isinstance(g, dict) else ""
            )

            st.subheader("Recent Visits")
            display_df = log_df[["timestamp", "page", "ip", "country", "city", "isp"]].copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(display_df, use_container_width=True, height=500)

            # Stats
            st.subheader("Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Visits by page**")
                page_counts = log_df["page"].value_counts()
                st.bar_chart(page_counts)
            with col2:
                st.markdown("**Visits by country**")
                country_counts = log_df["country"].value_counts()
                if len(country_counts) > 0:
                    st.bar_chart(country_counts)
                else:
                    st.info("No geolocation data yet.")

            # Unique IPs
            unique_ips = log_df["ip"].nunique()
            st.metric("Unique visitors (by IP)", unique_ips)
        else:
            st.info("No visits logged yet.")

    elif pwd:
        st.error("Wrong password.")

elif page == "Overview":
    log_visit("Overview")
    st.title("MMM Real Estate — Overview")

    nat = load_national()
    reg = load_regional()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", f"{nat['leads'].sum():,}", help="Total number of qualified leads generated across all regions and weeks. A lead is a potential customer who submitted a contact request to a real estate agency.")
    col2.metric("Total Downloads", f"{nat['app_downloads'].sum():,}", help="Total mobile app downloads over the period. Downloads are correlated with leads but also driven by digital campaigns (especially Google Ads).")
    total_spend = sum(nat[f"spend_{ch}"].sum() for ch in MEDIA_CHANNELS)
    col3.metric("Total Media Spend", format_euros(total_spend), help="Sum of all media investment across TV, Radio, RATP Display, and Google Ads over the entire period (2020–2025).")
    col4.metric("Avg Weekly Leads", f"{nat['leads'].mean():,.0f}", help="Average number of leads generated per week at the national level. Useful as a baseline to compare simulator predictions against.")

    # Temporal evolution
    st.subheader("Weekly Leads & App Downloads", help="Time series of national weekly leads and app downloads. Dips correspond to COVID lockdowns (2020), interest rate hikes (2023), and summer seasonality.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nat["date"], y=nat["leads"], name="Leads", mode="lines"))
    fig.add_trace(go.Scatter(x=nat["date"], y=nat["app_downloads"], name="App Downloads", mode="lines", opacity=0.7))
    fig.update_layout(xaxis_title="Date", yaxis_title="Count", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**How to read this chart:** Two time series plotted on the same axis. "
               "Leads (blue) represent qualified contacts; App Downloads (orange) follow a similar trend but at higher volume. "
               "Look for sharp dips (COVID lockdowns in spring 2020, interest rate shock in 2023) and seasonal patterns "
               "(summer dips in July–August, year-end slowdowns in December).")

    # Media spend over time
    st.subheader("Weekly Media Spend by Channel", help="Stacked area chart of weekly media investment by channel. TV runs in 4 monthly bursts (Jan, Apr, Sep, Nov), Radio in 6 bursts, RATP/Bus in 4 bursts, and Google Ads runs year-round.")
    spend_fig = go.Figure()
    for ch in MEDIA_CHANNELS:
        spend_fig.add_trace(go.Scatter(
            x=nat["date"], y=nat[f"spend_{ch}"],
            name=CHANNEL_LABELS[ch], mode="lines", stackgroup="one",
        ))
    spend_fig.update_layout(xaxis_title="Date", yaxis_title="Spend (€)", height=400)
    st.plotly_chart(spend_fig, use_container_width=True)
    st.caption("**How to read this chart:** Stacked area chart — the total height at any point represents combined weekly media spend. "
               "Each colored band is one channel. TV appears in sharp monthly bursts (Jan, Apr, Sep, Nov — like real BFM-style campaigns), "
               "Radio runs 6 bursts per year, RATP/Bus 4 bursts, while Google Ads forms a continuous always-on baseline "
               "with mild seasonal variation (spring boost, slight summer dip).")

    # Regional breakdown
    st.subheader("Leads by Region", help="Total leads per region over the entire period. Île-de-France dominates (~25% of national volume), followed by Auvergne-Rhône-Alpes and PACA.")
    region_totals = reg.groupby("region")["leads"].sum().sort_values(ascending=True)
    fig_reg = px.bar(x=region_totals.values, y=region_totals.index, orientation="h", labels={"x": "Total Leads", "y": "Region"})
    fig_reg.update_layout(height=500)
    st.plotly_chart(fig_reg, use_container_width=True)
    st.caption("**How to read this chart:** Horizontal bar chart ranking regions by total lead volume. "
               "Bar length is proportional to leads generated over the entire 2020–2025 period. "
               "Île-de-France leads by far (~25% of national volume), reflecting both population density "
               "and the exclusive impact of RATP Display advertising in that region.")

    # Macro variables
    st.subheader("Macro Environment", help="Key external factors affecting real estate demand. Interest rates rose sharply in 2022-2023, depressing buyer appetite. The Pinel tax incentive was phased out from 2023 to 2025.")
    macro_fig = go.Figure()
    macro_fig.add_trace(go.Scatter(x=nat["date"], y=nat["interest_rate_20y"], name="Interest Rate 20y (%)", yaxis="y1"))
    macro_fig.add_trace(go.Scatter(x=nat["date"], y=nat["pinel"], name="Pinel Coefficient", yaxis="y2"))
    macro_fig.update_layout(
        yaxis=dict(title="Interest Rate (%)"),
        yaxis2=dict(title="Pinel", overlaying="y", side="right"),
        height=350,
    )
    st.plotly_chart(macro_fig, use_container_width=True)
    st.caption("**How to read this chart:** Dual-axis line chart. The left axis (blue) shows the 20-year mortgage rate — "
               "note the steep climb from ~1.2% in 2020 to ~4% in 2023, which suppressed buyer demand. "
               "The right axis (red dashed) shows the Pinel tax incentive coefficient: 1.0 = full incentive (2020–2022), "
               "declining to 0 by 2025 as the scheme was phased out. Both factors negatively impacted lead generation.")


# ── Page: Channel Contributions ────────────────────────────────────────────────

elif page == "Channel Contributions":
    log_visit("Channel Contributions")
    st.title("Channel Contributions")

    results = load_model_results()
    preds = load_predictions()

    if results is None:
        st.warning("Model not fitted yet. Run `python -m src.mmm_model` first.")
        st.stop()

    # Contribution percentages
    total_contrib = results["total_contributions"]
    total_all = sum(total_contrib.values())

    st.subheader("Total Contribution by Channel", help="Number of incremental leads attributed to each media channel by the model, after removing baseline and control variable effects.")
    contrib_df = pd.DataFrame({
        "Channel": [CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS],
        "Leads Contributed": [total_contrib[ch] for ch in MEDIA_CHANNELS],
        "Share (%)": [100 * total_contrib[ch] / total_all for ch in MEDIA_CHANNELS],
    })
    st.dataframe(contrib_df.style.format({"Leads Contributed": "{:,.0f}", "Share (%)": "{:.1f}"}), use_container_width=True)

    # Waterfall chart
    st.subheader("Contribution Waterfall", help="Waterfall chart showing how each channel incrementally adds to the total media-driven leads. Each bar represents the marginal contribution of one channel.")
    waterfall_fig = go.Figure(go.Waterfall(
        name="Contributions",
        orientation="v",
        measure=["relative"] * len(MEDIA_CHANNELS) + ["total"],
        x=[CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS] + ["Total"],
        y=[total_contrib[ch] for ch in MEDIA_CHANNELS] + [0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    waterfall_fig.update_layout(height=400)
    st.plotly_chart(waterfall_fig, use_container_width=True)
    st.caption("**How to read this chart:** Each bar shows the incremental lead contribution of one channel. "
               "Bars stack upward from left to right; the final 'Total' bar shows the combined media-driven leads. "
               "A taller bar means the channel contributed more incremental leads over the period.")

    # ROAS
    st.subheader("ROAS by Channel", help="Return On Ad Spend — measures the number of incremental leads generated per euro spent. Higher ROAS means better cost-efficiency. Calculated as total attributed leads divided by total spend for each channel.")
    roas_df = pd.DataFrame({
        "Channel": [CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS],
        "Total Spend (€)": [results["total_spend"][ch] for ch in MEDIA_CHANNELS],
        "Total Leads": [total_contrib[ch] for ch in MEDIA_CHANNELS],
        "ROAS (leads/€)": [results["roas"][ch] for ch in MEDIA_CHANNELS],
    })
    st.dataframe(roas_df.style.format({
        "Total Spend (€)": "{:,.0f}",
        "Total Leads": "{:,.0f}",
        "ROAS (leads/€)": "{:.4f}",
    }), use_container_width=True)

    # Contributions over time
    if preds is not None:
        st.subheader("Channel Contributions Over Time")
        contrib_fig = go.Figure()
        for ch in MEDIA_CHANNELS:
            col = f"contrib_{ch}"
            if col in preds.columns:
                contrib_fig.add_trace(go.Scatter(
                    x=preds["date"], y=preds[col],
                    name=CHANNEL_LABELS[ch], stackgroup="one",
                ))
        contrib_fig.update_layout(xaxis_title="Date", yaxis_title="Lead Contributions", height=400)
        st.plotly_chart(contrib_fig, use_container_width=True)
        st.caption("**How to read this chart:** Stacked area showing each channel's weekly lead contribution over time. "
                   "The total height represents all media-attributed leads for that week. "
                   "Notice how TV and Radio contributions persist between bursts thanks to adstock (brand memory effect), "
                   "while Google Ads contributions follow spend almost instantly.")

    # Model fit
    if preds is not None:
        st.subheader("Model Fit: Actual vs Predicted")
        fit_fig = go.Figure()
        fit_fig.add_trace(go.Scatter(x=preds["date"], y=preds["y_actual"], name="Actual", mode="lines"))
        fit_fig.add_trace(go.Scatter(x=preds["date"], y=preds["y_pred"], name="Predicted", mode="lines", line=dict(dash="dash")))
        fit_fig.update_layout(height=400)
        st.plotly_chart(fit_fig, use_container_width=True)
        st.caption("**How to read this chart:** The solid line is the actual observed weekly leads; "
                   "the dashed line is the model's prediction. The closer the two lines track each other, "
                   "the better the model fit. Persistent gaps indicate periods where external factors "
                   "not captured by the model may be at play (R² ≈ 0.74).")


# ── Page: Response Curves ──────────────────────────────────────────────────────

elif page == "Response Curves":
    log_visit("Response Curves")
    st.title("Response Curves (Saturation)")

    st.info(
        "**How to read these charts:** Each channel has two graphs.\n\n"
        "**Saturation Curve** (top): shows how the media effect grows as you increase weekly spend. "
        "It rises steeply at first (good return on investment) then flattens (diminishing returns — "
        "spending more brings less and less). The **dots** show where your actual historical weekly budgets "
        "fell on the curve — if most dots are in the flat zone, you may be overspending on that channel.\n\n"
        "**Marginal Return** (bottom): shows how much additional effect each extra euro brings. "
        "A declining curve means you are entering the saturation zone — each additional euro is less effective."
    )

    st.warning(
        "**Note:** These curves use the logistic saturation function from the Bayesian model (not a simple Hill curve). "
        "The shape is learned from the data during training. On synthetic data, the exact shapes are illustrative — "
        "on real campaign data, these curves would reflect actual diminishing returns per channel."
    )

    nat = load_national()

    for ch in MEDIA_CHANNELS:
        st.subheader(CHANNEL_LABELS[ch])
        col_spend = f"spend_{ch}"
        max_spend = nat[col_spend].max()
        x_range = np.linspace(0, max_spend * 1.5, 200)

        # Adstock (single week approximation)
        # Saturation curve
        y_sat = hill_saturation(x_range, SATURATION_ALPHA[ch])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=y_sat, mode="lines", name="Saturation Curve"))

        # Actual spend points
        actual_spend = nat[col_spend].values
        actual_sat = hill_saturation(actual_spend, SATURATION_ALPHA[ch])
        fig.add_trace(go.Scatter(x=actual_spend, y=actual_sat, mode="markers", name="Actual Weeks", opacity=0.3, marker=dict(size=4)))

        fig.update_layout(
            xaxis_title="Weekly Spend (€)",
            yaxis_title="Saturated Effect",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Marginal return
        if max_spend > 0:
            dx = x_range[1] - x_range[0]
            marginal = np.gradient(y_sat, dx)
            marg_fig = go.Figure()
            marg_fig.add_trace(go.Scatter(x=x_range, y=marginal, mode="lines", name="Marginal Return"))
            marg_fig.update_layout(xaxis_title="Weekly Spend (€)", yaxis_title="Marginal Return", height=250)
            st.plotly_chart(marg_fig, use_container_width=True)


# ── Page: Budget Simulator ─────────────────────────────────────────────────────

elif page == "Budget Simulator":
    log_visit("Budget Simulator")
    st.title("Budget Simulator")

    results = load_model_results()
    bayesian = load_bayesian_posteriors()
    nat = load_national()

    if results is None:
        st.warning("Model not fitted yet. Run `python -m src.mmm_model` first.")
        st.stop()

    if bayesian:
        st.info("**Advanced model loaded** — predictions include a confidence range: instead of one number, you get a min-max range that covers 94% of likely outcomes.")
    else:
        st.warning("**Simple model** — shows a single estimate without confidence range. Train the advanced model on Colab GPU for min-max predictions.")

    st.markdown("Adjust weekly media buy per channel. TV, Radio and RATP have fixed annual production costs (creative, filming, printing) amortized over 52 weeks. **Google Ads is the main adjustment lever** — no fixed cost, instant effect.")

    # Annual fixed production costs (creative/production)
    FIXED_ANNUAL_COST = {
        "tv": 50_000,       # filming the ad
        "radio": 15_000,    # recording the spot
        "ratp_display": 20_000,  # design + printing
        "google_ads": 0,    # no fixed cost
    }
    fixed_weekly = {ch: FIXED_ANNUAL_COST[ch] / 52 for ch in MEDIA_CHANNELS}

    # Show fixed costs
    with st.expander("Fixed annual production costs (amortized weekly)"):
        fc_cols = st.columns(len(MEDIA_CHANNELS))
        for i, ch in enumerate(MEDIA_CHANNELS):
            fc_cols[i].metric(CHANNEL_LABELS[ch], f"{FIXED_ANNUAL_COST[ch]:,}€/yr", delta=f"{fixed_weekly[ch]:,.0f}€/wk")

    # Compute defaults from average spend
    avg_spend = {ch: nat[f"spend_{ch}"].mean() for ch in MEDIA_CHANNELS}

    st.markdown("#### Weekly media buy (diffusion)")
    cols = st.columns(len(MEDIA_CHANNELS))
    spend_inputs = {}
    step_map = {"tv": 5_000, "radio": 2_000, "ratp_display": 2_000, "google_ads": 1_000}
    for i, ch in enumerate(MEDIA_CHANNELS):
        with cols[i]:
            spend_inputs[ch] = st.slider(
                CHANNEL_LABELS[ch],
                min_value=0,
                max_value=int(nat[f"spend_{ch}"].max() * 1.4),
                value=int(avg_spend[ch]),
                step=step_map[ch],
                format="%d€",
            )

    # Prediction using fallback model coefficients
    coefficients = results["coefficients"]
    intercept = results["intercept"]

    predicted_leads = intercept
    channel_contribs = {}
    for ch in MEDIA_CHANNELS:
        spend = spend_inputs[ch]
        avg_weekly = results["total_spend"][ch] / 313
        x_norm = (spend / max(avg_weekly, 1))
        sat_val = x_norm ** SATURATION_ALPHA[ch] / (1 + x_norm ** SATURATION_ALPHA[ch])
        contrib = coefficients.get(f"sat_{ch}", 0) * sat_val
        channel_contribs[ch] = contrib
        predicted_leads += contrib

    # Add average control effects
    for ctrl in ["interest_rate_20y", "pinel", "covid_impact"]:
        avg_val = nat[ctrl].mean()
        predicted_leads += coefficients.get(ctrl, 0) * avg_val

    predicted_leads = max(0, predicted_leads)
    total_weekly_media = sum(spend_inputs.values())
    total_weekly_fixed = sum(fixed_weekly.values())
    total_weekly_spend = total_weekly_media + total_weekly_fixed

    # Bayesian Monte Carlo CI
    mc_result = None
    if bayesian:
        mc_result = _mc_simulate_leads(bayesian, spend_inputs, nat)

    # Display results
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    if mc_result:
        c1.metric("Predicted Weekly Leads", f"{mc_result['leads_mean']:,.0f}", help="Bayesian posterior mean prediction.")
        c1.caption(f"Range (94%): [{mc_result['leads_hdi_3']:,.0f} – {mc_result['leads_hdi_97']:,.0f}]")
    else:
        c1.metric("Predicted Weekly Leads", f"{predicted_leads:,.0f}", help="Ridge point estimate.")
    c2.metric("Weekly Media Buy", format_euros(total_weekly_media), help="Diffusion cost only (sliders).")
    c3.metric("Total Weekly Cost", format_euros(total_weekly_spend), help="Media buy + amortized production costs.")
    cpa_leads = mc_result["leads_mean"] if mc_result else predicted_leads
    cpa = total_weekly_spend / max(cpa_leads, 1)
    c4.metric("CPA (all-in)", format_euros(cpa), help="Total cost (media + production) per lead.")
    if mc_result:
        cpa_lo = total_weekly_spend / max(mc_result["leads_hdi_97"], 1)
        cpa_hi = total_weekly_spend / max(mc_result["leads_hdi_3"], 1)
        c4.caption(f"Range (94%): [{format_euros(cpa_lo)} – {format_euros(cpa_hi)}]")

    # Channel CPA breakdown
    st.subheader("Channel-Level Metrics")
    if mc_result:
        mc_ch = mc_result["channel_contribs"]
        sim_df = pd.DataFrame({
            "Channel": [CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS],
            "Media Buy (€/wk)": [spend_inputs[ch] for ch in MEDIA_CHANNELS],
            "Prod Cost (€/wk)": [fixed_weekly[ch] for ch in MEDIA_CHANNELS],
            "Total Cost (€/wk)": [spend_inputs[ch] + fixed_weekly[ch] for ch in MEDIA_CHANNELS],
            "Contrib (leads)": [mc_ch[ch]["mean"] for ch in MEDIA_CHANNELS],
            "Contrib HDI 3%": [mc_ch[ch]["hdi_3"] for ch in MEDIA_CHANNELS],
            "Contrib HDI 97%": [mc_ch[ch]["hdi_97"] for ch in MEDIA_CHANNELS],
            "CPA all-in (€)": [(spend_inputs[ch] + fixed_weekly[ch]) / max(mc_ch[ch]["mean"], 0.01) for ch in MEDIA_CHANNELS],
        })
        st.dataframe(sim_df.style.format({
            "Media Buy (€/wk)": "{:,.0f}",
            "Prod Cost (€/wk)": "{:,.0f}",
            "Total Cost (€/wk)": "{:,.0f}",
            "Contrib (leads)": "{:,.1f}",
            "Contrib HDI 3%": "{:,.1f}",
            "Contrib HDI 97%": "{:,.1f}",
            "CPA all-in (€)": "{:,.1f}",
        }), use_container_width=True)
    else:
        sim_df = pd.DataFrame({
            "Channel": [CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS],
            "Media Buy (€/wk)": [spend_inputs[ch] for ch in MEDIA_CHANNELS],
            "Prod Cost (€/wk)": [fixed_weekly[ch] for ch in MEDIA_CHANNELS],
            "Total Cost (€/wk)": [spend_inputs[ch] + fixed_weekly[ch] for ch in MEDIA_CHANNELS],
            "Contrib (leads)": [channel_contribs[ch] for ch in MEDIA_CHANNELS],
            "CPA all-in (€)": [(spend_inputs[ch] + fixed_weekly[ch]) / max(channel_contribs[ch], 0.01) for ch in MEDIA_CHANNELS],
        })
        st.dataframe(sim_df.style.format({
            "Media Buy (€/wk)": "{:,.0f}",
            "Prod Cost (€/wk)": "{:,.0f}",
            "Total Cost (€/wk)": "{:,.0f}",
            "Contrib (leads)": "{:,.1f}",
            "CPA all-in (€)": "{:,.1f}",
        }), use_container_width=True)

    # Pie chart of contributions
    st.subheader("Spend Allocation vs Contribution")
    pie_values = [max(0, mc_result["channel_contribs"][ch]["mean"]) if mc_result else max(0, channel_contribs[ch]) for ch in MEDIA_CHANNELS]
    pie_col1, pie_col2 = st.columns(2)
    with pie_col1:
        spend_pie = px.pie(
            values=[spend_inputs[ch] + fixed_weekly[ch] for ch in MEDIA_CHANNELS],
            names=[CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS],
            title="Total Cost Allocation (media + production)",
        )
        st.plotly_chart(spend_pie, use_container_width=True)
    with pie_col2:
        contrib_pie = px.pie(
            values=pie_values,
            names=[CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS],
            title="Lead Contributions",
        )
        st.plotly_chart(contrib_pie, use_container_width=True)
    st.caption("**How to read these charts:** Compare the two pie charts side by side. "
               "The left pie shows how your budget is split across channels; the right pie shows "
               "how the predicted leads are distributed.")


# ── Page: Forecast ─────────────────────────────────────────────────────────────

elif page == "Forecast":
    log_visit("Forecast")
    st.title("4-Week Forecast")

    results = load_model_results()
    bayesian = load_bayesian_posteriors()
    nat = load_national()

    if results is None:
        st.warning("Model not fitted yet. Run `python -m src.mmm_model` first.")
        st.stop()

    st.info(
        "**How to use:** Set the planned weekly budget per channel for each of the next 4 weeks. "
        "The model predicts leads and app downloads week by week, and highlights which channels "
        "to prioritize or scale back based on current saturation levels."
    )
    if bayesian:
        st.info("**Advanced model loaded** — forecasts include a confidence range (min-max covering 94% of likely outcomes).")
    else:
        st.warning(
            "**Simple model** — shows a single estimate without confidence range. "
            "Train the advanced model on Colab GPU for min-max predictions."
        )

    coefficients = results["coefficients"]
    intercept = results["intercept"]

    # Seasonality: get next 4 weeks from last date in data
    import datetime
    last_date = pd.to_datetime(nat["date"].max())
    forecast_weeks = [last_date + pd.Timedelta(weeks=i + 1) for i in range(4)]

    # Seasonal factors (simple approximation)
    def _seasonal_factor(d):
        month, week = d.month, d.isocalendar()[1]
        base = 1.0 + 0.15 * np.sin(2 * np.pi * (week - 13) / 52)
        if month in (7, 8):
            base *= 0.90
        if month == 12:
            base *= 0.92
        return base

    seasonal_factors = [_seasonal_factor(d) for d in forecast_weeks]

    # Input: budget per channel per week
    st.subheader("Planned Weekly Budgets")

    # Option to set uniform or per-week budgets
    uniform = st.checkbox("Same budget every week", value=True)

    avg_spend = {ch: nat[f"spend_{ch}"].mean() for ch in MEDIA_CHANNELS}
    step_map = {"tv": 5_000, "radio": 2_000, "ratp_display": 2_000, "google_ads": 1_000}

    week_budgets = []
    if uniform:
        cols = st.columns(len(MEDIA_CHANNELS))
        base_budget = {}
        for i, ch in enumerate(MEDIA_CHANNELS):
            with cols[i]:
                base_budget[ch] = st.slider(
                    CHANNEL_LABELS[ch],
                    min_value=0,
                    max_value=int(nat[f"spend_{ch}"].max() * 1.4),
                    value=int(avg_spend[ch]),
                    step=step_map[ch],
                    format="%d€",
                    key=f"fc_uniform_{ch}",
                )
        week_budgets = [base_budget] * 4
    else:
        for w in range(4):
            st.markdown(f"**Week {w + 1}** — {forecast_weeks[w].strftime('%d %b %Y')}")
            cols = st.columns(len(MEDIA_CHANNELS))
            wb = {}
            for i, ch in enumerate(MEDIA_CHANNELS):
                with cols[i]:
                    wb[ch] = st.slider(
                        CHANNEL_LABELS[ch],
                        min_value=0,
                        max_value=int(nat[f"spend_{ch}"].max() * 1.4),
                        value=int(avg_spend[ch]),
                        step=step_map[ch],
                        format="%d€",
                        key=f"fc_w{w}_{ch}",
                    )
            week_budgets.append(wb)

    # Predict each week
    st.markdown("---")
    st.subheader("Predicted Leads per Week")

    forecast_rows = []
    fc_hdi_3_list = []
    fc_hdi_97_list = []

    for w in range(4):
        budget = week_budgets[w]
        predicted = intercept
        week_contribs = {}

        for ch in MEDIA_CHANNELS:
            spend = budget[ch]
            avg_weekly = results["total_spend"][ch] / 313
            x_norm = spend / max(avg_weekly, 1)
            sat_val = x_norm ** SATURATION_ALPHA[ch] / (1 + x_norm ** SATURATION_ALPHA[ch])
            contrib = coefficients.get(f"sat_{ch}", 0) * sat_val
            week_contribs[ch] = contrib
            predicted += contrib

        # Controls at recent average
        for ctrl in ["interest_rate_20y", "pinel", "covid_impact"]:
            recent_avg = nat[ctrl].iloc[-13:].mean()
            predicted += coefficients.get(ctrl, 0) * recent_avg

        # Apply seasonality
        predicted *= seasonal_factors[w]
        predicted = max(0, predicted)

        # Bayesian CI for this week
        mc_w = None
        if bayesian:
            mc_w = _mc_simulate_leads(bayesian, budget, nat)
            mc_leads = mc_w["leads_mean"] * seasonal_factors[w]
            mc_lo = mc_w["leads_hdi_3"] * seasonal_factors[w]
            mc_hi = mc_w["leads_hdi_97"] * seasonal_factors[w]
            fc_hdi_3_list.append(max(0, mc_lo))
            fc_hdi_97_list.append(mc_hi)
            predicted = max(0, mc_leads)

        # Downloads estimate
        digital_boost = week_contribs.get("google_ads", 0) * 0.5
        downloads = int(predicted * 2.1 + digital_boost)

        row = {
            "Week": f"W{w+1} — {forecast_weeks[w].strftime('%d %b')}",
            "Leads": int(predicted),
            "App Downloads": downloads,
            "Total Spend": sum(budget.values()),
            "Seasonality": f"{seasonal_factors[w]:.2f}",
            **{f"{CHANNEL_LABELS[ch]} contrib": int(week_contribs[ch] * seasonal_factors[w]) for ch in MEDIA_CHANNELS},
        }
        if mc_w:
            row["Leads HDI 3%"] = int(fc_hdi_3_list[-1])
            row["Leads HDI 97%"] = int(fc_hdi_97_list[-1])

        forecast_rows.append(row)

    fc_df = pd.DataFrame(forecast_rows)
    fmt_dict = {"Leads": "{:,}", "App Downloads": "{:,}", "Total Spend": "{:,}€"}
    if bayesian:
        fmt_dict["Leads HDI 3%"] = "{:,}"
        fmt_dict["Leads HDI 97%"] = "{:,}"
    st.dataframe(fc_df.style.format(fmt_dict), use_container_width=True)

    # Weekly chart with error bars
    fc_fig = go.Figure()
    if bayesian and fc_hdi_3_list:
        fc_fig.add_trace(go.Bar(
            x=fc_df["Week"], y=fc_df["Leads"], name="Predicted Leads",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[fc_hdi_97_list[i] - fc_df["Leads"].iloc[i] for i in range(4)],
                arrayminus=[fc_df["Leads"].iloc[i] - fc_hdi_3_list[i] for i in range(4)],
            ),
        ))
    else:
        fc_fig.add_trace(go.Bar(x=fc_df["Week"], y=fc_df["Leads"], name="Predicted Leads"))
    fc_fig.add_trace(go.Scatter(x=fc_df["Week"], y=fc_df["App Downloads"], name="App Downloads", mode="lines+markers", yaxis="y2"))
    fc_fig.update_layout(
        yaxis=dict(title="Leads"),
        yaxis2=dict(title="Downloads", overlaying="y", side="right"),
        height=400,
    )
    st.plotly_chart(fc_fig, use_container_width=True)
    st.caption("**How to read this chart:** Bars show predicted weekly leads (error bars = 94% confidence range if advanced model is loaded), "
               "the line shows predicted app downloads.")

    # Recommendations
    st.subheader("Recommendations")

    # Compute marginal efficiency for each channel
    efficiency = {}
    for ch in MEDIA_CHANNELS:
        avg_weekly = results["total_spend"][ch] / 313
        current_spend = week_budgets[0][ch]
        x_norm = current_spend / max(avg_weekly, 1)
        sat_val = x_norm ** SATURATION_ALPHA[ch] / (1 + x_norm ** SATURATION_ALPHA[ch])
        # Marginal: derivative of Hill at current point
        marginal = SATURATION_ALPHA[ch] * x_norm ** (SATURATION_ALPHA[ch] - 1) / (1 + x_norm ** SATURATION_ALPHA[ch]) ** 2
        coef = abs(coefficients.get(f"sat_{ch}", 0))
        efficiency[ch] = marginal * coef / max(avg_weekly, 1)

    sorted_channels = sorted(efficiency.items(), key=lambda x: x[1], reverse=True)
    best_ch = sorted_channels[0][0]
    worst_ch = sorted_channels[-1][0]

    if bayesian:
        best_roas = bayesian["channels"].get(best_ch, {}).get("roas", {})
        st.success(f"**Increase** {CHANNEL_LABELS[best_ch]} — highest marginal return. "
                   f"ROAS: {best_roas.get('mean', 0):.5f} [{best_roas.get('hdi_3', 0):.5f} – {best_roas.get('hdi_97', 0):.5f}]")
    else:
        st.success(f"**Increase** {CHANNEL_LABELS[best_ch]} — highest marginal return at current spend level.")
    st.error(f"**Consider reducing** {CHANNEL_LABELS[worst_ch]} — lowest marginal return. "
             f"This channel is closer to saturation at the current budget level.")

    for ch, eff in sorted_channels:
        saturation_pct = week_budgets[0][ch] / max(nat[f"spend_{ch}"].max(), 1) * 100
        bar_label = "Low" if saturation_pct < 40 else ("Medium" if saturation_pct < 70 else "High")
        st.markdown(f"- **{CHANNEL_LABELS[ch]}**: marginal efficiency = {eff:.6f} — saturation level: **{bar_label}** ({saturation_pct:.0f}% of historical max)")


# ── Page: Goal Planner ─────────────────────────────────────────────────────────

elif page == "Goal Planner":
    log_visit("Goal Planner")
    st.title("Goal Planner — Reverse Budget Optimization")

    bayesian = load_bayesian_posteriors()
    results = load_model_results()
    nat = load_national()

    if not bayesian:
        st.warning(
            "**Bayesian model required** — The Goal Planner needs Bayesian posteriors to provide "
            "reliable saturation-aware optimization with confidence intervals.\n\n"
            "Fit the Bayesian model on Colab GPU, then copy `bayesian_posteriors.json` into `data/model/`."
        )
        st.markdown("""
### How it will work

1. **Set your target** — Enter the number of weekly leads you want to achieve
2. **Greedy optimization** — The algorithm fills the most cost-efficient channel first,
   then spills into the next best channel when saturation kicks in
3. **Budget output** — Optimal weekly budget per channel, expected leads, CPA breakdown with CI
""")
        st.stop()

    if results is None:
        st.warning("Model not fitted yet.")
        st.stop()

    st.info("**Advanced model loaded** — optimization accounts for diminishing returns and includes a confidence range on all predictions.")

    st.markdown(
        "Enter your weekly lead target. The greedy optimizer allocates budget to the channel "
        "with the highest marginal ROI at each step, respecting saturation (Hill curve from posteriors)."
    )

    # Inputs
    avg_leads = nat["leads"].mean()
    target_leads = st.number_input(
        "Target weekly leads",
        min_value=100,
        max_value=50_000,
        value=int(avg_leads * 1.2),
        step=500,
    )

    # Compute baseline (intercept + controls)
    intercept_mean = bayesian.get("intercept", {}).get("mean", results["intercept"])
    baseline = intercept_mean
    for ctrl in ["interest_rate_20y", "pinel", "covid_impact"]:
        ctrl_post = bayesian.get("control_coefficients", {}).get(ctrl, {})
        baseline += ctrl_post.get("mean", results["coefficients"].get(ctrl, 0)) * nat[ctrl].mean()
    baseline = max(0, baseline)

    leads_needed = target_leads - baseline
    st.caption(f"Baseline (intercept + controls): {baseline:,.0f} leads → media must generate {max(0, leads_needed):,.0f} additional leads")

    if leads_needed <= 0:
        st.success(f"Target of {target_leads:,} leads is already met by baseline demand alone. No media spend needed.")
        st.stop()

    # Greedy optimizer
    step_size = 1_000  # €1k increments
    max_budget_per_channel = {ch: int(nat[f"spend_{ch}"].max() * 3) for ch in MEDIA_CHANNELS}
    allocated = {ch: 0 for ch in MEDIA_CHANNELS}
    total_contrib = {ch: 0.0 for ch in MEDIA_CHANNELS}
    ch_data = bayesian["channels"]

    leads_so_far = 0.0
    max_iterations = 500

    for _ in range(max_iterations):
        if leads_so_far >= leads_needed:
            break

        # Find channel with best marginal return at current allocation
        best_ch = None
        best_marginal = -1.0

        for ch in MEDIA_CHANNELS:
            if allocated[ch] >= max_budget_per_channel[ch]:
                continue

            ch_info = ch_data.get(ch, {})
            coef = ch_info.get("coefficient", {}).get("mean", abs(results["coefficients"].get(f"sat_{ch}", 0)))
            alpha = ch_info.get("adstock_alpha", {}).get("mean", SATURATION_ALPHA[ch])

            avg_weekly = nat[f"spend_{ch}"].mean()
            current_spend = allocated[ch]
            next_spend = current_spend + step_size

            x_now = current_spend / max(avg_weekly, 1)
            x_next = next_spend / max(avg_weekly, 1)

            sat_now = x_now ** alpha / (1 + x_now ** alpha)
            sat_next = x_next ** alpha / (1 + x_next ** alpha)

            marginal_leads = coef * (sat_next - sat_now)
            marginal_roi = marginal_leads / step_size

            if marginal_roi > best_marginal:
                best_marginal = marginal_roi
                best_ch = ch

        if best_ch is None or best_marginal <= 0:
            break

        allocated[best_ch] += step_size
        ch_info = ch_data.get(best_ch, {})
        coef = ch_info.get("coefficient", {}).get("mean", abs(results["coefficients"].get(f"sat_{best_ch}", 0)))
        alpha = ch_info.get("adstock_alpha", {}).get("mean", SATURATION_ALPHA[best_ch])
        avg_weekly = nat[f"spend_{best_ch}"].mean()
        x = allocated[best_ch] / max(avg_weekly, 1)
        total_contrib[best_ch] = coef * (x ** alpha / (1 + x ** alpha))
        leads_so_far = sum(total_contrib.values())

    total_media_leads = leads_so_far
    total_predicted = baseline + total_media_leads
    total_budget = sum(allocated.values())

    # Display results
    st.markdown("---")
    st.subheader("Optimal Budget Allocation")

    success = total_predicted >= target_leads
    if success:
        st.success(f"Target of {target_leads:,} leads is achievable with {format_euros(total_budget)} weekly media spend.")
    else:
        st.error(f"Target of {target_leads:,} leads could not be fully reached. "
                 f"Maximum achievable: {total_predicted:,.0f} leads with {format_euros(total_budget)} spend.")

    # Monte Carlo CI on the optimized budget
    mc_opt = _mc_simulate_leads(bayesian, allocated, nat)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Leads", f"{mc_opt['leads_mean']:,.0f}")
    c1.caption(f"Range (94%): [{mc_opt['leads_hdi_3']:,.0f} – {mc_opt['leads_hdi_97']:,.0f}]")
    c2.metric("Total Weekly Spend", format_euros(total_budget))
    cpa_opt = total_budget / max(mc_opt["leads_mean"], 1)
    c3.metric("Blended CPA", format_euros(cpa_opt))
    # Probability of reaching target
    prob_reach = int(100 * (1 - (target_leads - mc_opt["leads_mean"]) / max(mc_opt["leads_hdi_97"] - mc_opt["leads_hdi_3"], 1))) if mc_opt["leads_mean"] >= target_leads else max(0, int(50 * (mc_opt["leads_hdi_97"] - target_leads) / max(mc_opt["leads_hdi_97"] - mc_opt["leads_mean"], 1)))
    prob_reach = min(99, max(1, prob_reach))
    c4.metric("P(reach target)", f"{prob_reach}%")

    # Per-channel breakdown
    st.subheader("Budget per Channel")
    goal_df = pd.DataFrame({
        "Channel": [CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS],
        "Weekly Budget (€)": [allocated[ch] for ch in MEDIA_CHANNELS],
        "Contrib (leads)": [mc_opt["channel_contribs"][ch]["mean"] for ch in MEDIA_CHANNELS],
        "Contrib HDI 3%": [mc_opt["channel_contribs"][ch]["hdi_3"] for ch in MEDIA_CHANNELS],
        "Contrib HDI 97%": [mc_opt["channel_contribs"][ch]["hdi_97"] for ch in MEDIA_CHANNELS],
        "CPA (€/lead)": [allocated[ch] / max(mc_opt["channel_contribs"][ch]["mean"], 0.01) for ch in MEDIA_CHANNELS],
    })
    st.dataframe(goal_df.style.format({
        "Weekly Budget (€)": "{:,.0f}",
        "Contrib (leads)": "{:,.1f}",
        "Contrib HDI 3%": "{:,.1f}",
        "Contrib HDI 97%": "{:,.1f}",
        "CPA (€/lead)": "{:,.1f}",
    }), use_container_width=True)

    # Pie chart
    goal_pie_col1, goal_pie_col2 = st.columns(2)
    with goal_pie_col1:
        fig_alloc = px.pie(
            values=[allocated[ch] for ch in MEDIA_CHANNELS],
            names=[CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS],
            title="Budget Allocation",
        )
        st.plotly_chart(fig_alloc, use_container_width=True)
    with goal_pie_col2:
        fig_contrib = px.pie(
            values=[max(0, mc_opt["channel_contribs"][ch]["mean"]) for ch in MEDIA_CHANNELS],
            names=[CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS],
            title="Lead Contributions",
        )
        st.plotly_chart(fig_contrib, use_container_width=True)



# ── Page: Model Validation ────────────────────────────────────────────────────

elif page == "Model Validation":
    log_visit("Model Validation")
    st.title("Model Validation — 3-Month Forecast Backtest")

    st.markdown("""
**How reliable are the predictions?** To answer this, we train the model on older data and test it on
the most recent 3 months (October - December 2025) — data the model has never seen.

**Quelle fiabilite pour les predictions ?** Pour repondre, on entraine le modele sur les donnees anciennes
et on le teste sur les 3 derniers mois (octobre - decembre 2025) — des donnees inconnues du modele.
""")

    import plotly.graph_objects as go
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error

    nat = load_national()

    # Split: train up to end Sep 2025, test = Oct-Dec 2025
    train_mask = nat["date"] < "2025-10-01"
    test_mask = nat["date"] >= "2025-10-01"
    train_df = nat[train_mask]
    test_df = nat[test_mask]

    control_cols = ["interest_rate_20y", "pinel", "covid_impact"]

    # Build features on full data (adstock needs history)
    features_all = {}
    for ch in MEDIA_CHANNELS:
        raw = nat[f"spend_{ch}"].values
        adstocked = geometric_adstock(raw, ADSTOCK_DECAY[ch])
        saturated = hill_saturation(adstocked, SATURATION_ALPHA[ch])
        features_all[f"sat_{ch}"] = saturated
    for col in control_cols:
        features_all[col] = nat[col].values

    X_all = pd.DataFrame(features_all)
    y_all = nat["leads"].values.astype(float)

    n_train = train_mask.sum()
    X_train, X_test = X_all.iloc[:n_train], X_all.iloc[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    # Calculate residual standard deviation from training set
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    residual_std = np.std(residuals)

    # Standard error grows with forecast horizon (sqrt of weeks ahead)
    n_test = len(y_test)
    weeks_ahead = np.arange(1, n_test + 1)
    se = residual_std * np.sqrt(weeks_ahead) / np.sqrt(n_train)
    sd = residual_std * np.ones(n_test)

    # 95% confidence interval (1.96 * SD for prediction interval)
    ci_95_upper = y_pred_test + 1.96 * sd
    ci_95_lower = y_pred_test - 1.96 * sd

    # Growing uncertainty band (SE-based, widens over time)
    ci_growing_upper = y_pred_test + 1.96 * residual_std * np.sqrt(weeks_ahead / weeks_ahead[0])
    ci_growing_lower = y_pred_test - 1.96 * residual_std * np.sqrt(weeks_ahead / weeks_ahead[0])

    test_dates = test_df["date"].values

    # Metrics
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    mae = mean_absolute_error(y_test, y_pred_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test period", f"{len(y_test)} weeks")
    m2.metric("Avg error (MAPE)", f"{mape:.1f}%")
    m3.metric("Avg absolute error", f"{mae:,.0f} leads/wk")
    m4.metric("Residual Std Dev", f"{residual_std:,.0f} leads")

    st.markdown("---")

    # Chart: Actual vs Predicted with confidence bands
    fig = go.Figure()

    # 95% confidence band (growing uncertainty)
    fig.add_trace(go.Scatter(
        x=np.concatenate([test_dates, test_dates[::-1]]),
        y=np.concatenate([ci_growing_upper, ci_growing_lower[::-1]]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Interval (widens over time)",
        showlegend=True,
    ))

    # Fixed SD band
    fig.add_trace(go.Scatter(
        x=np.concatenate([test_dates, test_dates[::-1]]),
        y=np.concatenate([ci_95_upper, ci_95_lower[::-1]]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.08)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Prediction Interval (fixed SD)",
        showlegend=True,
    ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_pred_test,
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#636EFA", dash="dash"),
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_test,
        mode="lines+markers",
        name="Actual",
        line=dict(color="#EF553B"),
    ))

    fig.update_layout(
        title="3-Month Backtest: Actual vs Predicted Leads (Oct-Dec 2025)",
        xaxis_title="Date",
        yaxis_title="Weekly Leads",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**How to read this chart:** The red line shows actual weekly leads. The blue dashed line shows "
        "the model's prediction. The light blue band is the 95% confidence interval — we expect the actual "
        "value to fall inside this band 95% of the time. Notice how the band widens over time: "
        "the further ahead we predict, the less certain we are."
    )

    # Weekly detail table
    st.subheader("Week-by-Week Detail")
    detail_df = pd.DataFrame({
        "Week": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in test_dates],
        "Actual": y_test.astype(int),
        "Predicted": y_pred_test.astype(int),
        "Error (%)": [f"{abs(a-p)/a*100:.1f}%" for a, p in zip(y_test, y_pred_test)],
        "CI 95% Low": ci_growing_lower.astype(int),
        "CI 95% High": ci_growing_upper.astype(int),
        "Inside CI?": ["Yes" if lo <= a <= hi else "No" for a, lo, hi in zip(y_test, ci_growing_lower, ci_growing_upper)],
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("What the metrics mean / Ce que signifient les metriques")
    st.markdown("""
| Metric | Meaning | Our result |
|--------|---------|------------|
| **Standard Deviation (SD)** | How spread out the model's errors are around zero. A smaller SD means more consistent predictions. | {sd:.0f} leads |
| **Standard Error (SE)** | How precise our average prediction is. SE decreases with more training data. | Grows from {se_start:.0f} to {se_end:.0f} over 3 months |
| **95% Confidence Interval** | The range where we're 95% sure the real value falls. Widens the further out we predict. | See chart above |
| **MAPE** | Average percentage error — how far off are we on average? Below 5% is excellent. | **{mape:.1f}%** |
""".format(sd=residual_std, se_start=se[0], se_end=se[-1], mape=mape))


# ── Page: Fixed Costs ─────────────────────────────────────────────────────────

elif page == "Fixed Costs":
    log_visit("Fixed Costs")
    st.title("Fixed Production Costs")

    st.markdown("""
Advertising isn't just about buying media time — before you can air a single ad, you need to **produce it**.
These are fixed costs that don't depend on how much you spend on diffusion. They are paid once per year
(or per campaign cycle) regardless of the media budget.

---

La publicite, ce n'est pas que l'achat d'espace — avant de diffuser une seule pub, il faut **la produire**.
Ce sont des couts fixes, independants du budget de diffusion. Ils sont payes une fois par an
(ou par cycle de campagne), quel que soit le budget media.
""")

    st.subheader("TV — 50 000 EUR/year")
    st.markdown("""
- **Filming the ad** (tournage) : scriptwriting, actors, crew, studio or location, post-production
- Typically shot once a year, then aired in **4 monthly bursts** (e.g. Jan, Apr, Sep, Nov) on channels like BFM
- The same creative can be reused across bursts, but a fresh version is usually produced annually
- This cost is **incompressible** — you can't run TV without producing the spot first
""")

    st.subheader("Radio — 15 000 EUR/year")
    st.markdown("""
- **Recording the spot** (enregistrement) : voiceover talent, sound design, studio time
- Much lighter production than TV — no video, no actors on set
- Aired in **6 monthly bursts** per year (e.g. Jan, Mar, May, Jul, Sep, Nov)
- Can be updated more easily mid-year if messaging changes
""")

    st.subheader("RATP / Bus — 20 000 EUR/year")
    st.markdown("""
- **Design + printing** (maquettes et impression) : graphic design, large format printing
- Posters for bus shelters, metro stations (IDF only)
- Aired in **4 monthly bursts** per year (e.g. Feb, May, Sep, Nov)
- New creative can be produced per burst, but typically 1-2 designs per year
""")

    st.subheader("Google Ads — 0 EUR fixed")
    st.markdown("""
- **No production cost** — ads are text/image based, created directly in the platform
- **100% variable** — every euro goes to media buy
- This is the **main adjustment lever**: you can scale up or down instantly, with immediate effect on leads
- Unlike TV/Radio/RATP, stopping Google Ads **immediately stops** lead generation from this channel (no adstock memory effect)
""")

    st.markdown("---")

    # Summary table
    st.subheader("Summary / Resume")
    cost_df = pd.DataFrame({
        "Channel": ["TV (BFM etc.)", "Radio", "RATP / Bus", "Google Ads"],
        "Annual Prod. Cost": ["50 000 EUR", "15 000 EUR", "20 000 EUR", "0 EUR"],
        "Weekly Amortized": ["962 EUR/wk", "288 EUR/wk", "385 EUR/wk", "0 EUR/wk"],
        "Bursts / Year": ["4 months", "6 months", "4 months", "Always-on"],
        "Flexibility": ["Low — plan months ahead", "Medium — easier to adjust", "Low — print lead times", "High — instant on/off"],
        "Adstock (memory)": ["Very high (months)", "High (months)", "Medium (weeks)", "Near zero (instant)"],
    })
    st.dataframe(cost_df, use_container_width=True, hide_index=True)

    st.info("**Key takeaway / Point cle :** Google Ads is the only channel where 100% of the budget goes to diffusion with zero production cost and instant effect. "
            "For TV, Radio and RATP, factor in the fixed production costs when evaluating true CPA — especially at low media budgets where the fixed cost weighs heavily.")


# ── Page: SL Benchmark ────────────────────────────────────────────────────────

elif page == "SL Benchmark":
    log_visit("SL Benchmark")
    st.title("SL Benchmark — Market Leader Reference")

    st.markdown("""
This page compares our model's assumptions with publicly available data from **SL**,
the leading French real estate listings platform (~14.6% market share, 21M monthly unique visitors).

Cette page compare les hypotheses de notre modele avec les donnees publiques de **SL**,
leader francais des annonces immobilieres (~14.6% de part de marche, 21M de visiteurs uniques/mois).
""")

    st.subheader("Lead Volume / Volume de leads")
    vol_cols = st.columns(4)
    vol_cols[0].metric("Leads / year", "48M", help="All contact types: calls, forms, callback requests")
    vol_cols[1].metric("Leads / month", "~4M")
    vol_cols[2].metric("Leads / day", "~130K", delta="1.5 leads/sec")
    vol_cols[3].metric("Leads per sale", "~50", help="50 leads across all portals to close 1 transaction")

    st.caption("30% of leads are never opened or processed by agencies — a major inefficiency in the funnel.")

    st.markdown("---")
    st.subheader("Annual Budget — ~20M EUR")

    budget_data = pd.DataFrame({
        "Channel": ["TV & Offline", "Google Ads (SEA)", "Social & App Install", "SEO & Content"],
        "Share": ["50%", "25%", "15%", "10%"],
        "Annual Budget": ["10M EUR", "5M EUR", "3M EUR", "2M EUR"],
        "Role": [
            "Brand awareness + Drive-to-App",
            "Capture immediate search intent",
            "Demographic targeting + Retention",
            "Long-term 'free' leads via organic traffic",
        ],
    })
    st.dataframe(budget_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Seasonality & Media Rhythm / Saisonnalite")

    season_data = pd.DataFrame({
        "Period": ["Jan-Feb", "Mar-Jun", "Jul-Aug", "Sep-Oct", "Nov-Dec"],
        "Intensity": ["Low", "VERY HIGH", "Low", "HIGH", "Medium"],
        "Budget / month": ["~1.0M EUR", "~2.5M EUR", "~0.8M EUR", "~2.2M EUR", "~1.2M EUR"],
        "Leads / month": ["~3.0M", "~5.5M", "~2.5M", "~5.0M", "~3.5M"],
        "Focus": [
            "SEO prep, retargeting",
            "Spring peak — TV wave 1 (families preparing Sept moves)",
            "Students / rentals, app installs",
            "Back-to-school — TV wave 2",
            "Sellers planning January projects",
        ],
    })
    st.dataframe(season_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Cost Benchmarks / Couts de reference")

    cost_data = pd.DataFrame({
        "Metric": [
            "CPL Google Ads (renter/buyer)",
            "CPL Google Ads (seller — high value)",
            "CPL Google Ads (investor — new builds)",
            "CPI Google App Campaigns",
            "CPI Apple Search Ads",
            "CPI Meta (Facebook/Instagram)",
            "TV budget per wave",
            "TV waves per year",
            "TV contacts per wave",
        ],
        "SL (estimated)": [
            "5-15 EUR",
            "60-120 EUR",
            "80-150 EUR",
            "2.50-6.00 EUR",
            "4.00-8.50 EUR",
            "1.50-4.00 EUR",
            "2-4M EUR",
            "3-4 waves",
            "~120-140M contacts (25-49 yr)",
        ],
        "Our model": [
            "~37 EUR (blended)",
            "N/A (single lead type)",
            "N/A",
            "N/A (not modeled separately)",
            "N/A",
            "N/A",
            "~50K EUR (4 bursts)",
            "4 bursts",
            "N/A (smaller scale)",
        ],
    })
    st.dataframe(cost_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Key Takeaways / Points cles")

    st.markdown("""
**Scale matters / L'echelle compte :**
- SL's blended CPL is ~0.42 EUR/lead thanks to massive brand awareness, SEO traffic, and economies of scale
- A mid-size player without brand recognition pays 50-200x more per lead
- Our model's CPLs (37-93 EUR) are realistic for a smaller platform

**TV is a brand machine / La TV est une machine a notoriete :**
- SL spends 10M EUR/year on TV — 50% of total budget
- TV doesn't generate leads directly, but creates the brand reflex that drives organic and app traffic months later
- The adstock (memory effect) of TV is the longest of all channels

**Google Ads is the control lever / Google Ads est le levier de controle :**
- 25% of SL's budget (5M EUR) goes to SEA
- Instant effect, no production cost, precise targeting
- But CPL varies 10x depending on lead type (renter vs seller vs investor)

**The app is the retention play / L'app est le levier de retention :**
- App users are 3-4x more active than web users
- Push notifications are free (vs paid email/SMS)
- CPI of 2.50-6 EUR per install, but only 1 in 4 installs becomes an active user
""")

    st.info("**Note:** SL data is based on public estimates and industry benchmarks (2025-2026). "
            "Actual figures may differ. Our model uses synthetic data calibrated to a mid-size player, not SL's scale.")


# ── Page: Regional Analysis ────────────────────────────────────────────────────

elif page == "Regional Analysis":
    log_visit("Regional Analysis")
    st.title("Regional Analysis")

    reg = load_regional()
    nat = load_national()

    # Region selector
    selected_year = st.selectbox("Year", sorted(reg["year"].unique()), index=len(reg["year"].unique()) - 1)
    year_data = reg[reg["year"] == selected_year]

    # Leads by region
    st.subheader(f"Total Leads by Region — {selected_year}")
    region_leads = year_data.groupby("region")["leads"].sum().sort_values(ascending=True)
    fig = px.bar(x=region_leads.values, y=region_leads.index, orientation="h", labels={"x": "Leads", "y": "Region"})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**How to read this chart:** Horizontal bar chart showing total leads for the selected year, ranked by region. "
               "Use the year selector above to compare how regional performance evolves over time. "
               "The ranking reflects both population size and local media exposure (RATP only benefits IDF).")

    # Heatmap: Region × Channel spend
    st.subheader("Media Spend Heatmap (Region × Channel)")
    heatmap_data = year_data.groupby("region")[[f"spend_{ch}" for ch in MEDIA_CHANNELS]].sum()
    heatmap_data.columns = [CHANNEL_LABELS[ch] for ch in MEDIA_CHANNELS]
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Channel", y="Region", color="Spend (€)"),
        aspect="auto",
        color_continuous_scale="YlOrRd",
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("**How to read this chart:** Each cell shows the total media spend for a given region (row) and channel (column). "
               "Darker/warmer colors indicate higher spend. Notice the RATP Display column: only Île-de-France has spend, "
               "all other regions are zero. TV and Google Ads are distributed proportionally to regional population weight.")

    # IDF focus — RATP impact
    st.subheader("Île-de-France — RATP Display Impact")
    idf = reg[reg["region"] == "Île-de-France"].copy()
    idf_fig = go.Figure()
    idf_fig.add_trace(go.Scatter(x=idf["date"], y=idf["leads"], name="IDF Leads", mode="lines"))
    idf_fig.add_trace(go.Bar(x=idf["date"], y=idf["spend_ratp_display"], name="RATP Spend (€)", yaxis="y2", opacity=0.3))
    idf_fig.update_layout(
        yaxis=dict(title="Leads"),
        yaxis2=dict(title="RATP Spend (€)", overlaying="y", side="right"),
        height=400,
    )
    st.plotly_chart(idf_fig, use_container_width=True)
    st.caption("**How to read this chart:** Dual-axis chart for Île-de-France only. "
               "The blue line (left axis) shows weekly IDF leads; the gray bars (right axis) show RATP/Bus spend. "
               "RATP runs in 4 monthly bursts per year (Feb, May, Sep, Nov). "
               "Look for correlation between spend bursts and lead uplifts in IDF.")

    # Regional performance table
    st.subheader("Regional Performance Summary")
    summary = year_data.groupby("region").agg(
        total_leads=("leads", "sum"),
        total_downloads=("app_downloads", "sum"),
        total_spend_tv=("spend_tv", "sum"),
        total_spend_radio=("spend_radio", "sum"),
        total_spend_ratp=("spend_ratp_display", "sum"),
        total_spend_google=("spend_google_ads", "sum"),
    ).reset_index()
    summary["total_spend"] = summary[["total_spend_tv", "total_spend_radio", "total_spend_ratp", "total_spend_google"]].sum(axis=1)
    summary["cpa"] = summary["total_spend"] / summary["total_leads"].clip(lower=1)
    summary = summary.sort_values("total_leads", ascending=False)
    st.dataframe(
        summary[["region", "total_leads", "total_downloads", "total_spend", "cpa"]].style.format({
            "total_leads": "{:,.0f}",
            "total_downloads": "{:,.0f}",
            "total_spend": "{:,.0f}",
            "cpa": "{:.1f}€",
        }),
        use_container_width=True,
    )
    st.caption("**How to read this table:** Each row is a region, sorted by total leads (descending). "
               "'total_spend' is the sum of all media channels for that region and year. "
               "'cpa' (Cost Per Acquisition) = total spend ÷ total leads — lower is better. "
               "Compare CPA across regions to identify where media investment is most efficient.")


# ── Page: FAQ ──────────────────────────────────────────────────────────────────

elif page == "FAQ":
    log_visit("FAQ")
    st.title("FAQ — Frequently Asked Questions")

    st.markdown("---")

    with st.expander("What is a Media Mix Model (MMM)?", expanded=True):
        st.markdown("""
A **Media Mix Model** is a statistical model that measures the impact of each media channel
(TV, Radio, Digital, Out-of-Home...) on business KPIs — here, **leads** (qualified contact requests
sent to real estate agencies) and **app downloads** (mobile application installs).

It allows you to:
- **Quantify the contribution** of each channel to both KPIs (leads & app downloads)
- **Calculate the ROAS** (Return On Ad Spend) for every investment
- **Simulate budget scenarios** to optimize media allocation
- **Isolate media effects** from external factors (seasonality, interest rates, COVID...)

Unlike digital attribution (last-click, multi-touch), MMM works on aggregated data
and captures offline media effects (TV, Radio, Out-of-Home) as well.

**The two KPIs tracked in this dashboard:**
- **Leads**: number of qualified contact requests submitted to partner real estate agencies each week. This is the primary conversion metric.
- **App Downloads**: number of weekly mobile app installs. Downloads are correlated with leads but also driven by digital campaigns (especially Google Ads) and brand awareness (TV).
""")

    with st.expander("What data is used?"):
        st.markdown("""
This POC uses **realistic synthetic data** generated from:

- **Real estate transaction volumes**: based on DVF (Demandes de Valeurs Foncières) trends —
  ~1.1M transactions/year in 2020, drop to ~870k in 2023, recovery in 2025
- **Media spend**: realistic budgets per channel (TV: 8–12M€/year, Google Ads: 5–8M€/year, etc.)
  with channel-specific seasonality patterns
- **Macro variables**: 20-year mortgage rates (real data — ~1.2% in 2020 → ~4% in 2023),
  Pinel tax incentive (phased out), COVID impact
- **13 French regions** with realistic weighting (Île-de-France ~25%, ARA ~12%, PACA ~10%...)

**Granularity**: weekly data from January 2020 to December 2025 (~313 weeks).
""")

    with st.expander("How does saturation (Hill curve) work?"):
        st.markdown("""
**Saturation** models the diminishing returns of advertising:
the first euros spent have a strong impact, then the effect tapers off.

We use a **Hill function**: `f(x) = x^α / (λ^α + x^α)`

- **α (alpha)** controls the curve shape:
  - Low α (e.g. Google Ads = 0.4) → the curve rises quickly then saturates slowly = **good returns across a wide spend range**
  - High α (e.g. TV = 0.85) → the curve saturates fast = **stronger diminishing returns**

In practice: an extra euro on Google Ads yields proportionally more than an extra euro on TV.
""")

    with st.expander("What is adstock?"):
        st.markdown("""
**Adstock** (or carry-over) models the fact that an ad continues to produce effects
**after it has aired**.

We use a **geometric decay model**: `adstock(t) = spend(t) + decay × adstock(t-1)`

- **TV** (decay = 0.7): long-lasting effect, ~3 weeks — a TV campaign keeps driving leads
  well after it stops airing
- **Radio** (decay = 0.4): short effect, ~1 week
- **RATP Display** (decay = 0.5): medium effect, ~2 weeks
- **Google Ads** (decay = 0.2): near-immediate effect — clicks convert very quickly
""")

    with st.expander("How to read ROAS?"):
        st.markdown("""
**ROAS (Return On Ad Spend)** measures the number of leads generated per euro spent.

| Channel | Typical ROAS | Interpretation |
|---------|-------------|----------------|
| Google Ads | ~0.011 | **Most efficient** — each euro generates the most leads |
| Radio | ~0.010 | Good cost-efficiency |
| RATP Display | ~0.008 | Decent, but limited to Île-de-France |
| TV | ~0.004 | **Least efficient** — high cost per lead, but strong branding reach |

**Note**: a low ROAS does not mean the channel is useless. TV drives brand awareness
and a halo effect that benefits other channels (not captured here).
""")

    with st.expander("Why is Google Ads the most profitable channel?"):
        st.markdown("""
Several factors explain Google Ads' superior profitability:

1. **Intent-based**: users are actively searching for a property → high conversion rate
2. **Slow saturation** (α = 0.4): returns remain strong even as budget increases
3. **Immediate effect** (decay = 0.2): no temporal decay — clicks convert quickly
4. **Precise targeting**: ability to target by region, intent, keyword

Conversely, **TV** has a high entry cost, saturates faster (α = 0.85), and reaches
a broad but less qualified audience.
""")

    with st.expander("How to use the Budget Simulator?"):
        st.markdown("""
The simulator lets you test budget scenarios **in real time**:

1. **Adjust the sliders** for each channel (weekly budget in €)
2. Observe the impact on **predicted leads**, **blended CPA**, and **per-channel CPA**
3. Compare the **pie charts**: budget allocation vs. lead contribution

**Tips:**
- Start by increasing Google Ads → it's the most efficient lever
- Reduce TV to see the impact → the drop in leads is moderate
- RATP Display only affects Île-de-France
- The Google Ads slider has finer steps (1,000€ increments) for precise tuning
""")

    with st.expander("Why does RATP Display only impact Île-de-France?"):
        st.markdown("""
**RATP Display** refers to advertising displayed across the RATP transit network
(metro, bus, RER, tramway), which operates exclusively in **Île-de-France**.

In the model:
- RATP spend is set to **€0** for the other 12 regions
- Only the IDF region benefits from this channel's effect
- This is visible on the **Regional Analysis** page → heatmap and IDF chart
""")

    with st.expander("What is the impact of macroeconomic variables?"):
        st.markdown("""
The model includes three control variables:

- **20-year mortgage rate**: **negative** impact on leads. When rates rise
  (e.g. ~4% in 2023), buyers are discouraged → fewer leads.
  The 2022–2023 rate surge explains part of the observed drop in leads.

- **Pinel tax incentive**: **positive** impact. This French tax incentive for rental investment
  stimulated investor demand. Its gradual phase-out (2023–2024) and removal (2025)
  reduced investor lead volume.

- **COVID**: **negative** one-off impact. The first lockdown (March–May 2020) cut leads
  by roughly half. Subsequent lockdowns had a milder effect (~-15 to -25%).
""")

    with st.expander("How is the model calculated?"):
        st.markdown("""
The model follows this logic:

```
leads = baseline (transactions × conversion_rate)
      + Σ channel_coefficient × saturation(adstock(spend))
      + β_interest × interest_rate
      + β_pinel × pinel_coefficient
      + β_covid × covid_impact
      + noise
```

**Steps:**
1. **Adstock**: applies geometric carry-over to media spend
2. **Saturation**: applies a Hill function to capture diminishing returns
3. **Regression**: a Ridge model estimates the coefficient for each variable
4. **Decomposition**: channel contribution = coefficient × transformed feature

The model achieves an **R² ≈ 0.74** (Ridge) / **~0.83** (Bayesian), meaning it explains 74-83% of the variance in leads. On unseen data (December 2025), the average prediction error is only **3.5%**.
""")

    with st.expander("How accurate are the predictions? (Train/Test validation)"):
        st.markdown("""
We tested the model's predictive power by training it on data up to **November 2025** and predicting **December 2025** (5 weeks the model had never seen).

| Week | Actual Leads | Predicted | Error |
|------|-------------|-----------|-------|
| Dec 1 | 13,968 | 14,040 | 0.5% |
| Dec 8 | 13,265 | 13,872 | 4.6% |
| Dec 15 | 14,544 | 13,818 | 5.0% |
| Dec 22 | 12,955 | 13,631 | 5.2% |
| Dec 29 | 13,343 | 13,621 | 2.1% |

**Average error: 3.5%** — the model predicts weekly leads with less than 4% error on unseen data.

This is a strong result, meaning the relationships learned (media spend → leads) generalize well to new weeks.
""")

    with st.expander("Can this POC be used in production?"):
        st.markdown("""
This POC is designed as a **demonstration** to illustrate the MMM methodology.

**To move to production, you would need to:**
- Replace synthetic data with real data (CRM, GA4, media plans)
- Use **PyMC-Marketing** with proper Bayesian fitting (MCMC) instead of the Ridge fallback
- Add temporal cross-validation (time-series split)
- Integrate tighter business priors (expected ROAS ranges per channel)
- Add budget optimization under constraints (min/max per channel, fixed total budget)
- Set up an automated data refresh pipeline
""")

# ── Page: FAQ (FR) ─────────────────────────────────────────────────────────────

elif page == "FAQ (FR)":
    log_visit("FAQ (FR)")
    st.title("FAQ — Foire Aux Questions")

    st.markdown("---")

    with st.expander("Qu'est-ce qu'un MMM (Media Mix Model) ?", expanded=True):
        st.markdown("""
Un **Media Mix Model** est un modèle statistique qui mesure l'impact de chaque canal média
(TV, Radio, Digital, Affichage...) sur des KPIs business — ici, les **leads** (demandes de contact
qualifiées envoyées aux agences immobilières) et les **téléchargements de l'app** (installations de l'application mobile).

Il permet de :
- **Quantifier la contribution** de chaque canal aux deux KPIs (leads & téléchargements app)
- **Calculer le ROAS** (Return On Ad Spend) de chaque investissement
- **Simuler des scénarios** budgétaires pour optimiser l'allocation média
- **Isoler l'effet média** des facteurs externes (saisonnalité, taux d'intérêt, COVID...)

Contrairement à l'attribution digitale (last-click, multi-touch), le MMM fonctionne sur des
données agrégées et capte aussi les effets des médias offline (TV, Radio, Affichage).

**Les deux KPIs suivis dans ce dashboard :**
- **Leads** : nombre de demandes de contact qualifiées soumises aux agences immobilières partenaires chaque semaine. C'est la métrique de conversion principale.
- **Téléchargements App** : nombre d'installations hebdomadaires de l'application mobile. Les téléchargements sont corrélés aux leads mais aussi portés par les campagnes digitales (notamment Google Ads) et la notoriété de marque (TV).
""")

    with st.expander("Quelles données sont utilisées ?"):
        st.markdown("""
Ce POC utilise des **données synthétiques réalistes** générées à partir de :

- **Volumes de transactions immobilières** : basés sur les tendances DVF (Demandes de Valeurs Foncières)
  avec ~1.1M transactions/an en 2020, chute à ~870k en 2023, reprise en 2025
- **Dépenses médias** : budgets réalistes par canal (TV : 8-12M€/an, Google Ads : 5-8M€/an, etc.)
  avec des saisonnalités propres à chaque canal
- **Variables macro** : taux d'intérêt 20 ans (données réelles ~1.2% en 2020 → ~4% en 2023),
  dispositif Pinel (fin progressive), impact COVID
- **13 régions françaises** avec des pondérations réalistes (IDF ~25%, ARA ~12%, PACA ~10%...)

**Granularité** : données hebdomadaires de janvier 2020 à décembre 2025 (~313 semaines).
""")

    with st.expander("Comment fonctionne la saturation (Hill curve) ?"):
        st.markdown("""
La **saturation** modélise les rendements décroissants de la publicité :
les premiers euros investis ont un fort impact, puis l'effet diminue.

On utilise une **fonction de Hill** : `f(x) = x^α / (λ^α + x^α)`

- **α (alpha)** contrôle la forme de la courbe :
  - α faible (ex : Google Ads = 0.4) → la courbe monte vite puis sature lentement = **bon rendement sur une large plage**
  - α élevé (ex : TV = 0.85) → la courbe sature rapidement = **rendements décroissants plus marqués**

Concrètement : un euro de plus en Google Ads rapporte proportionnellement plus qu'un euro de plus en TV.
""")

    with st.expander("Qu'est-ce que l'adstock ?"):
        st.markdown("""
L'**adstock** (ou carry-over) modélise le fait qu'une publicité continue à produire des effets
**après sa diffusion**.

On utilise un modèle **géométrique** : `adstock(t) = spend(t) + decay × adstock(t-1)`

- **TV** (decay = 0.7) : effet long, ~3 semaines — une campagne TV continue d'impacter les leads
  bien après sa diffusion
- **Radio** (decay = 0.4) : effet court, ~1 semaine
- **RATP Display** (decay = 0.5) : effet moyen, ~2 semaines
- **Google Ads** (decay = 0.2) : effet quasi-immédiat — les clics se convertissent très vite
""")

    with st.expander("Comment lire le ROAS ?"):
        st.markdown("""
Le **ROAS (Return On Ad Spend)** mesure le nombre de leads générés par euro investi.

| Canal | ROAS typique | Interprétation |
|-------|-------------|----------------|
| Google Ads | ~0.011 | **Le plus efficient** — chaque euro génère le plus de leads |
| Radio | ~0.010 | Bon rapport coût-efficacité |
| RATP Display | ~0.008 | Correct, mais limité à l'Île-de-France |
| TV | ~0.004 | **Le moins efficient** — coût élevé par lead, mais forte portée branding |

**Attention** : un ROAS faible ne signifie pas que le canal est inutile. La TV apporte de la
notoriété et un effet de halo qui bénéficie aux autres canaux (non mesuré ici).
""")

    with st.expander("Pourquoi Google Ads est le canal le plus rentable ?"):
        st.markdown("""
Plusieurs facteurs expliquent la rentabilité supérieure de Google Ads :

1. **Intent-based** : les utilisateurs cherchent activement un bien immobilier → taux de conversion élevé
2. **Saturation lente** (α = 0.4) : le rendement reste bon même quand on augmente le budget
3. **Effet immédiat** (decay = 0.2) : pas de déperdition temporelle, le clic se convertit rapidement
4. **Ciblage précis** : possibilité de cibler par région, intention, mot-clé

À l'inverse, la **TV** a un coût d'entrée élevé, sature plus vite (α = 0.85), et touche
une audience large mais moins qualifiée.
""")

    with st.expander("Comment utiliser le Budget Simulator ?"):
        st.markdown("""
Le simulateur permet de tester des scénarios budgétaires **en temps réel** :

1. **Ajustez les sliders** pour chaque canal (budget hebdomadaire en €)
2. Observez l'impact sur les **leads prédits**, le **CPA blended** et le **CPA par canal**
3. Comparez les **pie charts** : allocation budget vs contribution leads

**Conseils d'utilisation :**
- Commencez par augmenter Google Ads → c'est le levier le plus efficient
- Réduisez la TV pour voir l'impact → la baisse de leads est modérée
- Le RATP Display n'a d'effet qu'en Île-de-France
- Le slider Google Ads est plus progressif (pas de 1 000€) pour un réglage fin
""")

    with st.expander("Pourquoi le RATP Display n'impacte que l'Île-de-France ?"):
        st.markdown("""
Le **RATP Display** correspond à l'affichage publicitaire dans le réseau RATP
(métro, bus, RER, tramway), qui est exclusivement situé en **Île-de-France**.

Dans le modèle :
- Les dépenses RATP sont mises à **0€** pour les 12 autres régions
- Seule la région IDF bénéficie de l'effet de ce canal
- C'est visible dans la page **Regional Analysis** → heatmap et graphique IDF
""")

    with st.expander("Quel est l'impact des variables macro-économiques ?"):
        st.markdown("""
Le modèle intègre trois variables de contrôle :

- **Taux d'intérêt 20 ans** : impact **négatif** sur les leads. Quand les taux montent
  (ex : ~4% en 2023), les acheteurs sont découragés → moins de leads.
  La hausse de 2022-2023 explique une partie de la chute de leads observée.

- **Dispositif Pinel** : impact **positif**. Ce dispositif de défiscalisation immobilière
  stimulait l'investissement locatif. Sa fin progressive (2023-2024) puis suppression (2025)
  réduit le volume de leads investisseurs.

- **COVID** : impact **négatif** ponctuel. Le premier confinement (mars-mai 2020) a divisé
  les leads par ~2. Les confinements suivants ont eu un impact plus modéré (~-15 à -25%).
""")

    with st.expander("Comment est calculé le modèle ?"):
        st.markdown("""
Le modèle suit cette logique :

```
leads = baseline (transactions × taux_de_conversion)
      + Σ coefficient_canal × saturation(adstock(dépense))
      + β_taux × taux_intérêt
      + β_pinel × coefficient_pinel
      + β_covid × impact_covid
      + bruit
```

**Étapes :**
1. **Adstock** : applique un carry-over géométrique sur les dépenses média
2. **Saturation** : applique une fonction de Hill pour capturer les rendements décroissants
3. **Régression** : un modèle Ridge estime les coefficients de chaque variable
4. **Décomposition** : on obtient la contribution de chaque canal en multipliant coefficient × feature transformée

Le modele atteint un **R² ≈ 0.74** (Ridge) / **~0.83** (Bayesien), ce qui signifie qu'il explique 74-83% de la variance des leads. Sur des donnees inconnues (decembre 2025), l'erreur moyenne de prediction n'est que de **3.5%**.
""")

    with st.expander("Quelle est la precision des predictions ? (Validation train/test)"):
        st.markdown("""
Nous avons teste la capacite predictive du modele en l'entrainant sur les donnees jusqu'a **novembre 2025**
et en predisant **decembre 2025** (5 semaines que le modele n'avait jamais vues).

| Semaine | Leads reels | Prediction | Erreur |
|---------|------------|------------|--------|
| 1er dec | 13 968 | 14 040 | 0.5% |
| 8 dec | 13 265 | 13 872 | 4.6% |
| 15 dec | 14 544 | 13 818 | 5.0% |
| 22 dec | 12 955 | 13 631 | 5.2% |
| 29 dec | 13 343 | 13 621 | 2.1% |

**Erreur moyenne : 3.5%** — le modele predit les leads hebdomadaires avec moins de 4% d'erreur sur des donnees inconnues.

C'est un resultat solide : les relations apprises (depenses media → leads) se generalisent bien a de nouvelles semaines.
""")

    with st.expander("Ce POC peut-il être utilisé en production ?"):
        st.markdown("""
Ce POC est conçu comme une **démonstration** pour illustrer la méthodologie MMM.

**Pour passer en production, il faudrait :**
- Remplacer les données synthétiques par des données réelles (CRM, GA4, plans média)
- Utiliser **PyMC-Marketing** avec un vrai fitting bayésien (MCMC) au lieu du fallback Ridge
- Ajouter la validation croisée temporelle (time-series split)
- Intégrer des priors business plus fins (fourchettes ROAS attendues par canal)
- Ajouter l'optimisation de budget sous contraintes (min/max par canal, budget total fixe)
- Mettre en place un pipeline de rafraîchissement automatique des données
""")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("[:house: HomeVision — Image Search Engine](http://34.22.238.118:8000)")
st.sidebar.markdown("Made by **Julien G.**")
