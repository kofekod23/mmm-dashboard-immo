# Media Mix Model — Real Estate (France)

> **[Version francaise ci-dessous](#version-francaise)**

## What is this?

This project is built for a **real estate listings platform** — a website that publishes property listings on behalf of real estate agencies who pay a subscription. The platform advertises across TV, radio, bus/metro posters, and Google Ads to generate **leads** (prospective buyers/renters who contact an agency) and **app downloads**.

The question is: **which advertising channel actually drives results, and where should the next euro go?**

That's exactly what a **Media Mix Model (MMM)** does. It's a statistical model that looks at past advertising spend and results (leads, app downloads), and figures out how much each channel contributed — even when everything runs at the same time.

### The key insight

Not all advertising works the same way:
- **Google Ads** is like a light switch — you pay, you get clicks *right now*. Turn it off, it stops.
- **TV and Radio** are more like planting seeds — people see/hear your ad, and weeks or months later, they remember the platform when they're looking for a property. This is called **adstock** (the memory effect of advertising).
- **All channels saturate** — the first 10,000 euros on Google Ads bring more leads than the next 10,000. There are diminishing returns.

### What makes this project special

This project uses **Bayesian inference** — instead of giving you one number ("TV brings 100 leads"), it gives you a range with uncertainty ("TV brings 80-120 leads, 94% confidence"). This is more honest and more useful for decision-making.

## Results at a glance

| Channel | Annual Budget | CPA (Cost Per Lead) | Adstock (memory) |
|---------|--------------|-------------------|-----------------|
| Google Ads | ~2.8M EUR | ~37 EUR | Near zero (instant) |
| RATP/Bus | ~560K EUR | ~47 EUR | Medium (weeks) |
| Radio | ~850K EUR | ~53 EUR | High (months) |
| TV (BFM etc.) | ~1.2M EUR | ~93 EUR | Very high (months) |

- **Model accuracy (R2):** Ridge = 0.74, Bayesian = ~0.83
- Google Ads is the cheapest per lead, TV is the most expensive but builds long-term brand awareness

## How it works — the simple version

1. **Data** — 5 years of weekly data (2020-2025): ad spend per channel, leads generated, plus external factors (interest rates, COVID lockdowns, Pinel tax incentive)
2. **Transformations** — Each channel's spend goes through two filters:
   - *Adstock*: spreads the effect over time (TV ad today still generates leads 3 months from now)
   - *Saturation*: models diminishing returns (the Hill curve)
3. **Model fitting** — A Bayesian model (PyMC-Marketing) learns the relationship between transformed spend and leads
4. **Output** — Contribution per channel, ROAS, response curves, and a budget simulator

## Live Dashboard

The Streamlit dashboard includes:
- **Overview** — KPIs, time series, regional breakdown
- **Channel Contributions** — Which channel drives what, waterfall chart, ROAS
- **Response Curves** — Saturation and marginal return per channel
- **Budget Simulator** — Adjust spend, see predicted leads with confidence intervals
- **Forecast** — 4-week ahead predictions with error bars
- **Goal Planner** — Set a lead target, get the optimal budget allocation
- **Regional Analysis** — Per-region performance and CPA
- **FAQ** — Full explanations in English and French

## Training the Bayesian model

The Bayesian model must be trained on a **GPU**. On a Google Colab A100, training takes approximately **30 minutes**. Without a powerful GPU, training could take several hours or more.

### Steps

1. Open `notebooks/mmm_exploration.ipynb` in Google Colab
2. Select a GPU runtime (A100 recommended, L4 or T4 will work but slower)
3. Run all cells — cell 13 fits the Bayesian model, cell 14 exports the results
4. Download `bayesian_posteriors.json` and place it in `data/model/`
5. The dashboard automatically detects the Bayesian model and enables confidence intervals

Without the Bayesian model, the dashboard falls back to a Ridge regression (faster, no GPU needed, but no uncertainty estimates).

## Tech Stack

- **PyMC-Marketing** + **NumPyro** (JAX backend) — Bayesian MMM with GPU acceleration
- **Streamlit** — Interactive dashboard
- **Plotly** — Charts
- **scikit-learn** — Ridge regression fallback
- **Google Cloud Run** — Deployment

---

<a id="version-francaise"></a>

# Media Mix Model — Immobilier (France)

## C'est quoi ?

Ce projet est concu pour un **site d'annonces immobilieres** — une plateforme qui publie les annonces des agences immobilieres qui paient un abonnement. La plateforme fait de la pub en TV, radio, affichage bus/metro, et Google Ads pour generer des **leads** (acheteurs/locataires potentiels qui contactent une agence) et des **telechargements d'app**.

La question : **quel canal publicitaire genere vraiment des resultats, et ou mettre le prochain euro ?**

C'est exactement ce que fait un **Media Mix Model (MMM)**. C'est un modele statistique qui regarde les depenses publicitaires passees et les resultats (leads, telechargements d'app), et determine combien chaque canal a contribue — meme quand tout tourne en meme temps.

### L'idee cle

La pub ne fonctionne pas pareil selon le canal :
- **Google Ads** c'est comme un interrupteur — vous payez, vous avez des clics *tout de suite*. Vous coupez, ca s'arrete.
- **TV et Radio** c'est comme planter des graines — les gens voient/entendent la pub, et des semaines ou des mois plus tard, ils se souviennent de la plateforme quand ils cherchent un bien. C'est l'**adstock** (l'effet memoire de la pub).
- **Tous les canaux saturent** — les 10 000 premiers euros sur Google Ads ramenent plus de leads que les 10 000 suivants. Les rendements sont decroissants.

### Ce qui rend ce projet special

Ce projet utilise l'**inference bayesienne** — au lieu de donner un seul chiffre ("la TV amene 100 leads"), il donne une fourchette avec l'incertitude ("la TV amene 80-120 leads, confiance a 94%"). C'est plus honnete et plus utile pour prendre des decisions.

## Resultats en un coup d'oeil

| Canal | Budget annuel | CPA (Cout Par Lead) | Adstock (memoire) |
|-------|--------------|-------------------|-----------------|
| Google Ads | ~2.8M EUR | ~37 EUR | Quasi nul (instantane) |
| RATP/Bus | ~560K EUR | ~47 EUR | Moyen (semaines) |
| Radio | ~850K EUR | ~53 EUR | Eleve (mois) |
| TV (BFM etc.) | ~1.2M EUR | ~93 EUR | Tres eleve (mois) |

- **Precision du modele (R2) :** Ridge = 0.74, Bayesien = ~0.83
- Google Ads est le moins cher par lead, la TV est la plus chere mais construit la notoriete long terme

## Comment ca marche — version simple

1. **Donnees** — 5 ans de donnees hebdomadaires (2020-2025) : depenses pub par canal, leads generes, plus des facteurs externes (taux d'interet, confinements COVID, dispositif Pinel)
2. **Transformations** — Les depenses de chaque canal passent par deux filtres :
   - *Adstock* : etale l'effet dans le temps (une pub TV aujourd'hui genere encore des leads 3 mois apres)
   - *Saturation* : modelise les rendements decroissants (courbe de Hill)
3. **Entrainement** — Un modele bayesien (PyMC-Marketing) apprend la relation entre les depenses transformees et les leads
4. **Resultats** — Contribution par canal, ROAS, courbes de reponse, et un simulateur de budget

## Dashboard interactif

Le dashboard Streamlit comprend :
- **Vue d'ensemble** — KPIs, series temporelles, repartition regionale
- **Contributions par canal** — Quel canal genere quoi, graphique en cascade, ROAS
- **Courbes de reponse** — Saturation et rendement marginal par canal
- **Simulateur de budget** — Ajustez les depenses, voyez les leads predits avec intervalles de confiance
- **Previsions** — Predictions a 4 semaines avec barres d'erreur
- **Goal Planner** — Fixez un objectif de leads, obtenez le budget optimal
- **Analyse regionale** — Performance et CPA par region
- **FAQ** — Explications completes en anglais et francais

## Entrainer le modele bayesien

Le modele bayesien doit etre entraine sur un **GPU**. Sur Google Colab avec un A100, l'entrainement prend environ **30 minutes**. Sans GPU puissant, l'entrainement pourrait durer plusieurs heures voire plus.

### Etapes

1. Ouvrir `notebooks/mmm_exploration.ipynb` dans Google Colab
2. Selectionner un runtime GPU (A100 recommande, L4 ou T4 fonctionnent mais plus lents)
3. Executer toutes les cellules — la cellule 13 entraine le modele, la cellule 14 exporte les resultats
4. Telecharger `bayesian_posteriors.json` et le placer dans `data/model/`
5. Le dashboard detecte automatiquement le modele bayesien et active les intervalles de confiance

Sans le modele bayesien, le dashboard utilise une regression Ridge (plus rapide, pas besoin de GPU, mais pas d'estimation d'incertitude).

## Stack technique

- **PyMC-Marketing** + **NumPyro** (backend JAX) — MMM bayesien avec acceleration GPU
- **Streamlit** — Dashboard interactif
- **Plotly** — Graphiques
- **scikit-learn** — Regression Ridge (fallback)
- **Google Cloud Run** — Deploiement
