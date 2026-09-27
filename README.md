# AI-Powered Counterparty Credit Risk & Stress Testing Engine

An end-to-end, code-first portfolio project covering counterparty credit risk
(CCR) exposure analytics, credit concentration analysis, macro/thematic stress
testing, a PD machine-learning model with validation, an AI-assisted risk
analyst, and an interactive Streamlit dashboard — all built on a large
**synthetic** counterparty portfolio.

> **This is an educational portfolio project, not a regulatory implementation.**
> It does **not** implement Basel SA-CCR, IMM, ICAAP, CCAR or EBA stress
> testing, and it does **not** use any real customer, counterparty or bank
> data. Every simplification is documented below and in the module docstrings.
> The goal is to demonstrate the *shape* of a bank's counterparty-risk
> analytics stack, not to reproduce a specific regulatory or proprietary
> methodology.

---

## 1. Project Overview

The project simulates a derivatives-counterparty book of **5,000 synthetic
counterparties** across 10 industries, 9 EMEA countries and 5 product types,
and builds a full analytics stack on top of it:

- A deterministic **risk engine** that computes Current Exposure, a simplified
  Potential Future Exposure (PFE), Exposure at Default (EAD) and Expected Loss
  (EL) per counterparty, and aggregates them by industry / country / rating /
  product.
- A **credit concentration & early-warning module** that flags name, sector
  and rating concentration and screens counterparties against illustrative
  early-warning triggers (GREEN / AMBER / RED).
- A **stress-testing engine** with four macro scenarios (Base, Recession,
  Severe Recession, Market Shock) plus six thematic shocks (energy, FX,
  rates, real estate, EMEA recession, financial-sector stress).
- A **PD machine-learning model** (Logistic Regression, Random Forest,
  LightGBM/Gradient Boosting) with proper train/test evaluation, calibration
  and SHAP-based explainability.
- An **AI Risk Analyst** that explains — but never calculates — the numbers
  produced by the modules above.
- A **5-page Streamlit dashboard** tying everything together, plus a SQLite
  database and hand-written SQL queries for the same analytics.

## 2. Business Problem

Banks that trade derivatives, repos and securities-financing transactions
with corporate and financial-institution counterparties carry **counterparty
credit risk (CCR)**: the risk that a counterparty defaults before a trade's
final settlement, leaving the bank with a replacement-cost loss. Managing
that risk requires:

- Knowing today's exposure to every counterparty (**current exposure**).
- Anticipating how large that exposure could become before maturity
  (**PFE**), because derivatives exposure moves with market rates and prices.
- Netting off collateral to get the loss-given-default exposure (**EAD**).
- Combining exposure with the counterparty's creditworthiness (**PD**, **LGD**)
  to get an **expected loss** figure that can be aggregated, limited and
  capitalised against.
- Understanding where risk is **concentrated** (single name, sector, country,
  rating band) rather than just how large it is in total.
- Testing how all of the above behaves in a downturn (**stress testing**),
  and catching counterparties that are **deteriorating** before they default
  (**early warning indicators**).

## 3. Why Counterparty Credit Risk Matters

Unlike a loan, a derivative's exposure is uncertain and moves with the
market — a bank can be owed money by a counterparty one day and owe that
counterparty money the next. CCR losses were a central feature of the
2008 financial crisis (monoline insurers, Lehman Brothers' derivatives book)
and remain a core regulatory capital charge (CVA, default risk capital) and
a core desk-level risk (limits, wrong-way risk, margining) at every large
derivatives dealer, which is exactly the kind of analysis this project
demonstrates end to end.

## 4. Architecture

```
AI-Counterparty-Risk-Engine/
├── data/
│   ├── raw/                     synthetic counterparty data (generated)
│   └── processed/                risk-engine output table, stress results, SQLite db
├── notebooks/                    01-06: generation, EDA, risk engine, stress, PD, validation
├── sql/
│   ├── schema.sql                 SQLite schema (5 tables + 1 view)
│   └── queries.sql                Named analytical queries (Part 13)
├── src/
│   ├── config.py                  Every illustrative parameter, documented
│   ├── data_generator.py          Synthetic portfolio generator
│   ├── preprocessing.py            Financial ratios + credit health score
│   ├── risk_engine.py              PFE / EAD / Expected Loss + aggregation
│   ├── portfolio_analytics.py      Concentration analytics + Early Warning System
│   ├── stress_testing.py           Scenario & thematic stress engine
│   ├── database.py                  SQLite build + query helpers
│   ├── pd_model.py                  PD ML model training/evaluation
│   ├── explainability.py            Feature importance, SHAP, PSI, validation notes
│   └── ai_risk_analyst.py           LLM explains engine outputs (never calculates them)
├── app.py                         Streamlit dashboard (5 pages)
├── run_pipeline.py                One command: generate data -> build every table
├── tests/                         pytest unit tests for every risk calculation
├── docs/
│   ├── interview_questions.md      20 project-specific Q&A
│   └── resume_bullets.md            3 ATS-friendly bullets
├── requirements.txt
├── .env.example
└── .gitignore
```

### Data flow

```
generate_counterparties()             [data_generator.py]
        |
build_feature_table()                 [preprocessing.py]  -> financial ratios, credit score
        |
run_risk_engine()                     [risk_engine.py]     -> PFE, EAD, Expected Loss
        |
        +--> flag_high_risk_counterparties()  \
        +--> apply_early_warning_system()       [portfolio_analytics.py]
        +--> concentration_summary()           /
        |
        +--> run_scenarios() / apply_scenario()      [stress_testing.py]
        |
        +--> build_database()                        [database.py] -> risk_engine.db
        |
        +--> train_all_models()                      [pd_model.py]
        |         +--> tree_feature_importance() / compute_shap_values()  [explainability.py]
        |
        +--> ai_risk_analyst.ask()                    -> natural-language explanations
        |
        +--> app.py (Streamlit) -- ties every module above into 5 dashboard pages
```

Run the whole pipeline with:

```bash
pip install -r requirements.txt
python run_pipeline.py           # generates data + builds risk_engine.db
streamlit run app.py             # launches the dashboard
pytest                           # runs the unit test suite
```

## 5. Dataset

`src/data_generator.py` builds **5,000 synthetic counterparties** (no real
customer or confidential banking data). Fields include: identity (id, name,
industry, country, region), financials (revenue, assets, debt, cash, EBITDA,
interest expense, operating cash flow — current year and prior year),
trade/exposure fields (notional, maturity, current exposure, FX/IR exposure
split, collateral and collateral type), and the risk anchors PD, LGD and
`default_flag`.

**Industries:** Financial Services, Energy, Technology, Manufacturing,
Healthcare, Telecommunications, Real Estate, Consumer Goods, Industrials,
Transportation.
**Countries:** UK, Germany, France, Netherlands, Spain, Italy, Ireland,
Switzerland, UAE.
**Ratings:** AAA, AA, A, BBB, BB, B, CCC (investment-grade-heavy mix, as in a
derivatives-counterparty book).

### Documented synthetic-data assumptions (`config.py`)

Every numeric anchor lives in `src/config.py` with an explicit comment that
it is **illustrative**, not calibrated to market or default data:

- Each rating has an illustrative 1-year PD anchor, LGD anchor, typical
  Debt/EBITDA, cost of debt, Debt/Assets, Cash/Debt, current ratio and CSA
  (collateral-agreement) probability. Weaker ratings get higher PD, higher
  LGD, higher leverage, thinner liquidity — by construction.
- Financial statement fields (revenue, debt, EBITDA, cash, etc.) are drawn
  from the rating's typical ratios plus lognormal noise, so a firm's rating
  and its financials are **internally consistent** (e.g. a firm the model
  correctly recognises as weak has weak fundamentals, not just a bad label).
- `default_flag` is a Bernoulli draw with probability equal to each
  counterparty's own `pd` (with a small idiosyncratic financial-distress
  adjustment) — this is what lets `default_flag` be used as a PD-model
  target, and why the model's discrimination should not be read as evidence
  of real-world predictive power (see Limitations).
- Product type drives typical current-exposure-to-notional ratio, maturity
  distribution and LGD shift (e.g. Securities Financing has short maturity
  and a negative LGD shift for over-collateralisation-style products).
- Longer maturities increase PFE via a square-root-of-time maturity factor;
  higher collateral (net of a type-dependent haircut) reduces EAD.
- `relationship_diagnostics()` in `data_generator.py` prints correlation
  checks (e.g. rating vs PD, Debt/EBITDA vs PD) so the "should generally
  correlate" requirements above are verified on the generated sample, not
  just asserted in code comments.

## 6. Methodology (summary)

| Stage | Module | What it computes |
|---|---|---|
| Financial ratios & credit score | `preprocessing.py` | Debt/EBITDA, Interest Coverage, Current Ratio, Debt/Assets, Cash/Debt, OCF/Debt, EBITDA Margin, and a simple weighted, breakpoint-interpolated **credit health score** (0-100) |
| Exposure engine | `risk_engine.py` | Current Exposure, PFE, EAD, Expected Loss (counterparty / industry / country / rating / product level) |
| Concentration | `portfolio_analytics.py` | Top-10 by EAD/EL, industry/country/rating shares, Herfindahl-Hirschman Index (HHI) |
| Early Warning System | `portfolio_analytics.py` | 8 documented triggers -> GREEN / AMBER / RED |
| Stress testing | `stress_testing.py` | 4 macro scenarios + 6 thematic shocks, EL decomposed into PD/LGD/EAD effects |
| PD model | `pd_model.py` | Logistic Regression vs Random Forest vs LightGBM, ROC-AUC/precision/recall/F1/calibration |
| Model validation | `explainability.py` | Feature importance, SHAP, Population Stability Index, documented limitations |
| AI Risk Analyst | `ai_risk_analyst.py` | Routes a question to a Python "tool", then asks an LLM to explain the *already-computed* numbers |

## 7. Exposure Methodology

**Current Exposure** is generated directly (as a product-dependent fraction
of notional) to represent today's mark-to-market replacement cost if the
counterparty defaulted right now.

## 8. PFE Methodology

```
PFE = Notional x Add-on Factor x Maturity Factor
Maturity Factor = sqrt(clip(maturity_years, 0.04, 25))
```

Add-on factors are **product-dependent** (Interest Rate 1.0%, FX 4.0%,
Equity 12.0%, Commodity 15.0%, Securities Financing 3.0%) and only loosely of
the same order of magnitude as regulatory supervisory factors.

> **This is explicitly NOT SA-CCR.** A real SA-CCR calculation nets trades
> within a legal netting set, applies supervisory delta/maturity/supervisory-
> factor adjustments per asset class, computes a multiplier that reflects
> excess collateral, and aggregates hedging sets before add-ons are summed —
> none of that netting-set-level machinery is implemented here. This project
> works at the single-trade level for transparency.

## 9. EAD Methodology

```
Eligible Collateral = Collateral x (1 - Haircut[collateral_type])
EAD = max(Current Exposure + PFE - Eligible Collateral, 0)
```

Collateral haircuts are type-dependent (Cash 0%, Government Bonds 4%,
Corporate Bonds 10%, Equities 20%) — a simplified, non-regulatory haircut
schedule (real haircuts under SA-CCR/IMM also depend on residual maturity
and currency mismatch, which this project ignores).

## 10. Expected Loss

```
Expected Loss = PD x LGD x EAD
```

computed at counterparty, industry, country, rating and product level via
`risk_engine.aggregate_risk()`, with EAD-weighted average PD/LGD and each
segment's share of total portfolio EAD/EL.

## 11. Stress Testing

Four macro scenarios (`src/stress_testing.py`), each a simple multiplicative
shock to PD, LGD and exposure drivers:

| Scenario | PD | LGD | Exposure | FX / IR exposure |
|---|---|---|---|---|
| Base | 1.0x | 1.0x | 1.0x | — |
| Recession | 1.5x | 1.1x | 1.10x | — |
| Severe Recession | 2.0x | 1.2x | 1.25x | — |
| Market Shock | 1.3x | 1.1x | — | FX +30%, IR +20% |

Plus six thematic shocks (Energy, FX, Interest Rate, Real Estate downturn,
EMEA recession, Financial-sector stress), each tilting the same mechanics
toward a specific industry/country/product. Every scenario reports **Base
EL, Stressed EL, Incremental EL and % change**, and the incremental EL is
decomposed sequentially into a PD effect, an LGD effect and an EAD effect.

> **Not CCAR / EBA / ICAAP.** These are simple, illustrative multiplicative
> shocks, not macro-economic-model-derived risk-factor paths, and are not
> calibrated to any historical stress episode or supervisory scenario.

## 12. PD Model

`src/pd_model.py` trains Logistic Regression, Random Forest and LightGBM
(falls back to Gradient Boosting if LightGBM isn't installed) on 8 standard
credit-analytics features plus rating and industry, targeting `default_flag`,
with `class_weight="balanced"` / `scale_pos_weight` to handle the realistic
~2-3% default rate. Evaluated with ROC-AUC, average precision, precision,
recall, F1, confusion matrix and a calibration (reliability) table.

Random Forest was the strongest model on the generated sample (ROC-AUC
≈0.85). **This is expected to look better than a real-world PD model** —
see Limitations.

## 13. Model Validation

`src/explainability.py` adds: global feature importance (tree
impurity-based, and SHAP mean |value| when `shap` is installed), local
per-counterparty SHAP explanations, and a Population Stability Index (PSI)
demonstration between two random halves of the test-set score distribution
(a stand-in for "development vs current portfolio" drift monitoring, since
this project has only one time period of synthetic data).
`explainability.validation_notes()` explicitly separates what is
**implemented here** from what a **production model-validation function**
would additionally require (multi-year out-of-time validation, independent
replication, governance sign-off, etc.).

## 14. AI Risk Analyst

**Design principle: the LLM never calculates a risk number.** Every figure
the user sees is produced by `risk_engine.py` / `stress_testing.py` /
`portfolio_analytics.py`. The flow is:

```
user question -> keyword-based intent router -> a Python "tool" function
computes/looks up the answer from already-computed tables -> those numbers
(not raw data dumps) go into an LLM prompt -> the LLM writes the explanation
```

If `ANTHROPIC_API_KEY` is not set, every question still works: the module
falls back to a plain, rule-based narrative built from the *same* computed
dictionary, so the dashboard's AI Risk Analyst page is fully usable without
any API key. The key is read from an environment variable via
`python-dotenv` (`.env`, see `.env.example`) — never hard-coded.

## 15. Dashboard

`app.py` (Streamlit, 5 pages): **Portfolio Overview** (headline KPIs and
exposure/EL charts), **Counterparty Risk** (search, full risk profile, risk
flags, stress loss across all 4 scenarios), **Stress Testing** (scenario
selector, EL decomposition, scenario comparison, industry impact),
**Concentration Risk** (top names, industry/country/rating breakdowns, HHI,
Early Warning System summary and trigger frequency), and **AI Risk
Analyst** (chat-style Q&A over the modules above).

## 16. SQL Analytics

`sql/schema.sql` defines 5 tables (`counterparties`, `exposures`,
`risk_metrics`, `scenario_definitions`, `stress_results`) plus a
`v_counterparty_risk` view; `sql/queries.sql` holds named queries for top
counterparties by EAD, top industries by exposure, expected loss by rating,
high-risk counterparties, exposure concentration and stress loss by
industry, all runnable via `src/database.py`'s `run_named_query()`.

## 17. Results

On the generated 5,000-counterparty portfolio: total EAD ≈ €40bn, total
Expected Loss ≈ €429mn (≈107 bps of EAD); the top-10 counterparties hold
≈8% of EAD but ≈21% of Expected Loss (risk is more concentrated than raw
exposure); under the Severe Recession scenario, Expected Loss increases by
roughly +230%; the PD model (Random Forest) achieves ROC-AUC ≈0.85 on a
held-out test set. (Exact figures vary with the random seed and will differ
slightly each time `run_pipeline.py` regenerates the data.)

## 18. Limitations

- **Synthetic data only.** No real counterparty, customer or market data was
  used anywhere in this project.
- **Single-trade PFE/EAD, not netting-set-level SA-CCR/IMM.** No legal
  netting, no supervisory multiplier, no hedging-set aggregation.
- **`default_flag` was generated from the same `pd` used as a feature route**
  (via rating and financial ratios), so the PD model's discrimination is
  expected to look better than a model trained on real, noisier default
  history — this is a demonstration of workflow, not of real predictive
  power.
- **Stress scenarios are simple multiplicative shocks**, not macro-model-
  derived risk-factor paths, and are not calibrated to any historical
  episode or supervisory scenario.
- **No true out-of-time validation**: only one time period of data exists,
  so PSI is demonstrated on two random splits rather than development vs.
  current populations.
- **Early-warning thresholds are illustrative rule-of-thumb values**, not
  back-tested or calibrated to this (or any) portfolio's actual defaults.
- **No wrong-way risk modelling**, no rating-migration model, no independent
  model validation function, and no governance/sign-off process — all of
  which a production implementation would require.

## 19. Future Improvements

- Netting-set-level exposure aggregation and a closer (still clearly-labelled
  educational) approximation of SA-CCR's supervisory factors and multiplier.
- A rating-migration (transition matrix) model feeding into stressed PD.
- Multi-period synthetic data to support genuine out-of-time PD validation
  and real PSI/CSI monitoring.
- Wrong-way risk flagging (e.g. FX-product exposure to a counterparty whose
  revenue is correlated with that FX rate).
- A proper backtesting harness comparing realised vs. predicted defaults
  across "years" of simulated data.

## 20. Technology Stack

Python 3.11+, pandas, NumPy, SciPy, scikit-learn, LightGBM, SHAP,
imbalanced-learn-aware class weighting, Plotly, Streamlit, SQLite +
SQLAlchemy, Faker, python-dotenv, Anthropic API, pytest, joblib.

---

## Relevance to Credit / CCR / CIB Risk

This project was built to demonstrate practical, hands-on familiarity with
the kind of analysis performed in **CCR EMEA Capital & Stress Testing**,
**Credit Risk (EMEA Corporates)**, **Structured Finance Risk** and **CIB
Risk** roles:

- **Counterparty exposure monitoring** — current exposure, PFE and EAD
  computed and aggregated at counterparty, industry, country, rating and
  product level.
- **Credit risk analysis** — financial-statement-derived ratios (Debt/EBITDA,
  interest coverage, leverage, liquidity) rolled into an interpretable
  credit health score.
- **Stress testing & scenario analysis** — four macro scenarios and six
  thematic shocks, with Expected Loss impact decomposed into PD/LGD/EAD
  drivers.
- **Portfolio monitoring & concentration risk** — top-name, industry,
  country and rating concentration with HHI, plus an Early Warning System
  with documented, auditable triggers.
- **Collateral analysis** — type-dependent haircuts, eligible collateral,
  collateral coverage and unsecured exposure.
- **Quantitative analysis & risk metrics** — PD/LGD/PFE/EAD/Expected Loss
  computed end to end, plus a validated machine-learning PD model.
- **Python automation & SQL analytics** — a fully modular, tested Python
  codebase, and a parallel SQLite/SQL implementation of the same analytics.
- **AI-assisted workflow & risk reporting** — an LLM layer that explains
  computed risk numbers for faster analyst review, with a hard architectural
  guarantee that the LLM never calculates the numbers itself.

This project does **not** claim JPMorgan work experience, access to
JPMorgan data, or JPMorgan-specific methodology — it is an independent,
self-directed portfolio project built to demonstrate these concepts using
public tools and entirely synthetic data.
