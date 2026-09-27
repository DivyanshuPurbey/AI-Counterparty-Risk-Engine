# Resume Bullets — AI-Powered Counterparty Credit Risk & Stress Testing Engine

Targeting: CCR EMEA Capital & Stress Testing · Credit Risk (EMEA Corporates)
· Risk Analyst – Structured Finance · CIB Risk Analyst

---

1. **Built an end-to-end counterparty credit risk engine in Python** on a
   synthetic 5,000-counterparty EMEA derivatives portfolio, computing
   Current Exposure, Potential Future Exposure, Exposure at Default and
   Expected Loss (PD × LGD × EAD) with automated aggregation across
   industry, country, rating and product dimensions.

2. **Designed and implemented a 4-scenario + 6-theme credit stress-testing
   framework** (Recession, Severe Recession, Market Shock, and thematic FX/
   rate/energy/real-estate/EMEA-recession shocks) with Expected Loss impact
   decomposed into PD, LGD and EAD effects, plus a concentration and
   early-warning module (HHI, top-10 exposure share, GREEN/AMBER/RED
   triggers) covering 100% of the simulated book.

3. **Trained and validated a Probability-of-Default machine-learning model**
   (Logistic Regression, Random Forest, LightGBM; ROC-AUC ≈0.85) with SHAP
   explainability and Population Stability Index monitoring, and layered an
   LLM-based "AI Risk Analyst" — architected so the LLM only explains,
   never calculates, risk metrics — on top of a 5-page interactive
   Streamlit dashboard and a parallel SQLite/SQL analytics layer.

*(Numbers reflect this project's synthetic dataset and are not claims of
production, regulatory or employer-specific results.)*
