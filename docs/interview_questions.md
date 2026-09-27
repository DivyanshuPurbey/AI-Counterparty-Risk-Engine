# Interview Preparation — 20 Q&A on the AI-Powered CCR Engine

Each answer marks whether the concept is **[Implemented in my project]** or
a **[Concept I understand]** that this educational project does not fully
build out (e.g. real SA-CCR, IMM, ICAAP). Being explicit about that boundary
is itself a signal of understanding, not a weakness — say so if asked.

---

**1. What is Counterparty Credit Risk (CCR)?**
CCR is the risk that the counterparty to a bilateral contract (typically a
derivative, repo, or securities-financing transaction) defaults before the
final settlement of the cash flows, causing the bank to lose the positive
replacement value of that contract. *[Concept I understand — this framing
underpins the whole project's design.]*

**2. What's the difference between Credit Risk and CCR?**
Traditional (lending) credit risk has a known, mostly fixed exposure (the
loan balance). CCR's exposure is uncertain and **bilateral** — it moves with
market prices, can flip from an asset to a liability, and needs
forward-looking exposure measures (PFE) rather than just an outstanding
balance. *[Concept I understand.]*

**3. What is EAD?**
Exposure at Default: the exposure amount used to calculate expected loss if
default happens today, net of eligible collateral. In this project:
`EAD = max(Current Exposure + PFE − Eligible Collateral, 0)`.
*[Implemented in my project — `risk_engine.calculate_ead()`.]*

**4. What is PFE?**
Potential Future Exposure: an estimate of how large a derivative's exposure
could become before maturity, because market moves can increase what the
counterparty owes the bank. *[Implemented in my project, in a simplified
single-trade form — see Q5.]*

**5. How did you calculate PFE, and what would a production PFE look like?**
My project uses `PFE = Notional × product-dependent Add-on Factor ×
sqrt(clipped maturity)` — a transparent, single-trade proxy. A production
SA-CCR/IMM calculation instead nets trades within a legal netting set,
applies supervisory delta/maturity/factor adjustments per asset class,
computes a multiplier reflecting excess collateral, and aggregates hedging
sets before summing add-ons — none of which I implement, and I say so
explicitly in the README. *[Implemented in my project (simplified) /
Concept I understand (real SA-CCR).]*

**6. Why does maturity affect PFE?**
Longer-dated trades have more time for the underlying market factor to
diffuse away from today's level, so potential future exposure grows with
time — I model that with a square-root-of-time maturity factor
(`sqrt(maturity)`), consistent with a diffusive risk-factor assumption.
*[Implemented in my project.]*

**7. What is PD?**
Probability of Default: the estimated likelihood a counterparty defaults
within a given horizon (1 year, in this project). *[Implemented — rating-
anchored in the synthetic data, and re-estimated by my PD ML model.]*

**8. What is LGD?**
Loss Given Default: the fraction of exposure the bank expects to lose if
the counterparty defaults, after any recoveries — the complement of the
recovery rate. *[Implemented — rating-anchored per counterparty, used
directly in Expected Loss.]*

**9. Why is Expected Loss = PD × LGD × EAD?**
Expected Loss is the probability-weighted average loss: PD is the chance a
loss event happens at all, EAD is the size of exposure at that point, and
LGD is the fraction of that exposure actually lost after recoveries. It's
the standard decomposition used across credit risk (also seen in RWA/IRB
formulas, though I don't reproduce the IRB capital formula itself).
*[Implemented in my project.]*

**10. How does collateral affect exposure?**
Posted collateral, net of a type-dependent haircut (cash 0%, government
bonds 4%, corporate bonds 10%, equities 20% in my simplified schedule),
becomes "eligible collateral" and is subtracted from Current Exposure + PFE
to get EAD, floored at zero. Higher, higher-quality collateral shrinks EAD
and Expected Loss. *[Implemented in my project (simplified haircuts, no
maturity/currency-mismatch add-ons).]*

**11. What happens during stress testing in your project?**
Each scenario multiplies PD, LGD and exposure drivers (and, for market
shocks, FX/IR exposure specifically), recomputes EAD and Expected Loss under
the shocked assumptions, and reports Base EL, Stressed EL, Incremental EL
and % change, with the incremental EL split into PD/LGD/EAD effects.
*[Implemented in my project — `stress_testing.py`.]*

**12. How did you design your stress scenarios?**
I used four macro scenarios of increasing severity (Base, Recession, Severe
Recession, Market Shock) with hand-set, documented multipliers, plus six
thematic shocks tilting the same mechanics toward a specific
industry/country/product (e.g. an FX shock raises FX exposure and
industries with high FX product usage more than others). They're
illustrative multiplicative shocks, not derived from a macro-economic model.
*[Implemented in my project (simplified) / Concept I understand (real
macro-scenario design with GDP/rates/FX paths feeding a re-pricing engine).]*

**13. What is concentration risk?**
The risk that losses cluster in a small number of names, sectors, countries
or rating bands rather than being diversified — a portfolio can have
moderate total exposure but still be dangerously concentrated. I measure it
with top-10 exposure/EL share and a Herfindahl-Hirschman Index (HHI) by
industry, country and rating. *[Implemented in my project.]*

**14. What is wrong-way risk?**
Wrong-way risk is when a counterparty's probability of default is positively
correlated with the bank's exposure to that counterparty — e.g. an FX
derivative with a counterparty whose creditworthiness depends on that same
currency. My project does not model this (exposure and PD move independently
in my engine). *[Concept I understand — flagged in "Future Improvements" as
something I would add next.]*

**15. What is SA-CCR?**
The Standardised Approach for Counterparty Credit Risk: the current Basel
III regulatory method for computing EAD for derivatives, using supervisory
factors per asset class, netting sets, hedging sets and a collateral-aware
multiplier. My PFE/EAD calculation is loosely inspired by its shape
(notional × factor × maturity term) but is explicitly **not** SA-CCR — no
netting sets, no supervisory multiplier, no hedging-set logic.
*[Concept I understand, not implemented.]*

**16. What is IMM?**
The Internal Model Method: an advanced, regulator-approved approach where a
bank models the full distribution of future exposure via Monte Carlo
simulation of risk factors, rather than using SA-CCR's standardised
formula. My project's PFE is a closed-form proxy, not a simulated exposure
distribution. *[Concept I understand, not implemented.]*

**17. What is ICAAP?**
The Internal Capital Adequacy Assessment Process: a bank's own, holistic
assessment of the capital it needs to hold against all its material risks
(credit, market, operational, concentration, etc.), going beyond the Pillar
1 regulatory minimum. My project computes portfolio-level Expected Loss and
stress impacts but does not build a capital-adequacy or economic-capital
framework. *[Concept I understand, not implemented.]*

**18. How would this project differ from a production bank model?**
Production models would use real, multi-year default and exposure history;
netting-set-level SA-CCR/IMM exposure; regulator-reviewed and independently
validated PD/LGD models with formal outcomes analysis; governed,
back-tested stress scenarios (often supervisory, e.g. EBA/CCAR); wrong-way
risk and rating-migration modelling; and a full model-risk-management
process (challenger models, sign-off, ongoing monitoring). My project
demonstrates the *shape and logic* of each of these using synthetic data and
transparent, simplified formulas — a portfolio/learning exercise, not a
production system.

**19. Why use a machine-learning model for PD here, alongside the
rating-based PD anchors?**
The rating-anchored PDs in the synthetic data represent a ratings-based (or
"expert") approach; the ML model demonstrates a statistical, data-driven
alternative and lets me show a real train/test evaluation workflow —
discrimination (ROC-AUC), threshold metrics, calibration and explainability
— on top of the same feature set a real quantitative-analyst PD build would
start from (leverage, coverage, liquidity ratios, rating, industry).
*[Implemented in my project — `pd_model.py`.]*

**20. Why should the LLM not calculate risk metrics itself?**
LLMs are not deterministic or auditable calculators — they can be internally
inconsistent, hallucinate plausible-looking numbers, and can't be unit
tested the way `EAD = max(CE + PFE − Collateral, 0)` can. In my architecture
every PD/LGD/PFE/EAD/Expected Loss/stress/concentration number is produced
by plain, unit-tested pandas/numpy code; the LLM only receives those
already-computed numbers and turns them into a written explanation. This
keeps every figure traceable back to a specific, testable calculation — a
non-negotiable property for anything resembling risk reporting.
*[Implemented in my project — `ai_risk_analyst.py`'s explicit
question → Python tool → numbers → LLM architecture.]*
