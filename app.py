"""Streamlit dashboard for the AI-Powered Counterparty Credit Risk & Stress
Testing Engine.

Educational portfolio project - synthetic data, simplified methodology.
See README.md for full documentation of every assumption.

Run with:  streamlit run app.py   (from the project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg
from src import ai_risk_analyst as ai
from src import portfolio_analytics as pa
from src import stress_testing as strt
from src.risk_engine import aggregate_risk, portfolio_summary

st.set_page_config(page_title="AI Counterparty Credit Risk Engine", layout="wide", page_icon="📊")

PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Prism


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Loading portfolio data...")
def load_data() -> pd.DataFrame:
    path = cfg.PROCESSED_DATA_PATH
    if not path.exists():
        st.error(
            f"No processed data found at `{path}`.\n\n"
            "Run `python run_pipeline.py` first to generate the synthetic "
            "portfolio and populate the risk engine outputs."
        )
        st.stop()
    df = pd.read_csv(path)
    df = pa.apply_early_warning_system(pa.flag_high_risk_counterparties(df))
    return df


def fmt_mn(x: float) -> str:
    return f"€{x:,.1f}mn"


def fmt_bn(x: float) -> str:
    return f"€{x / 1000:,.2f}bn" if abs(x) >= 1000 else fmt_mn(x)


df = load_data()

# --------------------------------------------------------------------------- #
# Sidebar navigation + global disclaimer
# --------------------------------------------------------------------------- #

st.sidebar.title("📊 CCR Risk Engine")
page = st.sidebar.radio(
    "Navigate",
    [
        "1 — Portfolio Overview",
        "2 — Counterparty Risk",
        "3 — Stress Testing",
        "4 — Concentration Risk",
        "5 — AI Risk Analyst",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Educational portfolio project.** All data is synthetic. "
    "Methodology is simplified and is **not** SA-CCR, IMM, ICAAP, CCAR or "
    "EBA-compliant. See README for full documentation."
)

# --------------------------------------------------------------------------- #
# PAGE 1 — Portfolio Overview
# --------------------------------------------------------------------------- #

if page.startswith("1"):
    st.title("Portfolio Overview")
    summary = portfolio_summary(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Counterparties", f"{summary['n_counterparties']:,}")
    c2.metric("Total EAD", fmt_bn(summary["total_ead"]))
    c3.metric("Total Expected Loss", fmt_mn(summary["total_expected_loss"]))
    c4.metric("Avg PD", f"{summary['average_pd']:.2%}")
    c5.metric("EAD-wtd PD", f"{summary['ead_weighted_pd']:.2%}")
    c6.metric("EL / EAD", f"{summary['el_to_ead_bps']:.0f} bps")

    ews_summary = pa.ews_summary(df)
    red_n = int(df.loc[df["ews_status"] == "RED"].shape[0])
    st.markdown(f"**High-risk (RED early-warning) counterparties:** {red_n:,} of {len(df):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        by_ind = aggregate_risk(df, "industry")
        fig = px.bar(
            by_ind, x="ead", y="industry", orientation="h", color="industry",
            color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
            title="Exposure (EAD) by Industry", labels={"ead": "EAD (€mn)", "industry": ""},
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        by_ctry = aggregate_risk(df, "country")
        fig = px.bar(
            by_ctry, x="ead", y="country", orientation="h", color="country",
            color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
            title="Exposure (EAD) by Country", labels={"ead": "EAD (€mn)", "country": ""},
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        by_rating = aggregate_risk(df, "rating")
        fig = px.bar(
            by_rating, x="rating", y="ead", color="rating",
            category_orders={"rating": cfg.RATINGS},
            color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
            title="Exposure (EAD) by Rating", labels={"ead": "EAD (€mn)", "rating": "Rating"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        el_ind = aggregate_risk(df, "industry").sort_values("expected_loss", ascending=False)
        fig = px.bar(
            el_ind, x="expected_loss", y="industry", orientation="h", color="industry",
            color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
            title="Expected Loss by Industry", labels={"expected_loss": "Expected Loss (€mn)", "industry": ""},
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top 20 Counterparties by EAD")
    top20 = pa.top_n_by(df, "ead", 20)
    st.dataframe(top20, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# PAGE 2 — Counterparty Risk
# --------------------------------------------------------------------------- #

elif page.startswith("2"):
    st.title("Counterparty Risk")

    search = st.text_input("Search by counterparty ID or name", "")
    if search:
        mask = (
            df["counterparty_id"].str.contains(search, case=False, na=False)
            | df["counterparty_name"].str.contains(search, case=False, na=False)
        )
        matches = df.loc[mask]
    else:
        matches = df

    if matches.empty:
        st.warning("No counterparty matches that search.")
    else:
        options = (matches["counterparty_id"] + " — " + matches["counterparty_name"]).tolist()
        choice = st.selectbox("Select counterparty", options)
        cp_id = choice.split(" — ")[0]
        row = df.loc[df["counterparty_id"] == cp_id].iloc[0]

        st.subheader(f"{row['counterparty_name']}  ({row['counterparty_id']})")
        badge_color = {"GREEN": "🟢", "AMBER": "🟠", "RED": "🔴"}.get(row["ews_status"], "")
        st.markdown(
            f"**Rating:** {row['rating']}  |  **Industry:** {row['industry']}  |  "
            f"**Country:** {row['country']}  |  **Product:** {row['product_type']}  |  "
            f"**Early-warning status:** {badge_color} {row['ews_status']}"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Exposure", fmt_mn(row["current_exposure"]))
        c1.metric("PFE", fmt_mn(row["pfe"]))
        c2.metric("Collateral", fmt_mn(row["collateral"]))
        c2.metric("Eligible Collateral", fmt_mn(row["eligible_collateral"]))
        c3.metric("EAD", fmt_mn(row["ead"]))
        c3.metric("Unsecured Exposure", fmt_mn(row["unsecured_exposure"]))
        c4.metric("PD", f"{row['pd']:.2%}")
        c4.metric("LGD", f"{row['lgd']:.0%}")

        st.metric("Expected Loss", fmt_mn(row["expected_loss"]))

        st.markdown("#### Risk Flags")
        flags = [
            ("High EAD + low rating", row.get("flag_high_ead_low_rating", False)),
            ("High EAD + high PD", row.get("flag_high_ead_high_pd", False)),
            ("High unsecured exposure", row.get("flag_high_unsecured", False)),
            ("High PFE", row.get("flag_high_pfe", False)),
            ("Rating deteriorating vs. last year", row.get("flag_deteriorating", False)),
        ]
        cols = st.columns(len(flags))
        for c, (label, hit) in zip(cols, flags):
            c.markdown(f"{'🚩' if hit else '✅'} {label}")

        st.markdown("#### Financial Ratios")
        ratio_cols = [
            "debt_to_ebitda", "interest_coverage", "current_ratio", "debt_to_assets",
            "cash_to_debt", "ocf_to_debt", "ebitda_margin", "credit_health_score", "credit_score_band",
        ]
        ratios = {c: row[c] for c in ratio_cols if c in row.index}
        st.table(pd.DataFrame([ratios]).T.rename(columns={0: "value"}))

        st.markdown("#### Stress Loss (all scenarios)")
        stress_rows = []
        for key, scenario in strt.DEFAULT_SCENARIOS.items():
            renamed = scenario if scenario.name == key else scenario.with_overrides(name=key)
            stressed = strt.apply_scenario(df, renamed)
            r = stressed.loc[stressed["counterparty_id"] == cp_id].iloc[0]
            stress_rows.append({
                "Scenario": strt.scenario_label(key),
                "Stressed EAD": r["stressed_ead"],
                "Stressed Expected Loss": r["stressed_expected_loss"],
                "Incremental EL": r["incremental_expected_loss"],
            })
        st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# PAGE 3 — Stress Testing
# --------------------------------------------------------------------------- #

elif page.startswith("3"):
    st.title("Stress Testing")
    st.caption(
        "Simple multiplicative scenarios applied to PD, LGD and exposure drivers. "
        "**Not** CCAR / EBA / ICAAP scenarios — see README for the exact mechanics."
    )

    scenario_key = st.selectbox(
        "Scenario", list(strt.DEFAULT_SCENARIOS.keys()),
        format_func=strt.scenario_label, index=2,
    )
    scenario = strt.DEFAULT_SCENARIOS[scenario_key]
    stressed = strt.apply_scenario(df, scenario)
    summ = strt.summarise_scenario(stressed)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base EAD", fmt_bn(summ["base_ead"]))
    c1.metric("Stressed EAD", fmt_bn(summ["stressed_ead"]), delta=f"{summ['ead_pct_change']:+.1%}")
    c2.metric("Base Expected Loss", fmt_mn(summ["base_expected_loss"]))
    c2.metric(
        "Stressed Expected Loss", fmt_mn(summ["stressed_expected_loss"]),
        delta=f"{summ['expected_loss_pct_change']:+.1%}",
    )
    c3.metric("Incremental EL", fmt_mn(summ["incremental_expected_loss"]))
    c4.metric("EAD-weighted stressed PD", f"{summ['ead_weighted_stressed_pd']:.2%}")

    st.markdown("### Expected Loss Decomposition (PD / LGD / EAD effects)")
    decomp = pd.DataFrame({
        "Effect": ["PD effect", "LGD effect", "EAD effect"],
        "Incremental EL (€mn)": [summ["el_effect_pd"], summ["el_effect_lgd"], summ["el_effect_ead"]],
    })
    fig = px.bar(
        decomp, x="Effect", y="Incremental EL (€mn)", color="Effect",
        color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
        title="Incremental Expected Loss Decomposition",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Compare All Scenarios")
    _, all_summary = strt.run_scenarios(df)
    all_summary_disp = all_summary.copy()
    all_summary_disp["Scenario"] = all_summary_disp["scenario"].map(strt.scenario_label)
    fig2 = px.bar(
        all_summary_disp, x="Scenario", y="stressed_expected_loss", color="Scenario",
        color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
        title="Stressed Expected Loss by Scenario", labels={"stressed_expected_loss": "Expected Loss (€mn)"},
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(
        all_summary_disp[[
            "Scenario", "base_expected_loss", "stressed_expected_loss",
            "incremental_expected_loss", "expected_loss_pct_change",
        ]],
        use_container_width=True, hide_index=True,
    )

    st.markdown("### Industry Impact — Selected Scenario")
    by_ind_stress = strt.aggregate_stress(stressed, "industry")
    fig3 = px.bar(
        by_ind_stress, x="incremental_expected_loss", y="industry", orientation="h",
        color="industry", color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
        title=f"Incremental Expected Loss by Industry — {strt.scenario_label(scenario_key)}",
        labels={"incremental_expected_loss": "Incremental EL (€mn)", "industry": ""},
    )
    fig3.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------------------- #
# PAGE 4 — Concentration Risk
# --------------------------------------------------------------------------- #

elif page.startswith("4"):
    st.title("Concentration Risk")

    conc = pa.concentration_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top-10 EAD share", f"{conc['top10_ead_share']:.1%}")
    c2.metric("Top-10 EL share", f"{conc['top10_el_share']:.1%}")
    c3.metric("Industry HHI (EAD)", f"{conc['industry_hhi_ead']:.3f}")
    c4.metric("Rating HHI (EAD)", f"{conc['rating_hhi_ead']:.3f}")
    st.caption(
        "HHI (Herfindahl-Hirschman Index) = sum of squared exposure shares. "
        "Closer to 1 means more concentrated; closer to 1/N means more diversified."
    )

    tabs = st.tabs(["Top Counterparties", "By Industry", "By Country", "By Rating", "Early Warning System"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        col1.markdown("**Top 10 by EAD**")
        col1.dataframe(pa.top_n_by(df, "ead", 10), use_container_width=True, hide_index=True)
        col2.markdown("**Top 10 by Expected Loss**")
        col2.dataframe(pa.top_n_by(df, "expected_loss", 10), use_container_width=True, hide_index=True)

    with tabs[1]:
        by_ind = aggregate_risk(df, "industry")
        fig = px.pie(
            by_ind, names="industry", values="ead", template=PLOTLY_TEMPLATE,
            color_discrete_sequence=COLOR_SEQ, title="EAD Share by Industry",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_ind, use_container_width=True, hide_index=True)

    with tabs[2]:
        by_ctry = aggregate_risk(df, "country")
        fig = px.pie(
            by_ctry, names="country", values="ead", template=PLOTLY_TEMPLATE,
            color_discrete_sequence=COLOR_SEQ, title="EAD Share by Country",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_ctry, use_container_width=True, hide_index=True)

    with tabs[3]:
        by_rating = aggregate_risk(df, "rating")
        fig = px.bar(
            by_rating, x="rating", y="ead_share", category_orders={"rating": cfg.RATINGS},
            color="rating", color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
            title="EAD Share by Rating", labels={"ead_share": "Share of Total EAD"},
        )
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_rating, use_container_width=True, hide_index=True)

    with tabs[4]:
        ews = pa.ews_summary(df)
        color_map = {"GREEN": "#2ca02c", "AMBER": "#ff9f1c", "RED": "#d62728"}
        fig = px.bar(
            ews, x="ews_status", y="ead", color="ews_status",
            color_discrete_map=color_map, template=PLOTLY_TEMPLATE,
            title="EAD by Early-Warning Status", labels={"ead": "EAD (€mn)", "ews_status": "Status"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ews, use_container_width=True, hide_index=True)

        st.markdown("**Trigger frequency across the portfolio**")
        st.dataframe(pa.trigger_frequency(df), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# PAGE 5 — AI Risk Analyst
# --------------------------------------------------------------------------- #

else:
    st.title("AI Risk Analyst")
    st.caption(
        "This assistant only **explains** numbers already computed by the deterministic "
        "risk engine above — it never calculates PD, LGD, EAD, PFE, Expected Loss or "
        "stress results itself. Set `ANTHROPIC_API_KEY` in your environment (see `.env.example`) "
        "for LLM-written explanations; without a key you still get a plain, rule-based summary "
        "of the same computed data."
    )

    example_qs = [
        "Give me a portfolio overview",
        "Which counterparties are most vulnerable under severe recession?",
        "Which industries have the highest concentration risk?",
        "Show counterparties with high EAD and low credit ratings.",
        "Which counterparties have the highest unsecured exposure?",
        "Which industries are most sensitive to an FX shock?",
    ]
    picked = st.selectbox("Try an example question (or type your own below)", [""] + example_qs)
    question = st.text_input("Ask the AI Risk Analyst", value=picked)

    if st.button("Ask") and question:
        with st.spinner("Computing risk metrics and generating explanation..."):
            response = ai.ask(df, question)
        st.markdown(f"**Intent detected:** `{response.intent}`  |  **LLM used:** {response.used_llm}")
        st.markdown("### Answer")
        st.write(response.narrative)
        with st.expander("Show underlying computed data (what the LLM was given)"):
            st.json(response.data, expanded=False)
