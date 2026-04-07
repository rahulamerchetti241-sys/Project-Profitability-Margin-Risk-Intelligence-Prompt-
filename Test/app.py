import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Project Profitability & Margin Risk Intelligence",
    page_icon="📊",
    layout="wide",
)

MODEL_PATH = "profit_margin_model.joblib"
META_PATH = "model_meta.joblib"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta["features"]


def compute_risk_score(budget, labor_cost, actual_cost, delay_pct, resource_utilization_pct):
    score = 0
    reasons = []

    if actual_cost > budget:
        score += 30
        reasons.append("Actual cost exceeds budget (+30)")

    if delay_pct > 20:
        score += 25
        reasons.append("Delay is above 20% (+25)")

    if resource_utilization_pct < 70:
        score += 20
        reasons.append("Resource utilization is below 70% (+20)")

    if labor_cost > 0.5 * budget:
        score += 25
        reasons.append("Labor cost is unusually high (>50% of budget) (+25)")

    score = int(min(score, 100))

    if score < 35:
        level = "Low"
    elif score < 70:
        level = "Medium"
    else:
        level = "High"

    return score, level, reasons


def project_status(pred_margin):
    if pred_margin >= 20:
        return "High Profit ✅"
    if pred_margin >= 5:
        return "Low Margin ⚠️"
    return "Loss Risk 🔴"


def money(v):
    return f"${v:,.0f}"


def main():
    st.title("📊 Project Profitability & Margin Risk Intelligence")
    st.caption("Fast hackathon MVP: ML + rule-based margin risk insights")

    try:
        model, feature_cols = load_model()
    except Exception:
        st.error(
            "Model not found. Please run:\n"
            "1) python generate_data.py\n"
            "2) python train_model.py"
        )
        st.stop()

    st.subheader("1) Project Input")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        budget = st.number_input("Budget", min_value=10000.0, value=400000.0, step=10000.0)
        duration_months = st.number_input("Duration (months)", min_value=1, value=12, step=1)

    with col2:
        labor_cost = st.number_input("Labor Cost", min_value=0.0, value=140000.0, step=5000.0)
        material_cost = st.number_input("Material Cost", min_value=0.0, value=100000.0, step=5000.0)

    with col3:
        overhead = st.number_input("Overhead", min_value=0.0, value=50000.0, step=2000.0)
        actual_cost = st.number_input("Actual Cost", min_value=0.0, value=320000.0, step=5000.0)

    with col4:
        delay_pct = st.slider("Delay %", min_value=0, max_value=100, value=15)
        resource_utilization_pct = st.slider("Resource Utilization %", min_value=0, max_value=100, value=75)

    if st.button("🚀 Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame(
            [
                {
                    "budget": budget,
                    "duration_months": duration_months,
                    "labor_cost": labor_cost,
                    "material_cost": material_cost,
                    "overhead": overhead,
                    "actual_cost": actual_cost,
                    "delay_pct": delay_pct,
                    "resource_utilization_pct": resource_utilization_pct,
                }
            ]
        )[feature_cols]

        pred_margin = float(model.predict(input_df)[0])
        score, level, reasons = compute_risk_score(
            budget, labor_cost, actual_cost, delay_pct, resource_utilization_pct
        )
        status = project_status(pred_margin)

        st.subheader("2) Prediction Output")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted Profit Margin", f"{pred_margin:.2f}%")
        k2.metric("Project Status", status)
        k3.metric("Risk Score", f"{score}/100")
        k4.metric("Risk Level", level)

        st.progress(score / 100, text=f"Risk Gauge: {score}/100 ({level})")

        st.subheader("3) Risk Reasons")
        if reasons:
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.success("No major risk triggers detected.")

        st.subheader("4) Visual Intelligence")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            cost_df = pd.DataFrame(
                {
                    "Component": ["Labor", "Material", "Overhead", "Actual Cost", "Budget"],
                    "Amount": [labor_cost, material_cost, overhead, actual_cost, budget],
                }
            )
            fig_bar = px.bar(
                cost_df,
                x="Component",
                y="Amount",
                color="Component",
                title="Cost Breakdown",
                text_auto=".2s",
            )
            fig_bar.update_layout(showlegend=False, yaxis_title="Amount ($)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": "Risk Score Gauge"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ef4444" if score >= 70 else "#f59e0b" if score >= 35 else "#22c55e"},
                        "steps": [
                            {"range": [0, 35], "color": "#dcfce7"},
                            {"range": [35, 70], "color": "#fef3c7"},
                            {"range": [70, 100], "color": "#fee2e2"},
                        ],
                    },
                )
            )
            st.plotly_chart(gauge, use_container_width=True)

        # Optional visual: Profit vs Cost
        st.subheader("5) Profit vs Cost Snapshot")
        profit_value = budget * (pred_margin / 100)
        compare_df = pd.DataFrame(
            {
                "Metric": ["Budget", "Actual Cost", "Predicted Profit Value"],
                "Amount": [budget, actual_cost, profit_value],
            }
        )
        fig_compare = px.bar(compare_df, x="Metric", y="Amount", color="Metric", text_auto=".2s")
        fig_compare.update_layout(showlegend=False, yaxis_title="Amount ($)")
        st.plotly_chart(fig_compare, use_container_width=True)

        # Optional feature importance
        if hasattr(model, "feature_importances_"):
            st.subheader("6) Feature Importance (Optional)")
            fi_df = pd.DataFrame(
                {"Feature": feature_cols, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=False)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Model Feature Importance")
            st.plotly_chart(fig_fi, use_container_width=True)

        st.info(
            f"Estimated profit value: {money(profit_value)} | "
            f"Estimated total margin impact from budget {money(budget)}"
        )

    st.divider()
    st.caption("Tip: For demo flow, use high delay + low utilization + actual cost above budget to show High Risk.")


if __name__ == "__main__":
    main()