# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

warnings.filterwarnings("ignore")

# =====================================================
# OPENAI AVAILABILITY CHECK
# =====================================================
OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

llm = None
if OPENAI_AVAILABLE:
    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0.1,
        max_tokens=2000
    )

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config("AI Supply Chain Optimizer", layout="wide")
st.title("🤖 Multi-Agent Supply Chain Optimization System (A2A Enabled)")

if OPENAI_AVAILABLE:
    st.success("🔑 OpenAI Key detected — AI Strategy Agents ENABLED")
else:
    st.warning("⚠️ OpenAI Key not found — Running in Non-LLM Mode")

mode = st.radio(
    "Select Execution Mode",
    [
        "📈 Demand Forecasting Only",
        "📦 Inventory Optimization Only",
        "🚀 Full System (Demand → Inventory → Strategy → Risk)"
    ]
)

st.divider()

# =====================================================
# FILE UPLOADS
# =====================================================
forecast_file = None
inventory_file = None

if "Demand" in mode or "Inventory" in mode or "Full" in mode:
    forecast_file = st.file_uploader("📈 Upload Demand Data", type=["csv", "xlsx"])

if "Inventory" in mode or "Full" in mode:
    inventory_file = st.file_uploader("📦 Upload Inventory Master", type=["csv", "xlsx"])

# =====================================================
# AGENT 1 — DEMAND FORECASTING (NO LLM)
# =====================================================
def demand_forecasting_agent(df):
    df = df.rename(columns={
        "Planned Receipt Date": "ds",
        "Ordered Quantity": "y",
        "Item Desc": "Material"
    })
    df["ds"] = pd.to_datetime(df["ds"])

    results = []

    for mat in df["Material"].unique():
        item = df[df["Material"] == mat].sort_values("ds")
        train, test = item[:-5], item[-5:]
        actual = test["y"].values

        # Prophet
        prophet = Prophet()
        prophet.fit(train[["ds", "y"]])
        p_pred = prophet.predict(test[["ds"]])["yhat"].values
        rmse_p = sqrt(mean_squared_error(actual, p_pred))

        # ARIMA
        arima = ARIMA(train["y"], order=(1, 1, 1)).fit()
        a_pred = arima.forecast(steps=5)
        rmse_a = sqrt(mean_squared_error(actual, a_pred))

        # LSTM
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(train[["y"]])

        lookback = 3
        X, y_lstm = [], []

        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i, 0])
            y_lstm.append(scaled[i, 0])

        X = np.array(X).reshape(-1, lookback, 1)
        y_lstm = np.array(y_lstm)

        model = Sequential([
            LSTM(32, input_shape=(lookback, 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y_lstm, epochs=8, verbose=0)

        last_seq = scaled[-lookback:].reshape(1, lookback, 1)
        preds = []

        for _ in range(5):
            p = model.predict(last_seq, verbose=0)
            preds.append(p[0][0])
            last_seq = np.append(last_seq[:, 1:, :], [[[p[0][0]]]], axis=1)

        l_pred = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
        rmse_l = sqrt(mean_squared_error(actual, l_pred))

        rmses = {"Prophet": rmse_p, "ARIMA": rmse_a, "LSTM": rmse_l}
        best = min(rmses, key=rmses.get)

        results.append({
            "Material": mat,
            "BestModel": best,
            "ForecastedMonthlyDemand": round(np.mean(actual), 2),
            "RMSE": round(rmses[best], 2)
        })

    return results

# =====================================================
# AGENT 2 — INVENTORY OPTIMIZATION (NO LLM)
# =====================================================
def inventory_optimization_agent(inv_df, forecast):
    fmap = {f["Material"]: f for f in forecast}
    results = []

    for _, r in inv_df.iterrows():
        if r["Material"] not in fmap:
            continue

        monthly = fmap[r["Material"]]["ForecastedMonthlyDemand"]
        annual = monthly * 12
        daily = annual / 365

        EOQ = sqrt((2 * annual * r["OrderingCost"]) / r["HoldingCost"])
        ROP = daily * r["LeadTime"] + r.get("SafetyStock", 0)

        results.append({
            "Material": r["Material"],
            "EOQ": round(EOQ, 2),
            "ROP": round(ROP, 2)
        })

    return results

# =====================================================
# AGENT 3 — RISK SIMULATION (NO LLM)
# =====================================================
def risk_simulation_agent(merged):
    out = []

    for m in merged:
        monthly = m["Demand"]["ForecastedMonthlyDemand"]
        daily = monthly / 30
        rop = m["Inventory"]["ROP"]
        coverage = rop / daily if daily > 0 else 0

        out.append({
            "Material": m["Material"],
            "CoverageDays": round(coverage, 1),
            "DemandSurge(+20%)": round(monthly * 1.2, 2),
            "DemandDrop(-20%)": round(monthly * 0.8, 2),
            "StockoutRisk": "High" if coverage < 15 else "Medium" if coverage < 25 else "Low"
        })

    return out

# =====================================================
# LLM PROMPTS (USED ONLY IF KEY EXISTS)
# =====================================================
demand_strategy_agent_A2A = PromptTemplate(
    input_variables=["data"],
    template="""
You are a Demand Planning AI Agent.

Demand Forecast Output:
{data}

Explain demand stability, volatility, and forecast confidence per material.
Highlight implications for inventory decisions.
Bullet points only.
"""
)

inventory_strategy_agent_A2A = PromptTemplate(
    input_variables=["inventory", "demand_message"],
    template="""
You are an Inventory Strategy AI operating in an Agent-to-Agent (A2A) system.

Inventory Results:
{inventory}

Demand Strategy Message:
{demand_message}

For EACH material:
- Explain EOQ & ROP logic
- Discuss cost vs service trade-offs
- Link demand volatility to inventory risk
- Provide clear operational actions
Bullet points only.
"""
)

risk_mitigation_agent_A2A = PromptTemplate(
    input_variables=["risk", "inventory_message"],
    template="""
You are a Supply Chain Risk Mitigation AI.

Risk Simulation:
{risk}

Inventory Strategy Message:
{inventory_message}

Explain WHY risks exist and give mitigation actions per material.
Include buffers, contingency plans, and escalation triggers.
Bullet points only.
"""
)

# =====================================================
# RUN PIPELINE
# =====================================================
if st.button("▶ Run"):

    # ================= DEMAND ONLY =================
    if mode == "📈 Demand Forecasting Only" and forecast_file:
        df = pd.read_csv(forecast_file) if forecast_file.name.endswith(".csv") else pd.read_excel(forecast_file)
        demand = demand_forecasting_agent(df)

        st.subheader("📈 Demand Forecast")
        st.json(demand)

        if OPENAI_AVAILABLE:
            st.subheader("🤖 Demand Strategy Agent (A2A)")
            st.markdown((demand_strategy_agent_A2A | llm).invoke({"data": demand}).content)

    # ================= INVENTORY ONLY =================
    elif mode == "📦 Inventory Optimization Only" and forecast_file and inventory_file:
        df = pd.read_csv(forecast_file) if forecast_file.name.endswith(".csv") else pd.read_excel(forecast_file)
        inv = pd.read_csv(inventory_file) if inventory_file.name.endswith(".csv") else pd.read_excel(inventory_file)

        demand = demand_forecasting_agent(df)
        inventory = inventory_optimization_agent(inv, demand)

        st.subheader("📦 Inventory Optimization")
        st.json(inventory)

    # ================= FULL SYSTEM =================
    elif mode == "🚀 Full System (Demand → Inventory → Strategy → Risk)" and forecast_file and inventory_file:
        df = pd.read_csv(forecast_file) if forecast_file.name.endswith(".csv") else pd.read_excel(forecast_file)
        inv = pd.read_csv(inventory_file) if inventory_file.name.endswith(".csv") else pd.read_excel(inventory_file)

        demand = demand_forecasting_agent(df)
        inventory = inventory_optimization_agent(inv, demand)

        merged = [
            {"Material": d["Material"], "Demand": d, "Inventory": i}
            for d in demand for i in inventory if d["Material"] == i["Material"]
        ]

        risk = risk_simulation_agent(merged)

        st.subheader("📈 Demand Forecast")
        st.json(demand)

        st.subheader("📦 Inventory Optimization")
        st.json(inventory)

        st.subheader("⚠️ Risk Simulation")
        st.json(risk)

        if OPENAI_AVAILABLE:
            demand_msg = (demand_strategy_agent_A2A | llm).invoke({"data": demand}).content
            inventory_msg = (inventory_strategy_agent_A2A | llm).invoke({
                "inventory": inventory,
                "demand_message": demand_msg
            }).content
            risk_msg = (risk_mitigation_agent_A2A | llm).invoke({
                "risk": risk,
                "inventory_message": inventory_msg
            }).content

            st.subheader("🤖 Demand Strategy Agent (A2A)")
            st.markdown(demand_msg)

            st.subheader("🤖 Inventory Strategy Agent (A2A)")
            st.markdown(inventory_msg)

            st.subheader("🤖 Risk Mitigation Agent (A2A)")
            st.markdown(risk_msg)

    else:
        st.warning("Please upload the required files.")
