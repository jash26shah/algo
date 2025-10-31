import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Investment Asset Classes", page_icon="📈", layout="wide")

# -----------------------------
# DATA
# -----------------------------
data = {
    "ASSET": [
        "Fixed Deposits",
        "Government Bonds",
        "T-Bills",
        "Debt Mutual Funds",
        "Senior Citizen Savings Scheme",
        "Post Office MIS",
        "AAA Corporate Bonds",
        "Blue-Chip Dividend Stocks",
    ],
    "RISK": [
        "Very Low", "Very Low", "Very Low", "Low",
        "Very Low", "Very Low", "Low", "Low-Moderate"
    ],
    "REWARD": [
        "5% - 7%", "6% - 8%", "5% - 6%", "5% - 8%",
        "7% - 8.5%", "6% - 7.5%", "7% - 9%", "8% - 12%"
    ],
    "DURATION": [
        "1-5 years", "3-7 years", "1 year", "1-5 years",
        "5 years", "5 years", "3-5 years", "3+ years"
    ],
    "% ALLOCATION (Suggested)": [
        "15% - 25%", "20% - 30%", "5% - 10%", "15% - 20%",
        "10% - 15%", "10% - 15%", "5% - 10%", "5% - 10%"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# UI Title
# -----------------------------
st.title("📊 Low-Risk Investment Asset Classes (India)")
st.write("Choose filters to analyze suitable assets for safe investing.")

# -----------------------------
# Filters
# -----------------------------
risk_filter = st.multiselect("Select Risk Level", df["RISK"].unique(), default=df["RISK"].unique())
duration_filter = st.multiselect("Select Duration", df["DURATION"].unique(), default=df["DURATION"].unique())

filtered_df = df[
    (df["RISK"].isin(risk_filter)) &
    (df["DURATION"].isin(duration_filter))
]

# -----------------------------
# Display Data
# -----------------------------
st.subheader("📁 Asset Classes Table")
st.dataframe(filtered_df)

# -----------------------------
# Charts
# -----------------------------
st.subheader("📌 Allocation % (Suggested)")
pie_data = filtered_df.copy()
pie_data["Allocation %"] = pie_data["% ALLOCATION (Suggested)"].apply(lambda x: int(x.split("%")[0]))

fig = px.pie(pie_data, names="ASSET", values="Allocation %", title="Suggested Allocation Distribution")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Reward Range Comparison")
bar_data = filtered_df.copy()
bar_data["Min Reward"] = bar_data["REWARD"].apply(lambda x: float(x.split("-")[0].replace("%","")))
bar_data["Max Reward"] = bar_data["REWARD"].apply(lambda x: float(x.split("-")[1].replace("%","")))

fig2 = px.bar(
    bar_data,
    x="ASSET",
    y=["Min Reward", "Max Reward"],
    barmode="group",
    title="Reward Range by Asset"
)
st.plotly_chart(fig2, use_container_width=True)

st.success("✅ App Loaded Successfully")
