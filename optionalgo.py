"""
Streamlit app: Simulate an illiquid option market with an algorithm that both
provides and takes liquidity and can "dump" into human buyers when price
reaches a threshold (educational demo).

Run locally:
    pip install streamlit pandas matplotlib
    streamlit run streamlit_algo_dump_simulation.py

Disclaimer: Educational only. Market manipulation is illegal.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# --- Streamlit setup ---
st.set_page_config(page_title="Algo Dump — Option Liquidity Simulator", layout="wide")

st.title("Algo Liquidity / Dump Simulator (educational)")
st.markdown(
    """
    This app simulates a simplified illiquid options market:
    - An algorithm quotes wide bid/ask prices.
    - A human buyer places increasingly aggressive buy orders.
    - The algorithm both provides and takes liquidity, pushing the price above fair value.
    - When price exceeds a threshold, the algorithm dumps inventory to humans.

    ⚠️ **Educational only — not for real trading.**
    """
)

# --- Sidebar parameters ---
with st.sidebar.form(key="params"):
    st.header("Simulation Parameters")
    fair_price = st.number_input("Fair price", value=40.0)
    threshold_pct = st.slider("Dump threshold (% above fair)", 5, 100, 20)
    initial_bid = st.number_input("Algo initial bid", value=20.0)
    initial_ask = st.number_input("Algo initial ask", value=80.0)
    algo_passive_ask_after = st.number_input("Algo passive ask after dump", value=100.0)

    time_steps = st.number_input("Total time steps", value=20, min_value=5, max_value=200)
    human_start = st.number_input("Human starts trading at step", value=2, min_value=0, max_value=100)
    human_end = st.number_input("Human stops trading at step", value=10, min_value=human_start, max_value=200)

    human_base_limit = st.number_input("Human starting limit price", value=21.0)
    human_limit_step = st.number_input("Human limit increment per step", value=2.0)
    human_buy_size = st.number_input("Human buy size", value=1.0)

    algo_push_steps = st.number_input("Algo push steps after trade", value=2)
    dump_size = st.number_input("Algo dump size", value=5.0)
    human_buy_at_dump = st.number_input("Human forced buy at dump", value=2.0)

    run_button = st.form_submit_button("Run Simulation")

if not run_button:
    st.info("👉 Adjust parameters in the sidebar and click 'Run Simulation'")
    st.stop()

# --- Simulation logic ---
threshold = fair_price * (1 + threshold_pct / 100)

display_bid = initial_bid
display_ask = initial_ask
last_price = (display_bid + display_ask) / 2

human_cash = 1_000_000.0
human_position = 0.0
human_avg_price = None

events = []


def record_event(t, actor, action, price, size, note=""):
    events.append(
        {
            "time": int(t),
            "actor": actor,
            "action": action,
            "price": float(price),
            "size": float(size),
            "note": note,
        }
    )


for t in range(int(time_steps)):
    if t < human_start:
        record_event(t, "algo", "idle", last_price, 0.0, "no trades yet")
    elif human_start <= t < human_end:
        human_limit = human_base_limit + (t - human_start) * human_limit_step
        if human_limit >= display_ask:
            trade_price = display_ask
            human_position += human_buy_size
            human_cash -= trade_price * human_buy_size
            human_avg_price = (
                (human_avg_price * (human_position - human_buy_size) + trade_price * human_buy_size)
                / human_position
                if human_avg_price
                else trade_price
            )
            record_event(t, "human", "buy", trade_price, human_buy_size, f"limit {human_limit} >= ask {display_ask}")
            last_price = trade_price
            for step in range(int(algo_push_steps)):
                algo_buy_price = last_price + 1 + step
                last_price = algo_buy_price
                record_event(t, "algo", "aggressive_buy", algo_buy_price, 0.5, "algo pushes price")
            display_ask = max(display_ask, last_price + 2)
            display_bid = min(display_bid, last_price - 5)
        else:
            record_event(t, "human", "limit_post", human_limit, human_buy_size, "resting order")
            if (t - human_start) % 2 == 0:
                algo_buy_price = min(human_limit + 1, display_ask - 0.5)
                last_price = algo_buy_price
                record_event(t, "algo", "aggressive_buy", algo_buy_price, 0.5, "algo reacts")
                display_ask = max(display_ask, last_price + 2)
                display_bid = min(display_bid, last_price - 5)
    else:
        record_event(t, "algo", "idle", last_price, 0.0, "no human activity")

    if last_price >= threshold:
        dump_price = threshold
        record_event(t, "algo", "large_sell", dump_price, dump_size, "algo dumps at threshold")
        human_position += human_buy_at_dump
        human_cash -= dump_price * human_buy_at_dump
        human_avg_price = (
            (human_avg_price * (human_position - human_buy_at_dump) + dump_price * human_buy_at_dump)
            / human_position
            if human_avg_price
            else dump_price
        )
        record_event(t, "human", "buy", dump_price, human_buy_at_dump, "human bought dump")
        last_price = dump_price
        display_bid = initial_bid
        display_ask = algo_passive_ask_after
        record_event(t, "algo", "reset_quotes", display_bid, 0.0, f"bid={display_bid}, ask={display_ask}")
        break

df = pd.DataFrame(events)

# --- Results ---
if human_avg_price:
    pnl_mark_last = (last_price - human_avg_price) * human_position
    pnl_mark_fair = (fair_price - human_avg_price) * human_position
else:
    pnl_mark_last = pnl_mark_fair = 0

# --- Visualization ---
st.subheader("Price Events Over Time")
if not df.empty:
    fig, ax = plt.subplots(figsize=(8, 3))
    trade_df = df[df["action"].isin(["buy", "aggressive_buy", "large_sell"])]
    ax.plot(trade_df["time"], trade_df["price"], marker="o")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.set_title("Price Movement Simulation")
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning("No price-moving events found.")

st.subheader("Event Log")
st.dataframe(df)

st.subheader("Summary")
st.write(f"**Final Last Price:** {last_price:.2f}")
st.write(f"**Human Position:** {human_position:.2f}")
if human_avg_price:
    st.write(f"**Human Avg Buy Price:** {human_avg_price:.2f}")
st.write(f"**P&L mark-to-last:** {pnl_mark_last:.2f}")
st.write(f"**P&L mark-to-fair:** {pnl_mark_fair:.2f}")

csv = df.to_csv(index=False).encode()
st.download_button("Download Event Log (CSV)", csv, "event_log.csv", "text/csv")

st.caption("Educational simulation — not financial advice.")
