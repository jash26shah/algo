"""
Streamlit app: Simulate an illiquid option market with an algorithm that both
provides and takes liquidity and can "dump" into human buyers when price
reaches a threshold (educational demo).

Drop this file into a GitHub repo (e.g. app.py) and run locally with:
    pip install streamlit pandas matplotlib
    streamlit run streamlit_algo_dump_simulation.py

License / Disclaimer: Educational only. Market manipulation is illegal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Algo Dump — Option Liquidity Simulator", layout="wide")

st.title("Algo Liquidity / Dump Simulator (educational)")
st.markdown(
    """
    This is a simplified simulation that demonstrates a pattern:
    - Market starts illiquid with wide bid/ask from an 'algo' (market maker).
    - A human buyer places increasingly aggressive buys.
    - The algorithm both provides and takes liquidity, can push price up,
      and at a set threshold places a large sell (a 'dump') into buyers.

    *Educational only — do not use for trading or to manipulate markets.*
    """
)

# --- Controls ---
with st.sidebar.form(key='params'):
    st.header("Simulation parameters")
    fair_price = st.number_input("Fair price", value=40.0, step=1.0)
    threshold_pct = st.slider("Dump threshold (% above fair)", min_value=5, max_value=100, value=20)
    initial_bid = st.number_input("Algo initial bid (displayed)", value=20.0, step=1.0)
    initial_ask = st.number_input("Algo initial ask (displayed)", value=80.0, step=1.0)
    algo_passive_ask_after = st.number_input("Algo passive ask after dump", value=100.0, step=1.0)

    time_steps = st.number_input("Time steps (max events)", value=20, min_value=5, max_value=200)
    human_start = st.number_input("Human starts at time step", value=2, min_value=0, max_value=100)
    human_end = st.number_input("Human stops at time step", value=10, min_value=human_start, max_value=200)

    human_base_limit = st.number_input("Human starting limit price", value=21.0, step=1.0)
    human_limit_step = st.number_input("Human limit increment per step", value=2.0, step=0.5)
    human_buy_size = st.number_input("Human buy size per fill", value=1.0, step=0.1)

    algo_push_steps = st.number_input("Algo push steps after each human trade", value=2, min_value=0, max_value=10)
    dump_size = st.number_input("Algo dump size (contracts)", value=5.0, step=0.5)
    human_buy_at_dump = st.number_input("Human forced buy size at dump", value=2.0, step=0.1)

    run_button = st.form_submit_button("Run simulation")

if not run_button:
    st.info("Adjust parameters in the left panel and click 'Run simulation' to execute.")
    st.stop()

# --- Simulation logic ---
threshold = fair_price * (1.0 + threshold_pct / 100.0)

# State variables
display_bid = float(initial_bid)
display_ask = float(initial_ask)
last_price = (display_bid + display_ask) / 2.0

human_cash = 1_000_000.0
human_position = 0.0
human_avg_price = None

events = []

def record_event(t, actor, action, price, size, note=""):
    events.append({"time": int(t), "actor": actor, "action": action, "price": float(price), "size": float(size), "note": note})

for t in range(int(time_steps)):
    # Idle / passive times before human begins
    if t < human_start:
        if t == 0:
            record_event(t, "market", "open", last_price, 0.0, "initial mid")
        else:
            record_event(t, "algo", "idle", last_price, 0.0, "no trades")
    # Human active window
    elif human_start <= t < human_end:
        human_limit = human_base_limit + (t - human_start) * human_limit_step
        # Human posts a buy that will be executed if their limit >= displayed ask
        if human_limit >= display_ask:
            # Human buys at displayed ask
            trade_price = display_ask
            human_position += human_buy_size
            human_cash -= trade_price * human_buy_size
            human_avg_price = (human_avg_price * (human_position - human_buy_size) + trade_price * human_buy_size) / human_position if human_avg_price else trade_price
            record_event(t, "human", "buy", trade_price, human_buy_size, f"limit {human_limit} >= ask {display_ask}")
            last_price = trade_price
            # Algo pushes price upward by doing small aggressive buys
            for step in range(int(algo_push_steps)):
                algo_buy_price = last_price + 1.0 + step
                last_price = algo_buy_price
                record_event(t, "algo", "aggressive_buy", algo_buy_price, 0.5, "algo pushes price after human buy")
            # Update displayed quotes to reflect algorithm's new quote behaviour
            display_ask = max(display_ask, last_price + 2.0)
            display_bid = min(display_bid, last_price - 5.0)
        else:
            # Human posts a resting order — sometimes algo reacts by buying aggressively
            record_event(t, "human", "limit_post", human_limit, human_buy_size, "resting order")
            if (t - human_start) % 2 == 0:
                algo_buy_price = min(human_limit + 1.0, display_ask - 0.5)
                last_price = algo_buy_price
                record_event(t, "algo", "aggressive_buy", algo_buy_price, 0.5, "algo reacts to resting limit")
                display_ask = max(display_ask, last_price + 2.0)
                display_bid = min(display_bid, last_price - 5.0)
    else:
        # After human stops trading — idle / passive
        record_event(t, "algo", "idle", last_price, 0.0, "no human activity")

    # Check for dump condition
    if last_price >= threshold:
        # Algo places a large sell (dump) at threshold
        dump_price = threshold
        record_event(t, "algo", "large_sell", dump_price, dump_size, "algo dumps at threshold")
        # Human forced to buy some at dump
        human_position += human_buy_at_dump
        human_cash -= dump_price * human_buy_at_dump
        human_avg_price = (human_avg_price * (human_position - human_buy_at_dump) + dump_price * human_buy_at_dump) / human_position if human_avg_price else dump_price
        record_event(t, "human", "buy", dump_price, human_buy_at_dump, "human bought into dump")
        last_price = dump_price
        # Algo resets passive quotes
        display_bid = float(initial_bid)
        display_ask = float(algo_passive_ask_after)
        record_event(t, "algo", "reset_quotes", display_bid, 0.0, f"bid={display_bid}, ask={display_ask}")
        break

# Build DataFrame
df = pd.DataFrame(events)

# Compute human P&L (mark-to-last and mark-to-fair)
if human_avg_price is not None and human_position > 0:
    pnl_mark_last = (last_price - human_avg_price) * human_position
    pnl_mark_fair = (fair_price - human_avg_price) * human_position
else:
    pnl_mark_last = 0.0
    pnl_mark_fair = 0.0

# --- UI output ---
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Price events")
    # show a matplotlib line chart of price-changing events
    price_events = df[df['action'].isin(['buy', 'aggressive_buy', 'large_sell'])]
    if not price_events.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(price_events['time'], price_events['price'], marker='o')
        ax.set_xlabel('time step')
        ax.set_ylabel('price')
        ax.set_title('Price events over time')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write("No price-moving events recorded.")

    st.subheader("Event log")
    st.dataframe(df)

with col2:
    st.subheader("Summary stats")
    st.metric("Final last price", f"{last_price:.2f}")
    st.metric("Human position", f"{human_position:.2f} contracts")
    if human_avg_price:
        st.metric("Human average buy price", f"{human_avg_price:.2f}")
    st.write(f"P&L mark-to-last ({last_price:.2f}): {pnl_mark_last:.2f}")
    st.write(f"P&L mark-to-fair ({fair_price:.2f}): {pnl_mark_fair:.2f}")

# Provide CSV download
csv_buffer = StringIO()
if not df.empty:
    df.to_csv(csv_buffer, index=False)
    st.download_button(label="Download event log (CSV)", data=csv_buffer.getvalue(), file_name="algo_simulation_events.csv", mime="text/csv")

st.markdown("---")
st.markdown(
    "*Next steps / ideas:*\n"
    "- Add multiple human participants and random arrival processes.\n"
    "- Add explicit order book depth and resting orders tracking.\n"
    "- Add P&L for the Algo and risk limits.\n"
)

st.caption("This app is a teaching tool and does not represent any real market behavior precisely.")
