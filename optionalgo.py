# options.py
# Educational Streamlit app (safe-import version)
import os
import sys
import importlib
import subprocess

def safe_import(pkg_name, import_name=None):
    """
    Try to import pkg_name (or import_name if provided).
    If ImportError and not running on Streamlit Cloud, try to pip install it.
    If running on Streamlit Cloud, raise ImportError so deployment fails clearly.
    """
    module_name = import_name if import_name else pkg_name
    try:
        return importlib.import_module(module_name)
    except ImportError:
        # Detect Streamlit Cloud environment
        # Streamlit Cloud defines several env vars; STREAMLIT_RUNTIME is commonly present.
        if "STREAMLIT_RUNTIME" in os.environ or "STREAMLIT_DEPLOY" in os.environ:
            # On Streamlit Cloud: do not attempt runtime install (blocked).
            raise ImportError(f"Missing dependency '{pkg_name}'. Please add it to requirements.txt.")
        # Try to install locally
        print(f"Package '{pkg_name}' not found. Attempting to install via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        return importlib.import_module(module_name)

# Use safe_import for convenience locally
streamlit = safe_import("streamlit")
pd = safe_import("pandas")
plt = safe_import("matplotlib").pyplot
random = safe_import("random")  # random is stdlib; safe_import will import it quickly

# From this point, import names normally
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

# --- rest of your app logic (same as earlier) ---
st.set_page_config(page_title="Illiquid Option Simulation", layout="wide")
st.title("Illiquid Option & Orderbook Simulation — Educational")
st.markdown("Educational demo — not trading advice.")

with st.sidebar.form("params"):
    fair_price = st.number_input("Fair price", value=40.0)
    initial_bid = st.number_input("Initial bid", value=20.0)
    initial_ask = st.number_input("Initial ask", value=80.0)
    initial_size = st.number_input("Size at lone quotes", value=100)
    mm_spread = st.number_input("MM spread", value=5.0)
    mm_size = st.number_input("MM size", value=50)
    volatility = st.number_input("Volatility", value=2.5)
    steps = st.slider("Steps", 10, 200, 30)
    human_limit_price = st.number_input("Human limit buy price", value=21.0)
    human_size = st.number_input("Human buy size", value=10)
    run_btn = st.form_submit_button("Run")

if not run_btn:
    st.stop()

# Simple top-of-book sim...
book = {"bids": [(initial_bid, initial_size)], "asks": [(initial_ask, initial_size)]}
def mm_quotes(fair, spread, size): return (round(fair - spread, 2), size), (round(fair + spread, 2), size)
def get_mid(b): return None if not b["bids"] or not b["asks"] else (b["bids"][0][0] + b["asks"][0][0]) / 2

mm_bid, mm_ask = mm_quotes(fair_price, mm_spread, mm_size)
if mm_bid not in book["bids"]: book["bids"].append(mm_bid)
if mm_ask not in book["asks"]: book["asks"].append(mm_ask)
book["bids"].sort(key=lambda x: -x[0]); book["asks"].sort(key=lambda x: x[0])

history, trades = [], []
human_filled = 0; human_avg_price = None

for t in range(steps):
    mid = get_mid(book) or fair_price
    drift = (fair_price - mid) * 0.05
    shock = random.gauss(0, volatility)
    if random.random() < 0.25:
        if random.random() < 0.5 and book["bids"]:
            p, s = book["bids"][0]; take = min(s, max(1, int(s * 0.2)))
            trades.append({"time": t, "price": p, "size": take, "side": "sell_into_bid"})
            if s - take <= 0: book["bids"].pop(0)
            else: book["bids"][0] = (p, s - take)
        elif book["asks"]:
            p, s = book["asks"][0]; take = min(s, max(1, int(s * 0.2)))
            trades.append({"time": t, "price": p, "size": take, "side": "buy_from_ask"})
            if s - take <= 0: book["asks"].pop(0)
            else: book["asks"][0] = (p, s - take)
    mm_bid, mm_ask = mm_quotes(fair_price + random.gauss(0,1.0), mm_spread, mm_size)
    if mm_bid not in book["bids"]: book["bids"].append(mm_bid)
    if mm_ask not in book["asks"]: book["asks"].append(mm_ask)
    book["bids"].sort(key=lambda x: -x[0]); book["asks"].sort(key=lambda x: x[0])

    if human_filled < human_size and book["asks"] and book["asks"][0][0] <= human_limit_price:
        ask_price, ask_size = book["asks"][0]; fill_qty = min(human_size - human_filled, ask_size)
        trades.append({"time": t, "price": ask_price, "size": fill_qty, "side": "human_buy_fill"})
        if human_avg_price is None: human_avg_price = ask_price
        else: human_avg_price = (human_avg_price * human_filled + ask_price * fill_qty) / (human_filled + fill_qty)
        human_filled += fill_qty
        if ask_size - fill_qty <= 0: book["asks"].pop(0)
        else: book["asks"][0] = (ask_price, ask_size - fill_qty)

    history.append({"time": t, "bid": book["bids"][0][0] if book["bids"] else None, "ask": book["asks"][0][0] if book["asks"] else None, "mid": get_mid(book)})

import pandas as pd
df_hist = pd.DataFrame(history)
df_trades = pd.DataFrame(trades)

st.line_chart(df_hist.set_index("time")["mid"])
st.dataframe(df_trades)
if human_filled:
    pnl = (fair_price - human_avg_price) * human_filled
    st.success(f"Filled {human_filled} @ {human_avg_price:.2f} | P&L vs fair: {pnl:.2f}")
else:
    st.warning("Human order did not fill.")
