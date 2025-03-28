import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Black-Scholes Model", layout="wide")

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    put = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put

st.sidebar.title("📊 Black-Scholes Model")

# Inputs
S = st.sidebar.number_input("Current Asset Price", 50.0, 200.0, 100.0)
K = st.sidebar.number_input("Strike Price", 50.0, 200.0, 100.0)
T = st.sidebar.number_input("Time to Maturity (Years)", 0.1, 5.0, 1.0)
sigma = st.sidebar.number_input("Volatility (σ)", 0.01, 1.0, 0.2)
r = st.sidebar.number_input("Risk-Free Interest Rate", 0.0, 0.2, 0.05)

call_price, put_price = black_scholes(S, K, T, r, sigma)

st.title("🧠 Black-Scholes Pricing Model")
st.markdown(f"**Call Value:** ${call_price:.2f}")
st.markdown(f"**Put Value:** ${put_price:.2f}")

st.markdown("---")
st.subheader("🎯 Options Price - Interactive Heatmap")

# Heatmap inputs
st.sidebar.subheader("Heatmap Parameters")
S_min = st.sidebar.number_input("Min Spot Price", 0.5*S, S, 0.8*S)
S_max = st.sidebar.number_input("Max Spot Price", S, 1.5*S, 1.2*S)
V_min = st.sidebar.slider("Min Volatility for Heatmap", 0.01, 1.0, 0.1)
V_max = st.sidebar.slider("Max Volatility for Heatmap", 0.01, 1.0, 0.5)

# Create grids
spot_range = np.linspace(S_min, S_max, 20)
vol_range = np.linspace(V_min, V_max, 20)
call_matrix = np.zeros((len(vol_range), len(spot_range)))
put_matrix = np.zeros((len(vol_range), len(spot_range)))

for i, v in enumerate(vol_range):
    for j, s in enumerate(spot_range):
        call_matrix[i, j], put_matrix[i, j] = black_scholes(s, K, T, r, v)

# Plotting
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Call Price Heatmap")
    fig1, ax1 = plt.subplots()
    c = ax1.imshow(call_matrix, cmap="viridis", aspect="auto",
                   extent=[spot_range[0], spot_range[-1], vol_range[0], vol_range[-1]],
                   origin="lower")
    plt.colorbar(c, ax=ax1)
    ax1.set_xlabel("Spot Price")
    ax1.set_ylabel("Volatility")
    st.pyplot(fig1)

with col2:
    st.markdown("#### Put Price Heatmap")
    fig2, ax2 = plt.subplots()
    c = ax2.imshow(put_matrix, cmap="plasma", aspect="auto",
                   extent=[spot_range[0], spot_range[-1], vol_range[0], vol_range[-1]],
                   origin="lower")
    plt.colorbar(c, ax=ax2)
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Volatility")
    st.pyplot(fig2)
