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

# Primary Inputs
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

# Heatmap Parameters (lower resolution for clarity)
st.sidebar.subheader("Heatmap Parameters")
S_min = st.sidebar.number_input("Min Spot Price", 0.5 * S, S, 0.8 * S)
S_max = st.sidebar.number_input("Max Spot Price", S, 1.5 * S, 1.2 * S)
V_min = st.sidebar.slider("Min Volatility for Heatmap", 0.01, 1.0, 0.1)
V_max = st.sidebar.slider("Max Volatility for Heatmap", 0.01, 1.0, 0.5)

# Create lower resolution grids
spot_points = 10  # try 10 points for clarity
vol_points = 10
spot_range = np.linspace(S_min, S_max, spot_points)
vol_range = np.linspace(V_min, V_max, vol_points)
call_matrix = np.zeros((vol_points, spot_points))
put_matrix = np.zeros((vol_points, spot_points))

for i, v in enumerate(vol_range):
    for j, s in enumerate(spot_range):
        call_matrix[i, j], put_matrix[i, j] = black_scholes(s, K, T, r, v)

# Use Streamlit columns for side-by-side plots
col1, col2 = st.columns(2)

# Plot Call Price Heatmap with Labels
with col1:
    st.markdown("#### Call Price Heatmap")
    fig1, ax1 = plt.subplots(figsize=(4.5, 3.5))
im1 = ax1.imshow(call_matrix, cmap="viridis")

# Add text annotations centered on each square
for i in range(call_matrix.shape[0]):
    for j in range(call_matrix.shape[1]):
        value = call_matrix[i, j]
        ax1.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)

# Formatting
ax1.set_xticks(np.arange(len(spot_range)))
ax1.set_yticks(np.arange(len(vol_range)))
ax1.set_xticklabels([f"{s:.0f}" for s in spot_range])
ax1.set_yticklabels([f"{v:.2f}" for v in vol_range])
ax1.set_xlabel("Spot Price")
ax1.set_ylabel("Volatility")
plt.colorbar(im1, ax=ax1)
st.pyplot(fig1)

# Plot Put Price Heatmap with Labels
with col2:
    st.markdown("#### Put Price Heatmap")
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
im2 = ax2.imshow(put_matrix, cmap="plasma")

for i in range(put_matrix.shape[0]):
    for j in range(put_matrix.shape[1]):
        value = put_matrix[i, j]
        ax2.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)

ax2.set_xticks(np.arange(len(spot_range)))
ax2.set_yticks(np.arange(len(vol_range)))
ax2.set_xticklabels([f"{s:.0f}" for s in spot_range])
ax2.set_yticklabels([f"{v:.2f}" for v in vol_range])
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Volatility")
plt.colorbar(im2, ax=ax2)
st.pyplot(fig2)
