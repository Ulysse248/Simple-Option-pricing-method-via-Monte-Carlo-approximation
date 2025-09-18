import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricing_methods import plot_grid, MonteCarloPricing, BlackScholesPricing

st.title("Monte Carlo European Option Pricing")


st.write("Under Black-Scholes with a risk-neutral measure, the asset price at T follows a probability distribution:")

st.latex(r"S_T = S_0 \, \exp\big((r - \frac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z\big), \quad Z \sim N(0,1)")

st.write("Then, the option price is the present value of the expected payoff.")
st.latex(r"V = e^{-rT}\mathbb{E}(\text{Payoff}(S_T))")
st.write("We use a Monte Carlo method to approximate the option price at T.")
st.latex(r"\tilde{V} = e^{-rT}\frac{1}{N}\Sigma_{i=1}^N \text{Payoff}(S_T^{(i)})")

st.latex(r"S_T^{(i)}, \mathrm{realisations \ of \ } S_T")


# User inputs
S0 = st.number_input("Initial asset price S0", value=100.0)
K = st.number_input("Strike price K", value=110.0)
T = st.number_input("Time to maturity T", value=2.0)
N = st.number_input("Number of Monte Carlo paths", value=1000)
r_min = st.slider("Min risk-free rate r", 0.0, 0.1, 0.0)
r_max = st.slider("Max risk-free rate r", 0.0, 0.1, 0.1)
sigma_min = st.slider("Min volatility σ", 0.0, 1.0, 0.1)
sigma_max = st.slider("Max volatility σ", 0.0, 1.0, 0.6)
option_type = st.selectbox(
    "Option type", 
    [("Call", 1), ("Put", -1)],
    format_func=lambda x: x[0]  # display only the first element
)
type_value = option_type[1]
grid_size=10

# Grid
r_values = np.linspace(r_min, r_max, grid_size)
sigma_values = np.linspace(sigma_min, sigma_max, grid_size)

mc_Pricing = MonteCarloPricing(N, 1, 1, K, T, type_value, S0)
plot_grid(r_values, sigma_values, mc_Pricing)

bs_Pricing = BlackScholesPricing(1, 1, K, T, type_value, S0)
plot_grid(r_values, sigma_values, bs_Pricing)
