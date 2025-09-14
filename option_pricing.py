import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
r_val_plot = np.linspace(0.0, 0.1, grid_size+1)       # 0% to 10%
sigma_val_plot = np.linspace(0.1, 0.6, grid_size+1)

# Payoff function
def payoff(s, K, type): # 1 is call, -1 is put
    temp = type*(s-K)
    temp[temp < 0]=0
    return temp

# Option pricing function
def option_pricing(N,r,sigma,K,T, type_value=1, S0=100):
    temp_ST_factor = S0 * np.exp((r-0.5*sigma**2)*T) 
    vect_ST = []
    for _ in range(N):
        Z = np.random.randn()
        STi =  temp_ST_factor * np.exp(sigma*np.sqrt(T)*Z)
        vect_ST.append(STi)

    vect_ST = np.array(vect_ST)
    option_price_estimate = 1/N*np.exp(-r*T)*sum(payoff(vect_ST, K, type_value))
    return option_price_estimate

# Building the price grid$
price_grid = np.zeros((len(r_values), len(sigma_values)))

for i, r in enumerate(r_values):
    for j, sigma in enumerate(sigma_values):
        price= option_pricing(N, r, sigma, K, T, type_value=type_value, S0=100)
        price_grid[i, j] = price


# Plot heatmap
fig, ax = plt.subplots(figsize=(14,10))
# cax = ax.imshow(price_grid, origin='lower',
#                 extent=[sigma_values[0], sigma_values[-1], r_values[0], r_values[-1]],
#                 aspect='auto', cmap='viridis')
cax = ax.imshow(price_grid, origin='lower', aspect='auto', cmap='viridis')
ax.set_xticks(range(len(sigma_values)))
ax.set_xticklabels([f"{s:.2f}" for s in sigma_values])
ax.set_yticks(range(len(r_values)))
ax.set_yticklabels([f"{r:.2f}" for r in r_values])

fig.colorbar(cax, label='Call option price')

# Annotate each cell at the center
for i in range(len(r_values)):
    for j in range(len(sigma_values)):
        price = price_grid[i, j]
        # Use pixel indices as coordinates
        ax.text(j, i, f"{price:.2f}", color='black', ha='center', va='center', fontsize=12)

plt.xlabel('Volatility σ')
plt.ylabel('Risk-free rate r')
plt.title('European Call Option Price Grid T=2, K=110')
plt.show()

ax.set_xlabel('Volatility σ')
ax.set_ylabel('Risk-free rate r')
ax.set_title('European Call Option Price Grid')
st.pyplot(fig)
