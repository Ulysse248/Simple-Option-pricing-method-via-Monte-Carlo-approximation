import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm


class MonteCarloPricing():
    def __init__(self, N, r, sigma, K, T, type_value, S0):
        self.N = N            # number of Monte Carlo iterations
        self.r = r            # risk-free interest rate
        self.sigma = sigma    # volatility
        self.K = K
        self.T = T            # time to maturity
        self.type_value = type_value  # 1 for call, -1 for put
        self.S0 = S0          # initial asset price
    
        # Payoff function
    def payoff(self, s, K, type): # 1 is call, -1 is put
        temp = type*(s-K)
        temp[temp < 0]=0
        return temp

    # Option pricing function
    def option_pricing(self):
        temp_ST_factor = self.S0 * np.exp((self.r-0.5*self.sigma**2)*self.T) 
        vect_ST = []
        Z = np.random.randn(self.N)  # generate N standard normal random numbers
        vect_ST = temp_ST_factor * np.exp(self.sigma * np.sqrt(self.T) * Z)

        option_price_estimate = 1/self.N*np.exp(-self.r*self.T)*sum(self.payoff(vect_ST, self.K, self.type_value))
        return option_price_estimate, vect_ST

    def variance(self, vect_ST):
        vect_payoff = self.payoff(vect_ST, self.K, self.type_value)
        mean_vect_payoff = 1/self.N*sum(vect_payoff)
        return 1/self.N*np.exp(-2*self.r*self.T)/(self.N-1)*(sum((vect_payoff-mean_vect_payoff)**2))

class BlackScholesPricing():
    def __init__(self, r, sigma, K, T, type_value, S0):
        self.r = r            # risk-free interest rate
        self.sigma = sigma    # volatility
        self.K = K
        self.T = T            # time to maturity
        self.type_value = type_value  # 1 for call, -1 for put
        self.S0 = S0          # initial asset price

    def option_pricing(self):
        d1 = (np.log(self.S0/self.K) + (self.r+1/2*self.sigma**2)*self.T) / (self.sigma * np.sqrt(self.T)) 
        d2 = d1 - self.sigma * np.sqrt(self.T)
        phi_d1 = norm.cdf(d1)
        phi_d2 = norm.cdf(d2)

        if self.type_value==1: # Call option 
            return self.S0*phi_d1 - self.K*np.exp(-self.r*self.T)*phi_d2, []
        else:                  # Put Option (type_value = -1)
            return -self.S0*(1-phi_d1) + self.K*np.exp(-self.r*self.T)*(1-phi_d2), []

def plot_grid(r_values, sigma_values, PricingMethod):
    # Building the price grid$
    price_grid = np.zeros((len(r_values), len(sigma_values)))
    variance_grid = np.zeros((len(r_values), len(sigma_values)))

    IsMonteCarlo =  isinstance(PricingMethod, MonteCarloPricing)
    for i, r in enumerate(r_values):
        PricingMethod.r = r
        for j, sigma in enumerate(sigma_values):
            PricingMethod.sigma = sigma
            price, vect_ST = PricingMethod.option_pricing()
            price_grid[i, j] = price
            if IsMonteCarlo:            
                variance = PricingMethod.variance(vect_ST)
                variance_grid[i,j] = variance
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

    fig.colorbar(cax, label='Option price')

    # Annotate each cell at the center
    for i in range(len(r_values)):
        for j in range(len(sigma_values)):
            price = price_grid[i, j]
            # Use pixel indices as coordinates
            ax.text(j, i, f"{price:.2f}", color='black', ha='center', va='center', fontsize=12)
            if IsMonteCarlo:
                variance = variance_grid[i,j]
                ax.text(j, i-0.3, f"{np.sqrt(variance):.2f}", color='black', ha='center', va='center', fontsize=9)

    plt.xlabel('Volatility σ')
    plt.ylabel('Risk-free rate r')
    plt.title('European Option Price Grid')
    plt.show()

    IsBlackScholes = isinstance(PricingMethod, BlackScholesPricing)
    ax.set_xlabel('Volatility σ')
    ax.set_ylabel('Risk-free rate r')
    plt.title(f"{'Monte Carlo estimation of' if IsMonteCarlo else 'Black-Scholes model' if IsBlackScholes else ''} "
        f"European Option Price {'(and std) ' if IsMonteCarlo else ''}Grid")
    st.pyplot(fig)  

