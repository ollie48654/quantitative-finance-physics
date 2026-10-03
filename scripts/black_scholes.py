import numpy as np
from scipy.stats import norm
import pandas as pd
from pathlib import Path

EXPORTS_DIR = Path(__file__).parent.parent / 'exports'
EXPORTS_DIR.mkdir(exist_ok=True)

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return round(float(price), 4)

def greeks(S, K, T, r, sigma):
    if T <= 0:
        return {'delta':0,'gamma':0,'theta':0,'vega':0,'rho':0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return {
        'delta': round(delta, 4), 'gamma': round(gamma, 6),
        'theta': round(theta, 4), 'vega': round(vega, 4),
        'rho':   round(rho, 4)
    }

def price_surface(S, r=0.05, sigma=0.25):
    strikes    = np.linspace(S * 0.7, S * 1.3, 30)
    maturities = np.linspace(0.05, 2.0, 30)
    results = []
    for K in strikes:
        for T in maturities:
            results.append({
                'strike':         round(K, 2),
                'maturity_years': round(T, 3),
                'call_price':     black_scholes(S, K, T, r, sigma, 'call'),
                'put_price':      black_scholes(S, K, T, r, sigma, 'put'),
                'moneyness':      round(S / K, 3)
            })
    df = pd.DataFrame(results)
    df.to_csv(EXPORTS_DIR / 'option_surface.csv', index=False)
    print(f"Saved option_surface.csv — {len(df)} price points")
    return df

if __name__ == '__main__':
    S, K, T, r, sigma = 189.45, 190.0, 0.5, 0.05, 0.289
    call = black_scholes(S, K, T, r, sigma, 'call')
    put  = black_scholes(S, K, T, r, sigma, 'put')
    g    = greeks(S, K, T, r, sigma)
    print(f"AAPL Call: ${call}  Put: ${put}")
    print(f"Greeks: {g}")
    price_surface(S, r, sigma)