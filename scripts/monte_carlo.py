import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

EXPORTS_DIR = Path(__file__).parent.parent / 'exports'
DOCS_DIR    = Path(__file__).parent.parent / 'docs'
EXPORTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({'font.family':'sans-serif','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

def simulate_gbm(S0, mu, sigma, T=1.0, dt=1/252, n_paths=10000, seed=42):
    np.random.seed(seed)
    n_steps = int(T / dt)
    Z = np.random.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    paths = np.hstack([np.full((n_paths, 1), S0), paths])
    return paths

def price_option_mc(S0, K, T, r, sigma, option_type='call', n_paths=100000, seed=42):
    paths  = simulate_gbm(S0, r, sigma, T=T, n_paths=n_paths, seed=seed)
    finals = paths[:, -1]
    payoffs = (np.maximum(finals - K, 0) if option_type == 'call'
               else np.maximum(K - finals, 0))
    price   = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    return round(float(price), 4), round(float(std_err), 4)

def convergence_study(S0, K, T, r, sigma):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.black_scholes import black_scholes as bs_price
    bs = bs_price(S0, K, T, r, sigma, 'call')
    path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
    results = []
    for n in path_counts:
        price, se = price_option_mc(S0, K, T, r, sigma, n_paths=n)
        results.append({'n_paths':n,'mc_price':price,'std_error':se,'bs_price':bs,'abs_error':abs(price-bs)})
        print(f"  n={n:>7,}: MC=${price:.4f}  Error vs BS={abs(price-bs):.4f}")
    pd.DataFrame(results).to_csv(EXPORTS_DIR / 'mc_convergence.csv', index=False)

def plot_paths(paths, ticker='AAPL'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t = np.linspace(0, 1, paths.shape[1])
    for i in range(200):
        axes[0].plot(t, paths[i], alpha=0.05, color='#185FA5', linewidth=0.5)
    axes[0].plot(t, np.percentile(paths, 5,  axis=0), 'r--', lw=1.5, label='5th pct')
    axes[0].plot(t, np.percentile(paths, 50, axis=0), 'k-',  lw=2.0, label='Median')
    axes[0].plot(t, np.percentile(paths, 95, axis=0), 'g--', lw=1.5, label='95th pct')
    axes[0].set_title(f'{ticker} — 10,000 Monte Carlo Paths\n(Geometric Brownian Motion)', fontweight='bold')
    axes[0].set_xlabel('Time (years)')
    axes[0].set_ylabel('Stock Price ($)')
    axes[0].legend()
    final = paths[:, -1]
    axes[1].hist(final, bins=80, color='#185FA5', alpha=0.75, edgecolor='white', linewidth=0.3)
    axes[1].axvline(np.mean(final), color='red', linewidth=2, label=f'Mean: ${np.mean(final):.2f}')
    axes[1].set_title('Distribution of Final Prices\n(Log-Normal)', fontweight='bold')
    axes[1].set_xlabel('Final Stock Price ($)')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'monte_carlo_paths.png', bbox_inches='tight')
    plt.close()
    print("Saved monte_carlo_paths.png")

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    S0, K, T, r, sigma = 189.45, 190.0, 0.5, 0.05, 0.289
    print("Simulating 10,000 GBM paths...")
    paths = simulate_gbm(S0, mu=r, sigma=sigma, T=T, n_paths=10000)
    plot_paths(paths, ticker='AAPL')
    print("\nConvergence study:")
    convergence_study(S0, K, T, r, sigma)