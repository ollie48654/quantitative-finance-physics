import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

DATA_DIR    = Path(__file__).parent.parent / 'data'
EXPORTS_DIR = Path(__file__).parent.parent / 'exports'
DOCS_DIR    = Path(__file__).parent.parent / 'docs'

plt.rcParams.update({'font.family':'sans-serif','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

TICKERS = ['AAPL','MSFT','NVDA','JPM','GOOGL']

def load_returns():
    df = pd.read_csv(DATA_DIR / 'prices.csv', parse_dates=['date'])
    pivot = df.pivot(index='date', columns='ticker', values='close')
    return pivot.pct_change().dropna()[TICKERS]

def portfolio_stats(weights, mean_returns, cov_matrix, rf=0.05):
    w      = np.array(weights)
    ret    = np.dot(w, mean_returns) * 252
    vol    = np.sqrt(w @ cov_matrix @ w) * np.sqrt(252)
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe

def value_at_risk(returns, weights, confidence=0.95):
    port_ret = returns @ weights
    var_pct  = np.percentile(port_ret, (1 - confidence) * 100)
    cvar     = port_ret[port_ret <= var_pct].mean()
    return {
        'var_95_daily':  round(float(var_pct * 100), 4),
        'cvar_95_daily': round(float(cvar * 100), 4),
        'interpretation': f"On 95% of days, daily loss < {abs(var_pct*100):.2f}%"
    }

def efficient_frontier(returns, n_portfolios=5000):
    mean_ret = returns.mean()
    cov_mat  = returns.cov()
    n        = len(TICKERS)
    rets, vols, sharpes, weights = [], [], [], []
    np.random.seed(42)
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_stats(w, mean_ret, cov_mat)
        rets.append(r); vols.append(v); sharpes.append(s); weights.append(w)
    results = pd.DataFrame({
        'return':rets,'vol':vols,'sharpe':sharpes,
        **{f'w_{t}':[w[i] for w in weights] for i,t in enumerate(TICKERS)}
    })
    return results, mean_ret, cov_mat

def find_optimal_portfolio(mean_returns, cov_matrix, rf=0.05):
    n = len(TICKERS)
    result = minimize(
        lambda w: -portfolio_stats(w, mean_returns, cov_matrix, rf)[2],
        np.array([1/n]*n), method='SLSQP',
        bounds=tuple((0,1) for _ in range(n)),
        constraints={'type':'eq','fun':lambda w: np.sum(w)-1}
    )
    opt_w = result.x
    r, v, s = portfolio_stats(opt_w, mean_returns, cov_matrix, rf)
    print("\nOptimal Portfolio (Max Sharpe):")
    for ticker, w in zip(TICKERS, opt_w):
        print(f"  {ticker}: {w*100:.1f}%")
    print(f"  Return: {r*100:.2f}%  Vol: {v*100:.2f}%  Sharpe: {s:.3f}")
    return opt_w, r, v, s

def plot_efficient_frontier(results, opt_r, opt_v):
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(results['vol']*100, results['return']*100,
                    c=results['sharpe'], cmap='viridis', alpha=0.5, s=8)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')
    ax.scatter(opt_v*100, opt_r*100, color='red', s=200, zorder=5, marker='*', label='Max Sharpe')
    ax.set_xlabel('Annual Volatility (%)')
    ax.set_ylabel('Annual Expected Return (%)')
    ax.set_title('Efficient Frontier — 5,000 Random Portfolios', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'efficient_frontier.png', bbox_inches='tight')
    plt.close()
    print("Saved efficient_frontier.png")

def plot_var(returns, equal_w, opt_w):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, weights, title in [
        (axes[0], equal_w, 'Equal-Weight Portfolio'),
        (axes[1], opt_w,   'Optimal Portfolio')
    ]:
        port_ret = returns @ weights
        var95    = np.percentile(port_ret, 5)
        ax.hist(port_ret*100, bins=80, color='#185FA5', alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(var95*100, color='red', linewidth=2, linestyle='--', label=f'95% VaR: {var95*100:.2f}%')
        ax.set_title(f'{title}\nDaily Return Distribution', fontweight='bold')
        ax.set_xlabel('Daily Return (%)')
        ax.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'var_distribution.png', bbox_inches='tight')
    plt.close()
    print("Saved var_distribution.png")

if __name__ == '__main__':
    print("Loading return data...")
    returns = load_returns()
    equal_w = np.array([0.2]*5)
    var_res = value_at_risk(returns, equal_w)
    print(f"VaR: {var_res['interpretation']}")
    results, mean_ret, cov_mat = efficient_frontier(returns)
    opt_w, opt_r, opt_v, opt_s = find_optimal_portfolio(mean_ret, cov_mat)
    results.to_csv(EXPORTS_DIR / 'efficient_frontier.csv', index=False)
    plot_efficient_frontier(results, opt_r, opt_v)
    plot_var(returns, equal_w, opt_w)
    print("Done.")