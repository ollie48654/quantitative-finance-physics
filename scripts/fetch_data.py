import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'GOOGL']

def fetch_prices(period='5y'):
    print(f"Fetching {period} of price data for: {TICKERS}")
    frames = []
    for ticker in TICKERS:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        df['ticker'] = ticker
        df = df[['date','ticker','open','high','low','close','volume']]
        frames.append(df)
        print(f"  {ticker}: {len(df)} trading days")
    combined = pd.concat(frames, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values(['ticker','date'])
    combined['daily_return'] = combined.groupby('ticker')['close'].pct_change()
    combined.to_csv(DATA_DIR / 'prices.csv', index=False)
    print(f"Saved {len(combined):,} rows to data/prices.csv")
    return combined

def get_volatility_params():
    df = pd.read_csv(DATA_DIR / 'prices.csv', parse_dates=['date'])
    params = {}
    for ticker in TICKERS:
        sub = df[df['ticker'] == ticker]['daily_return'].dropna()
        params[ticker] = {
            'mu':         float(sub.mean() * 252),
            'sigma':      float(sub.std() * np.sqrt(252)),
            'last_price': float(df[df['ticker'] == ticker]['close'].iloc[-1])
        }
        print(f"  {ticker}: mu={params[ticker]['mu']:.3f}, sigma={params[ticker]['sigma']:.3f}, S0=${params[ticker]['last_price']:.2f}")
    return params

if __name__ == '__main__':
    fetch_prices()
    print("\nAnnualised parameters:")
    get_volatility_params()