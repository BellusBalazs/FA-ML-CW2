import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the tickers and the time period
tickers = ['AAPL', 'JNJ', 'XOM', 'JPM', 'PG', 'HD', 'BA', 'NEM', 'NEE', 'AMT']
start_date = '2015-05-25'
end_date = '2025-05-25'

# Download adjusted close price data
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Calculate equally weighted portfolio returns
portfolio_returns = daily_returns.mean(axis=1)

# Calculate cumulative returns
cumulative_returns = (1 + portfolio_returns).cumprod()

# Set initial investment value
initial_investment = 1000

# Calculate portfolio value over time
portfolio_value = cumulative_returns * initial_investment

# --- Plot 1: Portfolio value ---
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label='Equally Weighted Portfolio Value')
plt.title('Portfolio Value of Equally Weighted Portfolio (2015–2025)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Normalized stock prices ---
normalized_prices = data / data.iloc[0]  # Normalize to 1 at start

plt.figure(figsize=(12, 7))
for ticker in tickers:
    plt.plot(normalized_prices[ticker], label=ticker)

plt.title('Normalized Stock Prices (2015–2025)')
plt.xlabel('Date')
plt.ylabel('Normalized Price (Start = 1)')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
