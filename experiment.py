import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from trading_env import TradingEnv
from ppo_agent_long_only import PPOAgent

def download_data_ohlcv(tickers, start, end):
    raw_data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    panel = {}
    for ticker in tickers:
        panel[ticker] = raw_data[ticker][['High', 'Low', 'Close', 'Volume']]
    df = pd.concat(panel.values(), axis=1, keys=panel.keys())
    return df.ffill()

def compute_sharpe(returns):
    returns = np.array(returns)
    excess_returns = returns - 0.0
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    return mean_return / (std_return + 1e-8) * np.sqrt(252)

def simulate(env, agent, reward_type, train=False):
    obs = env.reset()
    done = False
    equity_curve = [env.balance]
    state_buffer, action_buffer, reward_buffer, prob_buffer = [], [], [], []
    basic_rewards = []
    actions_taken = []
    asset_weights_over_time = []

    while not done:
        action, probs = agent.get_action(obs)
        actions_taken.append(action)
        next_obs, reward, done, _ = env.step(action)
        basic_rewards.append(reward)

        if train:
            if reward_type == "basic":
                shaped_reward = reward
            elif reward_type == "utility":
                wealth_return = (env.balance - equity_curve[-1]) / max(equity_curve[-1], 1e-8)
                shaped_reward = np.log(1 + wealth_return)
            elif reward_type == "risk_penalty":
                current_returns = env.df.xs('Close', axis=1, level=1).pct_change().iloc[env.current_step]
                shaped_reward = reward - 0.5 * np.std(current_returns)
            elif reward_type == "drawdown_penalty":
                max_balance = max(equity_curve)
                drawdown = (max_balance - env.balance) / max_balance if max_balance > 0 else 0
                shaped_reward = reward - drawdown
            else:
                raise ValueError("Unknown reward type")
        else:
            shaped_reward = reward

        state_buffer.append(obs)
        action_buffer.append(action)
        reward_buffer.append(shaped_reward)
        prob_buffer.append(probs)

        obs = next_obs
        equity_curve.append(env.balance)
        asset_weights_over_time.append(action)

    if train:
        agent.train(state_buffer, action_buffer, reward_buffer, prob_buffer)

    return equity_curve, basic_rewards, actions_taken, asset_weights_over_time

def run_experiment():
    tickers = ['AAPL', 'JNJ', 'XOM', 'JPM', 'PG', 'HD', 'BA', 'NEM', 'NEE', 'AMT']
    reward_types = ['basic'] #Change back!!
    start, end = '2019-01-01', '2025-01-01'

    window_size = 10
    epochs = 30
    results = {}
    df = download_data_ohlcv(tickers, start, end)
    df = pd.DataFrame(df)

    print("Shape before: " + str(df.shape))
    df = df.dropna(how='all')
    df = df.ffill().bfill()
    valid_mask = df.notna().all(axis=1)
    valid_dates = df.index[valid_mask]

    if len(valid_dates) == 0:
        raise ValueError("No time period where all tickers have full data on all features.")

    start_full = valid_dates[0]
    end_full = valid_dates[-1]

    print(f"Full data available from {start_full.date()} to {end_full.date()}")
    df = df.loc[start_full:end_full]
    print("Shape after restricting to full data period: " + str(df.shape))

    adj_close = df.xs('Close', axis=1, level=1)
    returns_df = adj_close.pct_change().dropna()
    equally_weighted_returns = returns_df.mean(axis=1)
    benchmark_equity = (1 + equally_weighted_returns).cumprod() * 1000

    results['Recent Period'] = {}

    for reward_type in reward_types:
        print(f"Training PPO with reward: {reward_type}")
        agent = PPOAgent(n_assets=len(tickers), window_size=window_size, features_per_asset=4)
        env = TradingEnv(df, window_size=window_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            simulate(env, agent, reward_type, train=True)

        test_env = TradingEnv(df, window_size=window_size)
        equity_curve, basic_rewards, actions, asset_weights_over_time = simulate(test_env, agent, reward_type, train=False)
        sharpe = compute_sharpe(basic_rewards)

        results['Recent Period'][reward_type] = {
            'equity': equity_curve,
            'sharpe': sharpe,
            'actions': actions,
            'asset_weights': asset_weights_over_time
        }

        print(f"{reward_type} | Sharpe: {sharpe:.2f} | Final Value: {equity_curve[-1]:.2f}")

    results['Recent Period']['benchmark'] = benchmark_equity

    plt.figure(figsize=(12, 5))
    for reward_type in reward_types:
        strategy = results['Recent Period'][reward_type]
        plt.plot(strategy['equity'], label=f"{reward_type} (Sharpe: {strategy['sharpe']:.2f})")
    plt.plot(results['Recent Period']['benchmark'].values, 'k--', label='Buy & Hold (Equally Weighted)')
    plt.title("Equity Curves - Recent Period")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for reward_type in reward_types:
        plt.figure(figsize=(12, 5))
        weights_array = np.array(results['Recent Period'][reward_type]['asset_weights'])
        for asset_idx, ticker in enumerate(tickers):
            plt.plot(weights_array[:, asset_idx], label=ticker)
        plt.title(f"Asset Weights Over Time - Strategy: {reward_type}")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Weight")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_experiment()
