import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from trading_env import TradingEnv
from ppo_agent import PPOAgent

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    return data.ffill()

def compute_sharpe(returns):
    mean = np.mean(returns)
    std = np.std(returns)
    return mean / std * np.sqrt(252)

def simulate(env, agent, reward_type):
    obs = env.reset()
    done = False
    equity_curve = [env.balance]
    state_buffer, action_buffer, reward_buffer, prob_buffer = [], [], [], []

    while not done:
        action, probs = agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)

        if reward_type == "basic":
            shaped_reward = reward
        elif reward_type == "risk_penalty":
            shaped_reward = reward - 0.5 * np.std(env.df.pct_change().iloc[env.current_step])
        elif reward_type == "drawdown_penalty":
            max_balance = max(equity_curve)
            drawdown = (max_balance - env.balance) / max_balance
            shaped_reward = reward - drawdown
        else:
            raise ValueError("Unknown reward type")

        state_buffer.append(obs)
        action_buffer.append(action)
        reward_buffer.append(shaped_reward)
        prob_buffer.append(probs)

        obs = next_obs
        equity_curve.append(env.balance)

    agent.train(state_buffer, action_buffer, reward_buffer, prob_buffer)
    return equity_curve, reward_buffer

def run_experiment():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    reward_types = ['basic', 'risk_penalty', 'drawdown_penalty']
    market_periods = {
        'Bull Market (2017)': ('2017-01-01', '2018-01-01'),
        'Bear Market (2022)': ('2022-01-01', '2023-01-01'),
        'Stagnation (2015)': ('2015-01-01', '2016-01-01'),
    }

    epochs = 30
    window_size = 10
    results = {}

    for period_name, (start, end) in market_periods.items():
        df = download_data(tickers, start, end)
        returns_df = df.pct_change().dropna()
        equally_weighted_returns = returns_df.mean(axis=1)
        benchmark_equity = (1 + equally_weighted_returns).cumprod()

        results[period_name] = {}
        print(f"\n📅 {period_name}")

        for reward_type in reward_types:
            print(f"🧪 Training PPO with reward: {reward_type}")
            agent = PPOAgent(n_assets=len(tickers), window_size=window_size)
            env = TradingEnv(df, window_size=window_size)

            for _ in range(epochs):
                simulate(env, agent, reward_type)

            # Test the final agent
            test_env = TradingEnv(df, window_size=window_size)
            final_equity, final_rewards = simulate(test_env, agent, reward_type)
            sharpe = compute_sharpe(final_rewards)

            results[period_name][reward_type] = {
                'equity': final_equity,
                'sharpe': sharpe
            }

            print(f"🔹 Sharpe Ratio: {sharpe:.2f}")

        # Add benchmark to results
        results[period_name]['benchmark'] = benchmark_equity

    # Plotting
    for period_name in market_periods:
        plt.figure(figsize=(10, 5))
        for reward_type in reward_types:
            plt.plot(results[period_name][reward_type]['equity'], label=f"{reward_type} (Sharpe: {results[period_name][reward_type]['sharpe']:.2f})")
        plt.plot(results[period_name]['benchmark'].values * 1000, 'k--', label='Buy & Hold (Equally Weighted)')
        plt.title(f"Equity Curves - {period_name}")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_experiment()
