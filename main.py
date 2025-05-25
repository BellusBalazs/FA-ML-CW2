import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from trading_env import TradingEnv
from ppo_agent_long_only import PPOAgent


def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    return data.ffill()


def compute_sharpe(returns):
    returns = np.array(returns)
    excess_returns = returns - 0.0  # Assume 0 risk-free rate
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

    while not done:
        action, probs = agent.get_action(obs)
        actions_taken.append(action)
        next_obs, reward, done, _ = env.step(action)

        basic_rewards.append(reward)

        if train:
            if reward_type == "basic":
                shaped_reward = reward
            elif reward_type == "risk_penalty":
                current_returns = env.df.pct_change().iloc[env.current_step]
                shaped_reward = reward - 0.5 * np.std(current_returns)
            elif reward_type == "drawdown_penalty":
                max_balance = max(equity_curve)
                drawdown = (max_balance - env.balance) / max_balance if max_balance > 0 else 0
                shaped_reward = reward - drawdown
            else:
                raise ValueError("Unknown reward type")
        else:
            shaped_reward = reward  # No shaping in test

        state_buffer.append(obs)
        action_buffer.append(action)
        reward_buffer.append(shaped_reward)
        prob_buffer.append(probs)

        obs = next_obs
        equity_curve.append(env.balance)

    if train:
        agent.train(state_buffer, action_buffer, reward_buffer, prob_buffer)

    return equity_curve, basic_rewards, actions_taken


def run_experiment():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    reward_types = ['basic', 'risk_penalty', 'drawdown_penalty']
    market_periods = {
        'Bull Market (2017-2019)': ('2017-01-01', '2020-01-01'),
        'Bear Market (2022)': ('2022-01-01', '2023-01-01'),
        'Stagnation (2015)': ('2015-01-01', '2016-01-01'),
        'Recent 5 Years': ('2019-01-01', '2024-12-31')
    }

    window_size = 30
    epochs = 30
    results = {}

    for period_name, (start, end) in market_periods.items():
        df = download_data(tickers, start, end)
        returns_df = df.pct_change().dropna()
        equally_weighted_returns = returns_df.mean(axis=1)
        benchmark_equity = (1 + equally_weighted_returns).cumprod() * 1000

        results[period_name] = {}
        print(f"\n📅 {period_name}")

        for reward_type in reward_types:
            print(f"🧪 Training PPO with reward: {reward_type}")
            agent = PPOAgent(n_assets=len(tickers), window_size=window_size)
            env = TradingEnv(df, window_size=window_size)

            for _ in range(epochs):
                simulate(env, agent, reward_type, train=True)

            # Evaluate
            test_env = TradingEnv(df, window_size=window_size)
            equity_curve, basic_rewards, actions = simulate(test_env, agent, reward_type, train=False)
            sharpe = compute_sharpe(basic_rewards)

            results[period_name][reward_type] = {
                'equity': equity_curve,
                'sharpe': sharpe,
                'actions': actions
            }

            print(f"🔹 {reward_type} | Sharpe: {sharpe:.2f} | Final Value: {equity_curve[-1]:.2f}")

        # Save benchmark
        results[period_name]['benchmark'] = benchmark_equity

    # Plot
    for period_name in market_periods:
        # Plot equity curves
        plt.figure(figsize=(12, 5))
        for reward_type in reward_types:
            strategy = results[period_name][reward_type]
            plt.plot(strategy['equity'], label=f"{reward_type} (Sharpe: {strategy['sharpe']:.2f})")
        plt.plot(results[period_name]['benchmark'].values, 'k--', label='Buy & Hold (Equally Weighted)')
        plt.title(f"Equity Curves - {period_name}")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_experiment()
