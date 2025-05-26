import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env_new_long import TradingEnv


def download_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    panel = {}
    for ticker in tickers:
        panel[ticker] = raw[ticker][['High', 'Low', 'Close', 'Volume']]
    df = pd.concat(panel.values(), axis=1, keys=panel.keys())
    return df.ffill()


def compute_sharpe(returns):
    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / (std_return + 1e-8) * np.sqrt(252)


def run_single_experiment(tickers, train_start, train_end, test_start, test_end, label):
    print(f"\n=== Running Experiment: {label} ===")
    df_full = download_data(tickers, start=train_start, end=test_end)

    # Preprocess
    df_full = df_full.dropna(how='all').ffill().bfill()
    valid_mask = df_full.notna().all(axis=1)
    df_full = df_full.loc[valid_mask]

    df_train = df_full.loc[train_start:train_end]
    df_test = df_full.loc[test_start:test_end]

    window_size = 10
    total_timesteps = 30000

    reward_types = ['basic', 'utility', 'risk_penalty', 'drawdown_penalty']

    results = {}

    for reward_type in reward_types:
        print(f"\n--- Training with reward type: {reward_type} ---")
        env = DummyVecEnv([lambda: TradingEnv(df_train, window_size=window_size, reward_type=reward_type)])

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            batch_size=64,
            n_steps=1024,
            learning_rate=3e-4,
            ent_coef=0.01,
            clip_range=0.2,
            seed=42,
        )

        model.learn(total_timesteps=total_timesteps)

        # Test phase
        test_env = TradingEnv(df_test, window_size=window_size, reward_type=reward_type)
        obs = test_env.reset()
        done = False
        equity_curve = [test_env.balance]
        weights_over_time = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            weights_over_time.append(action if action.ndim == 1 else action[0])
            obs, reward, done, _ = test_env.step(action)
            equity_curve.append(test_env.balance)

        portfolio_returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = compute_sharpe(portfolio_returns)

        results[reward_type] = {
            'equity_curve': equity_curve,
            'weights': np.array(weights_over_time),
            'sharpe': sharpe,
        }

        print(f"Reward: {reward_type} | Sharpe: {sharpe:.2f}")

    # Plot all equity curves for this experiment
    plt.figure(figsize=(12, 6))
    for rt, res in results.items():
        plt.plot(res['equity_curve'], label=f'{rt} (Sharpe: {res["sharpe"]:.2f})')

    adj_close = df_test.xs('Close', axis=1, level=1)
    returns_df = adj_close.pct_change().dropna()
    equal_weight_returns = returns_df.mean(axis=1)
    benchmark = (1 + equal_weight_returns).cumprod() * 1000
    plt.plot(benchmark.values, 'k--', label='Buy & Hold Equally Weighted')

    plt.title(f'Equity Curve Comparison: {label}')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot weights for each reward type
    window = 10
    for rt, res in results.items():
        weights_smooth = pd.DataFrame(res['weights'], columns=tickers).rolling(window=window, min_periods=1).mean().values
        plt.figure(figsize=(14, 6))
        plt.stackplot(
            range(len(weights_smooth)),
            weights_smooth.T,
            labels=tickers,
            alpha=0.8
        )
        plt.title(f'Smoothed Portfolio Weights Over Time ({label} - {rt})')
        plt.xlabel('Time Step')
        plt.ylabel('Weight')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def run_all_experiments():
    tickers = ['SPY', 'EFA', 'TLT', 'GLD', 'BTC-USD']

    # COVID Crash Period (Short Crisis Test)
    run_single_experiment(
        tickers=tickers,
        train_start='2017-01-01',
        train_end='2020-02-14',
        test_start='2020-02-15',
        test_end='2020-04-15',
        label='COVID Crash Period'
    )

    # Bull Market Year 2023
    run_single_experiment(
        tickers=tickers,
        train_start='2018-01-01',
        train_end='2022-12-31',
        test_start='2023-01-01',
        test_end='2023-12-31',
        label='Bull Market 2023'
    )


if __name__ == "__main__":
    run_all_experiments()
