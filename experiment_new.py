import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_env_new import TradingEnv


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


def run_experiment():
    tickers = ['AAPL', 'JNJ', 'XOM', 'JPM', 'PG', 'HD', 'BA', 'NEM', 'NEE', 'AMT']
    start_date = '2020-05-25'
    end_date = '2025-05-25'

    df = download_data(tickers, start_date, end_date)

    # Clean data
    df = df.dropna(how='all').ffill().bfill()
    valid_mask = df.notna().all(axis=1)
    valid_dates = df.index[valid_mask]
    df = df.loc[valid_dates[0]:valid_dates[-1]]

    window_size = 10
    total_timesteps = 30000  # reduced for speed but still meaningful

    # reward_types = ['basic', 'utility', 'risk_penalty', 'drawdown_penalty']
    reward_types = ['basic']  # you can add others if you want
    results = {}

    for reward_type in reward_types:
        print(f"\n=== Starting training with reward_type = '{reward_type}' ===")
        start_time = time.time()

        env = DummyVecEnv([lambda: TradingEnv(df, window_size=window_size, reward_type=reward_type)])

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

        # Simple progress logging callback for training
        class ProgressCallback:
            def __init__(self, total_steps, log_interval=5000):
                self.total_steps = total_steps
                self.log_interval = log_interval
                self.last_logged = 0

            def __call__(self, locals_, globals_):
                num_timesteps = locals_['self'].num_timesteps
                if num_timesteps - self.last_logged >= self.log_interval:
                    pct = (num_timesteps / self.total_steps) * 100
                    print(f"Training progress: {num_timesteps}/{self.total_steps} steps ({pct:.1f}%)")
                    self.last_logged = num_timesteps
                return True

        progress_callback = ProgressCallback(total_timesteps)

        model.learn(total_timesteps=total_timesteps, callback=progress_callback)

        training_time = time.time() - start_time
        print(f"Training finished in {training_time:.1f} seconds.")

        # Test phase
        test_env = TradingEnv(df, window_size=window_size, reward_type=reward_type)
        obs = test_env.reset()
        done = False
        equity_curve = [test_env.balance]
        weights_over_time = []
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # DEBUG: print action shape & content once (optional)
            if step == 0:
                print(f"Action shape: {action.shape}, action: {action}")

            # Append weights robustly depending on shape
            if len(action.shape) == 1:
                weights_over_time.append(action)
            else:
                weights_over_time.append(action[0])

            obs, reward, done, _ = test_env.step(action)
            equity_curve.append(test_env.balance)
            step += 1
            if step % 50 == 0 or done:
                print(f"Testing progress ({reward_type}): Step {step}, Balance: {test_env.balance:.2f}")

        weights_over_time = np.array(weights_over_time)

        portfolio_returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = compute_sharpe(portfolio_returns)

        results[reward_type] = {
            'equity_curve': equity_curve,
            'sharpe': sharpe,
            'training_time': training_time,
            'weights_over_time': weights_over_time
        }

        print(f"Reward: {reward_type} | Sharpe ratio: {sharpe:.3f}\n")

    # Plot all equity curves together
    plt.figure(figsize=(14, 7))
    for reward_type, res in results.items():
        plt.plot(res['equity_curve'], label=f'{reward_type} (Sharpe: {res["sharpe"]:.2f})')

    # Benchmark: equal weighted buy & hold
    adj_close = df.xs('Close', axis=1, level=1)
    returns_df = adj_close.pct_change().dropna()
    equally_weighted_returns = returns_df.mean(axis=1)
    benchmark_equity = (1 + equally_weighted_returns).cumprod() * 1000
    plt.plot(benchmark_equity.values, 'k--', label='Buy & Hold Equally Weighted')

    plt.title('Portfolio Equity Curve Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot weights over time for each reward type
    for reward_type, res in results.items():
        weights = res['weights_over_time']
        plt.figure(figsize=(14, 7))
        for i in range(weights.shape[1]):
            plt.plot(weights[:, i], label=f'Asset {tickers[i]}')
        plt.title(f'Portfolio Weights Over Time ({reward_type})')
        plt.xlabel('Time Step')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_experiment()
