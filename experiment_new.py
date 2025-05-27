import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.stats import ttest_ind
from portfolio_data import tickers
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

def run_experiment():
    tickers = ['AAPL', 'JNJ', 'XOM', 'JPM', 'PG', 'HD', 'BA', 'NEM', 'NEE', 'AMT']

    start_date = '2015-01-01'
    end_date = '2025-01-01'
    split_date = '2021-01-01'  # train before this, test after this

    df = download_data(tickers, start=start_date, end=end_date)
    df = df.dropna(how='all').ffill().bfill()
    valid_mask = df.notna().all(axis=1)
    valid_dates = df.index[valid_mask]
    df = df.loc[valid_dates[0]:valid_dates[-1]]

    # Split into train/test
    df_train = df[df.index < split_date]
    df_test = df[df.index >= split_date]

    window_size = 10
    total_timesteps = 30000

    reward_types = ['basic', 'utility', 'risk_penalty', 'drawdown_penalty']
    results = {}

    # Define best hyperparameters per reward type
    best_params_map = {
        "basic": {
            "window_size": 10,
            "learning_rate": 0.0003781883305632542,
            "ent_coef": 0.0029601365379751574,
            "clip_range": 0.2808970044622227,
            "gamma": 0.9916943747451629,
            "batch_size": 96,
        },
        "utility": {
            "window_size": 10,
            "learning_rate": 0.0005857771202038145,
            "ent_coef": 0.00013946547206372564,
            "clip_range": 0.13242027579299662,
            "gamma": 0.9873288747259636,
            "batch_size": 128,
        },
        "risk_penalty": {
            "window_size": 10,
            "learning_rate": 0.0007244999961292659,
            "ent_coef": 0.005664041210269082,
            "clip_range": 0.19428946199026847,
            "gamma": 0.9775287191580475,
            "batch_size": 32,
        },
        "drawdown_penalty": {
            "window_size": 10,
            "learning_rate": 0.00030996579474128487,
            "ent_coef": 0.0001339889565950314,
            "clip_range": 0.2399693801816262,
            "gamma": 0.969700160687714,
            "batch_size": 64,
        },
    }

    for reward_type in reward_types:
        print(f"\n=== Starting training with reward_type = '{reward_type}' ===")
        start_time = time.time()

        # Use the right window_size for env from best params
        params = best_params_map[reward_type]
        env = DummyVecEnv(
            [lambda: TradingEnv(df_train, window_size=params["window_size"], reward_type=reward_type)])

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            ent_coef=params["ent_coef"],
            clip_range=params["clip_range"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            n_steps=1024,
            verbose=0,
            seed=42,
        )

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
        test_env = TradingEnv(df_test, window_size=params["window_size"], reward_type=reward_type)
        obs = test_env.reset()
        done = False
        equity_curve = [test_env.balance]
        weights_over_time = []
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # Normalize weights to sum to 1
            if len(action.shape) == 1:
                weights = action / (np.sum(action) + 1e-8)
                weights_over_time.append(weights)
            else:
                weights = action[0] / (np.sum(action[0]) + 1e-8)
                weights_over_time.append(weights)

            obs, reward, done, _ = test_env.step(weights)
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

    # ... rest of your code unchanged

    # Plot all equity curves
    plt.figure(figsize=(14, 7))
    for reward_type, res in results.items():
        plt.plot(res['equity_curve'], label=f'{reward_type} (Sharpe: {res["sharpe"]:.2f})')

    # Benchmark
    adj_close = df_test.xs('Close', axis=1, level=1)
    returns_df = adj_close.pct_change().dropna()
    equally_weighted_returns = returns_df.mean(axis=1)
    benchmark_equity = (1 + equally_weighted_returns).cumprod() * 1000
    plt.plot(benchmark_equity.values, 'k--', label='Buy & Hold Equally Weighted')

    plt.title('Portfolio Equity Curve Comparison (Test Period Only)')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot smoothed weights
    for reward_type, res in results.items():
        window = 20
        weights = res['weights_over_time']
        smoothed_weights = pd.DataFrame(weights, columns=tickers).rolling(window=window, min_periods=1).mean().values

        plt.figure(figsize=(14, 7))
        plt.stackplot(
            range(smoothed_weights.shape[0]),
            smoothed_weights.T,
            labels=[f'{t}' for t in tickers],
            alpha=0.8
        )
        plt.title(f'Smoothed Portfolio Weights Over Time ({reward_type})')
        plt.xlabel('Time Step')
        plt.ylabel('Weight')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    # Calculate benchmark stats
    benchmark_returns = equally_weighted_returns.values
    benchmark_eq = benchmark_equity.values
    benchmark_cum_return = (benchmark_eq[-1] / benchmark_eq[0]) - 1
    benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)
    benchmark_sharpe = compute_sharpe(benchmark_returns)

    # Summary table
    print("\n=== Summary of Performance Across Reward Types and Benchmark ===")
    summary_data = {
        "Reward Type": [],
        "Sharpe Ratio": [],
        "Cumulative Return": [],
        "Volatility": [],
        "Training Time (s)": [],
        "T-stat (vs B&H)": [],
        "p-value": [],
    }

    for reward_type, res in results.items():
        eq = np.array(res['equity_curve'])
        returns = np.diff(eq) / eq[:-1]
        cum_return = eq[-1] / eq[0] - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = compute_sharpe(returns)

        # t-test against benchmark
        min_len = min(len(returns), len(benchmark_returns))
        t_stat, p_val = ttest_ind(returns[:min_len], benchmark_returns[:min_len], equal_var=False)

        summary_data["Reward Type"].append(reward_type)
        summary_data["Sharpe Ratio"].append(round(sharpe, 4))
        summary_data["Cumulative Return"].append(round(cum_return, 4))
        summary_data["Volatility"].append(round(volatility, 4))
        summary_data["Training Time (s)"].append(round(res['training_time'], 2))
        summary_data["T-stat (vs B&H)"].append(round(t_stat, 4))
        summary_data["p-value"].append(round(p_val, 4))

    # Add benchmark row
    summary_data["Reward Type"].append("Buy & Hold")
    summary_data["Sharpe Ratio"].append(round(benchmark_sharpe, 4))
    summary_data["Cumulative Return"].append(round(benchmark_cum_return, 4))
    summary_data["Volatility"].append(round(benchmark_volatility, 4))
    summary_data["Training Time (s)"].append("-")
    summary_data["T-stat (vs B&H)"].append("-")
    summary_data["p-value"].append("-")

    # Print table
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
