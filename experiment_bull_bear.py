import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from scipy.stats import ttest_ind
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


class LoggingCallback(BaseCallback):
    def __init__(self, log_interval=5000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            print(f"Step: {self.n_calls}")
        return True


def run_single_experiment(tickers, train_start, train_end, test_start, test_end, label):
    print(f"\n=== Running Experiment: {label} ===")
    df_full = download_data(tickers, start=train_start, end=test_end)
    df_full = df_full.dropna(how='all').ffill().bfill()
    valid_mask = df_full.notna().all(axis=1)
    df_full = df_full.loc[valid_mask]

    df_train = df_full.loc[train_start:train_end]
    df_test = df_full.loc[test_start:test_end]

    reward_types = ['basic', 'utility', 'risk_penalty', 'drawdown_penalty']
    results = {}

    # Best hyperparameters per reward type
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

    total_timesteps = 30000

    for reward_type in reward_types:
        print(f"\n--- Training with reward type: {reward_type} ---")
        params = best_params_map[reward_type]
        env = DummyVecEnv([lambda: TradingEnv(df_train, window_size=params["window_size"], reward_type=reward_type)])

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            ent_coef=params["ent_coef"],
            clip_range=params["clip_range"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            n_steps=2048,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            use_sde=False,
            vf_coef=0.4,
            n_epochs=10,
            verbose=0,
            seed=42,
        )

        model.learn(total_timesteps=total_timesteps, callback=LoggingCallback(log_interval=5000))

        test_env = TradingEnv(df_test, window_size=params["window_size"], reward_type=reward_type)
        obs = test_env.reset()
        done = False
        equity_curve = [test_env.balance]
        weights_over_time = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = action / (np.sum(action) + 1e-8)
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

    # Benchmark: Equally weighted Buy & Hold
    adj_close = df_test.xs('Close', axis=1, level=1)
    returns_df = adj_close.pct_change().dropna()
    equally_weighted_returns = returns_df.mean(axis=1)
    benchmark_equity = (1 + equally_weighted_returns).cumprod() * 1000

    benchmark_returns = equally_weighted_returns.values
    benchmark_eq = benchmark_equity.values
    benchmark_cum_return = (benchmark_eq[-1] / benchmark_eq[0]) - 1
    benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)
    benchmark_sharpe = compute_sharpe(benchmark_returns)

    # Plot equity curves
    plt.figure(figsize=(12, 6))
    min_len = min([len(res['equity_curve']) for res in results.values()])
    for rt, res in results.items():
        plt.plot(res['equity_curve'][:min_len], label=f'{rt} (Sharpe: {res["sharpe"]:.2f})')
    plt.plot(benchmark_eq[:min_len], 'k--', label='Buy & Hold Equally Weighted')
    plt.title(f'Equity Curve Comparison: {label}')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot smoothed weights
    window = 20
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

    # Print summary table
    print("\n=== Summary of Performance Across Reward Types and Benchmark ===")
    summary_data = {
        "Reward Type": [],
        "Sharpe Ratio": [],
        "Cumulative Return": [],
        "Volatility": [],
        "T-stat (vs B&H)": [],
        "p-value": [],
    }

    for reward_type, res in results.items():
        eq = np.array(res['equity_curve'])
        returns = np.diff(eq) / eq[:-1]
        cum_return = eq[-1] / eq[0] - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = compute_sharpe(returns)

        min_len = min(len(returns), len(benchmark_returns))
        t_stat, p_val = ttest_ind(returns[:min_len], benchmark_returns[:min_len], equal_var=False)

        summary_data["Reward Type"].append(reward_type)
        summary_data["Sharpe Ratio"].append(round(sharpe, 4))
        summary_data["Cumulative Return"].append(round(cum_return, 4))
        summary_data["Volatility"].append(round(volatility, 4))
        summary_data["T-stat (vs B&H)"].append(round(t_stat, 4))
        summary_data["p-value"].append(round(p_val, 4))

    summary_data["Reward Type"].append("Buy & Hold")
    summary_data["Sharpe Ratio"].append(round(benchmark_sharpe, 4))
    summary_data["Cumulative Return"].append(round(benchmark_cum_return, 4))
    summary_data["Volatility"].append(round(benchmark_volatility, 4))
    summary_data["T-stat (vs B&H)"].append("-")
    summary_data["p-value"].append("-")

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))


def run_all_experiments():
    tickers = ['AAPL', 'JNJ', 'XOM', 'JPM', 'PG', 'HD', 'BA', 'NEM', 'NEE', 'AMT']

    # COVID Crash Period
    run_single_experiment(
        tickers=tickers,
        train_start='2018-01-01',
        train_end='2020-02-14',
        test_start='2020-02-15',
        test_end='2020-04-15',
        label='COVID Crash Period'
    )

    # Bull Market 2024
    run_single_experiment(
        tickers=tickers,
        train_start='2022-01-01',
        train_end='2024-05-31',
        test_start='2024-06-01',
        test_end='2024-08-01',
        label='Bull Market 2024'
    )


if __name__ == "__main__":
    run_all_experiments()
