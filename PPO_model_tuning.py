import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import optuna

from optuna.pruners import MedianPruner

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env_new_long import TradingEnv
import logging
optuna.logging.set_verbosity(logging.WARNING)


# === CONFIGURATION === #
TICKERS = ['AAPL', 'JNJ', 'XOM', 'JPM', 'PG', 'HD', 'BA', 'NEM', 'NEE', 'AMT']
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'
SPLIT_DATE = '2021-01-01'

TOTAL_TIMESTEPS = 30000
N_TRIALS = 60
REWARD_TYPES = ['basic', 'utility', 'risk_penalty', 'drawdown_penalty']


def download_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    panel = {ticker: raw[ticker][['High', 'Low', 'Close', 'Volume']] for ticker in tickers}
    df = pd.concat(panel.values(), axis=1, keys=panel.keys())
    return df.ffill().bfill()


def compute_sharpe(returns):
    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / (std_return + 1e-8) * np.sqrt(252)


def evaluate_model(model, env, max_steps=1000):
    obs = env.reset()
    done = False
    equity_curve = [env.balance]
    step_count = 0

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        equity_curve.append(env.balance)
        step_count += 1

    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = compute_sharpe(returns)
    return sharpe, equity_curve, model.policy.state_dict()


def plot_equity(equity_curve, label):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label=label)
    plt.title(f"Equity Curve - {label}")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def tune_for_reward_type(reward_type, train_df, test_df):
    print(f"\n Starting tuning for reward type: {reward_type}")

    def objective(trial):
        window_size = trial.suggest_categorical("window_size", [10, 30, 50, 100])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        ent_coef = trial.suggest_loguniform("ent_coef", 1e-5, 0.05)
        clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
        gamma = trial.suggest_uniform("gamma", 0.95, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 96, 128])

        env = DummyVecEnv([lambda: TradingEnv(train_df, window_size=window_size, reward_type=reward_type)])
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            clip_range=clip_range,
            gamma=gamma,
            batch_size=batch_size,
            n_steps=1024,
            verbose=0,
            seed=42
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        test_env = TradingEnv(test_df, window_size=window_size, reward_type=reward_type)
        sharpe, _, _ = evaluate_model(model, test_env)

        trial.report(sharpe, step=0)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return sharpe

    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"Best Sharpe for {reward_type}: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Train best model fully for final evaluation and weights
    best_params = study.best_params
    env = DummyVecEnv([lambda: TradingEnv(train_df, window_size=best_params['window_size'], reward_type=reward_type)])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=best_params['learning_rate'],
        ent_coef=best_params['ent_coef'],
        clip_range=best_params['clip_range'],
        gamma=best_params['gamma'],
        batch_size=best_params['batch_size'],
        n_steps=1024,
        verbose=0,
        seed=42
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    final_env = TradingEnv(test_df, window_size=best_params['window_size'], reward_type=reward_type)
    sharpe, equity_curve, final_weights = evaluate_model(model, final_env)

    print(f"Final Sharpe (re-evaluated) for {reward_type}: {sharpe:.4f}")
    print(f"Final policy weights snapshot (sample):")
    # Just print first layer weights or a small snippet for brevity
    for name, param in final_weights.items():
        print(f"  {name}: {param.flatten()[:5]} ...")
        break

    # Optionally plot equity curve for each reward type
    plot_equity(equity_curve, f"Reward: {reward_type}")

    return {
        "reward_type": reward_type,
        "sharpe": sharpe,
        "best_params": best_params,
        "final_weights": final_weights,
        "equity_curve": equity_curve,
    }


if __name__ == "__main__":
    start_time = time.time()

    print("Downloading data...")
    df = download_data(TICKERS, START_DATE, END_DATE)
    df = df.dropna(how='any')

    train_df = df.loc[:SPLIT_DATE]
    test_df = df.loc[SPLIT_DATE:]

    results = {}

    for rt in REWARD_TYPES:
        results[rt] = tune_for_reward_type(rt, train_df, test_df)

    print(f"\n Total tuning time: {time.time() - start_time:.2f} seconds")
