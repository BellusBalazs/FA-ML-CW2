import yfinance as yf
import pandas as pd
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env_new_long import TradingEnv  # adjust if needed


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



def evaluate_model(model, env):
    obs = env.reset()
    done = False
    equity_curve = [env.balance]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        equity_curve.append(env.balance)

    portfolio_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = compute_sharpe(portfolio_returns)
    return sharpe

def run_hyperparameter_search():
    tickers = ['AAPL', 'JNJ', 'XOM', 'JPM', 'PG']
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    df = download_data(tickers, start_date, end_date)
    df = df.dropna(how='all').ffill().bfill()
    valid_mask = df.notna().all(axis=1)
    valid_dates = df.index[valid_mask]
    df = df.loc[valid_dates[0]:valid_dates[-1]]

    # Split dates for train/test split, e.g., 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    window_size = 10

    # Hyperparameter grids
    learning_rates = [1e-4, 3e-4, 1e-3]
    ent_coefs = [0.0, 0.01, 0.05]
    clip_ranges = [0.1, 0.2, 0.3]
    gammas = [0.95, 0.98, 0.99]

    total_timesteps = 5000  # smaller for speed
    batch_size = 64

    best_sharpe = -np.inf
    best_params = None

    for lr in learning_rates:
        for ent_coef in ent_coefs:
            for clip_range in clip_ranges:
                for gamma in gammas:
                    print(f"\nTesting params: lr={lr}, ent_coef={ent_coef}, clip_range={clip_range}, gamma={gamma}")

                    # Train environment on training data only
                    env = DummyVecEnv([lambda: TradingEnv(train_df, window_size=window_size, reward_type='basic')])

                    model = PPO(
                        "MlpPolicy",
                        env,
                        verbose=0,
                        batch_size=batch_size,
                        n_steps=1024,
                        learning_rate=lr,
                        ent_coef=ent_coef,
                        clip_range=clip_range,
                        gamma=gamma,
                        seed=42,
                    )

                    model.learn(total_timesteps=total_timesteps)

                    # Evaluate model on test data only
                    test_env = TradingEnv(test_df, window_size=window_size, reward_type='basic')
                    sharpe = evaluate_model(model, test_env)
                    print(f"Sharpe ratio: {sharpe:.4f}")

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {
                            'learning_rate': lr,
                            'ent_coef': ent_coef,
                            'clip_range': clip_range,
                            'gamma': gamma
                        }

    print("\n=== Best Hyperparameters ===")
    print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
    print(best_params)

if __name__ == "__main__":
    start_time = time.time()
    run_hyperparameter_search()
    print(f"\nTotal tuning time: {time.time() - start_time:.1f} seconds")
