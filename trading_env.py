import numpy as np
import pandas as pd
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=10, initial_balance=1000):
        super(TradingEnv, self).__init__()
        self.df = df
        self.n_assets = df.shape[1]
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.n_assets),
            dtype=np.float32
        )

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.done = False
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._get_obs()

    def _get_obs(self):
        return self.df.iloc[self.current_step - self.window_size:self.current_step].values

    def step(self, action):
        action = np.clip(action, 0, 1)
        self.weights = action / (np.sum(action) + 1e-8)
        returns = self.df.pct_change().iloc[self.current_step].values
        portfolio_return = np.dot(self.weights, returns)
        self.balance *= (1 + portfolio_return)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), portfolio_return, done, {}

    def render(self):
        print(f"Balance: {self.balance:.2f}, Weights: {self.weights}")
