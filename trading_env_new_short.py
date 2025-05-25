import numpy as np
import gym
from gym import spaces


class TradingEnv(gym.Env):
    def __init__(self, df, window_size=10, initial_balance=1000, reward_type='basic'):
        super().__init__()

        # Data and environment parameters
        self.df = df  # Expect MultiIndex columns: (asset, feature)
        self.n_assets = len(df.columns.levels[0])
        self.features_per_asset = len(df.columns.levels[1])
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reward_type = reward_type

        # Action space: portfolio weights for each asset [0, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Observation space: recent price/volume window
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.n_assets * self.features_per_asset),
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.done = False
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.equity_curve = [self.balance]
        return self._get_obs()

    def _get_obs(self):
        # Return last `window_size` rows for all features flattened for all assets
        window = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return window.astype(np.float32)

    def step(self, action):
        # Normalize actions to sum to 1 for portfolio allocation
        action = np.clip(action, 0, 1)
        self.weights = action / (np.sum(action) + 1e-8)

        # Calculate portfolio return based on Close prices
        adj_close = self.df.xs('Close', axis=1, level=1)
        asset_returns = adj_close.pct_change().iloc[self.current_step].values
        portfolio_return = np.dot(self.weights, asset_returns)

        # Update balance
        prev_balance = self.balance
        self.balance *= (1 + portfolio_return)

        # Reward shaping
        if self.reward_type == 'basic':
            reward = portfolio_return
        elif self.reward_type == 'utility':
            wealth_return = (self.balance - prev_balance) / max(prev_balance, 1e-8)
            reward = np.log(1 + wealth_return)
        elif self.reward_type == 'risk_penalty':
            reward = portfolio_return - 0.5 * np.std(asset_returns)
        elif self.reward_type == 'drawdown_penalty':
            max_balance = max(self.equity_curve)
            drawdown = (max_balance - self.balance) / max_balance if max_balance > 0 else 0
            reward = portfolio_return - drawdown
        else:
            reward = portfolio_return  # fallback

        # Update step and equity curve
        self.current_step += 1
        self.equity_curve.append(self.balance)
        self.done = self.current_step >= len(self.df) - 1

        obs = self._get_obs()
        info = {}
        return obs, reward, self.done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Weights: {self.weights}")
