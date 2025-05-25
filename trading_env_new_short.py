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

        # Action space: portfolio weights for each asset, from -1 (fully short) to +1 (fully long)
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
        # Start equally weighted long portfolio by default
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.equity_curve = [self.balance]
        return self._get_obs()

    def _get_obs(self):
        # Return last `window_size` rows for all features flattened for all assets
        window = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return window.astype(np.float32)

    def step(self, action):
        # Clip action to allowed range [-1, 1]
        action = np.clip(action, -1, 1)

        # Normalize absolute sum of weights to be <= 1 (max total exposure)
        abs_sum = np.sum(np.abs(action)) + 1e-8
        if abs_sum > 1:
            weights = action / abs_sum
        else:
            weights = action  # if sum(abs(weights)) < 1, cash is held for the rest

        self.weights = weights

        # Cash weight is residual
        cash_weight = 1 - np.sum(np.abs(weights))

        # Get asset returns for this step
        adj_close = self.df.xs('Close', axis=1, level=1)
        asset_returns = adj_close.pct_change().iloc[self.current_step].values

        # Portfolio return = weighted asset returns + cash return (0)
        portfolio_return = np.dot(weights, asset_returns) + cash_weight * 0

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
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Weights: {self.weights}, Cash weight: {1 - np.sum(np.abs(self.weights)):.4f}")
