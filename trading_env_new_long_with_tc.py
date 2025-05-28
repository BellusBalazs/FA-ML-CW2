import numpy as np
import gym
from gym import spaces


class TradingEnv(gym.Env):
    def __init__(self, df, window_size=10, initial_balance=1000, reward_type='basic', transaction_cost=0.0):
        super().__init__()

        # Data and environment parameters
        self.df = df  # Expect MultiIndex columns: (asset, feature)
        self.n_assets = len(df.columns.levels[0])
        self.features_per_asset = len(df.columns.levels[1])
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reward_type = reward_type
        self.transaction_cost = transaction_cost

        # Action space: portfolio weights for each asset [0, 1]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

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
        self.total_cost = 0

        return self._get_obs()

    def _get_obs(self):
        # Get the current observation window
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        return window.values.astype(np.float32)

    def step(self, action):
        # Normalize actions to sum to 1 for portfolio allocation
        action = np.clip(action, 0, 1)
        new_weights = action / (np.sum(action) + 1e-8)

        # Calculate returns for assets (percentage change)
        adj_close = self.df.xs('Close', axis=1, level=1)
        prev_prices = adj_close.iloc[self.current_step - 1].values
        current_prices = adj_close.iloc[self.current_step].values
        asset_returns = (current_prices - prev_prices) / prev_prices

        # Calculate portfolio return using OLD weights (previous step's weights)
        portfolio_return = np.dot(self.weights, asset_returns)

        # Calculate transaction cost based on rebalancing (change in weights)
        weight_change = np.abs(new_weights - self.weights)
        transaction_fee = self.transaction_cost * np.sum(weight_change)
        self.total_cost += transaction_fee

        # Calculate reward BEFORE applying transaction cost
        # Reward shaping uses the portfolio return (before costs)
        if self.reward_type == 'basic':
            reward = portfolio_return
        elif self.reward_type == 'utility':
            reward = np.log(1 + portfolio_return)
        elif self.reward_type == 'risk_penalty':
            reward = portfolio_return - 0.5 * np.std(asset_returns)
        elif self.reward_type == 'drawdown_penalty':
            max_balance = max(self.equity_curve)
            drawdown = (max_balance - self.balance) / max_balance if max_balance > 0 else 0
            reward = portfolio_return - drawdown
        else:
            reward = portfolio_return  # fallback

        # Update balance considering portfolio return AND transaction costs multiplicatively
        self.balance *= (1 + portfolio_return - transaction_fee)

        # Calculate net return after fees for logging or additional use
        net_return = (self.balance - self.equity_curve[-1]) / self.equity_curve[-1]

        # Update weights, step, equity curve, and done flag
        self.weights = new_weights
        self.current_step += 1
        self.equity_curve.append(self.balance)
        self.done = self.current_step >= len(self.df) - 1

        obs = self._get_obs()
        info = {'transaction_fee': transaction_fee, 'net_return': net_return}
        return obs, reward, self.done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Weights: {self.weights}")
