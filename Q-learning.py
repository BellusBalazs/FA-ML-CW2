import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from collections import defaultdict

def safe_float(x):
    return x.item() if hasattr(x, 'item') else float(x)

def get_data(stock="AAPL", start="2015-01-01", end="2020-01-01"):
    print(f"Downloading data for {stock} from {start} to {end}...")
    data = yf.download(stock, start=start, end=end)
    data["Return"] = data["Close"].pct_change()
    data["SMA"] = data["Close"].rolling(window=5).mean()
    data["RSI"] = compute_rsi(data["Close"], window=14)
    data = data.dropna()
    print(f"Data downloaded: {len(data)} rows after cleaning.\n")
    return data

def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_state(row):
    return (
        int(safe_float(row["Return"]) * 100),
        int(safe_float(row["RSI"]) // 10),
    )

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = defaultdict(lambda: np.zeros(len(actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.actions))
        return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state):
        q_predict = self.Q[state][action]
        q_target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (q_target - q_predict)

def run_episode(agent, data, reward_fn, action_space):
    cash, stock, portfolio = 10000, 0, []
    trades = 0
    for i in range(len(data) - 1):
        row = data.iloc[i]
        next_row = data.iloc[i + 1]
        state = get_state(row)
        action = agent.choose_action(state)
        price = safe_float(row["Close"])

        # Actions: 0 = Buy, 1 = Hold, 2 = Sell
        if action_space == "simple":
            if action == 0 and cash >= price:
                stock += 1
                cash -= price
                trades += 1
            elif action == 2 and stock > 0:
                stock -= 1
                cash += price
                trades += 1

        next_price = safe_float(next_row["Close"])
        reward = reward_fn(price, next_price, stock)

        next_state = get_state(next_row)
        agent.learn(state, action, reward, next_state)

        portfolio_value = cash + stock * price
        portfolio.append(portfolio_value)

    return portfolio, trades

def reward_return(price, next_price, stock):
    return (next_price - price) * stock

def reward_risk_adj(price, next_price, stock):
    raw_return = (next_price - price) * stock
    risk_penalty = 0.01 * abs(raw_return)
    return raw_return - risk_penalty

def reward_utility(price, next_price, stock):
    gain = (next_price - price) * stock
    return np.log(1 + max(gain, -0.99))

def buy_and_hold(data):
    start_price = safe_float(data.iloc[0]["Close"])
    shares = 10000 // start_price
    final_value = shares * safe_float(data.iloc[-1]["Close"])
    print(f"Buy and Hold: start price = ${start_price:.2f}, shares = {shares}, final portfolio = ${final_value:.2f}\n")
    return [shares * safe_float(p) for p in data["Close"].values]

if __name__ == "__main__":
    data = get_data()
    actions = ["Buy", "Hold", "Sell"]
    num_episodes = 50

    results = {}
    for label, reward_fn in zip(["Return", "RiskAdj", "Utility"], [reward_return, reward_risk_adj, reward_utility]):
        print(f"Training agent with reward: {label}")
        agent = QLearningAgent(actions)
        final_portfolios = []
        total_trades = []

        for episode in range(num_episodes):
            portfolio, trades = run_episode(agent, data, reward_fn, action_space="simple")
            final_portfolios.append(portfolio[-1])
            total_trades.append(trades)
            if (episode + 1) % 10 == 0 or episode == 0:
                print(f" Episode {episode + 1}/{num_episodes} - Final portfolio: ${portfolio[-1]:.2f}, Trades: {trades}")

        print(f"Training complete for {label}. Best portfolio: ${max(final_portfolios):.2f}, Average trades: {np.mean(total_trades):.1f}\n")
        results[label] = portfolio

    results["BuyHold"] = buy_and_hold(data)

    plt.figure(figsize=(10, 6))
    for key in results:
        plt.plot(results[key], label=key)
    plt.legend()
    plt.title("Q-Learning Trading Agent vs Buy-and-Hold")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.grid()
    plt.show()
