# src/market_env.py
import numpy as np

class MarketEnvironment:
    def __init__(self, S0, total_shares, T, N, sigma, gamma, eta):
        self.S0 = S0
        self.total_shares = total_shares
        self.T = T
        self.N = N
        self.dt = T/N
        self.sigma = sigma
        self.gamma = gamma
        self.eta = eta
        
    def simulate_price_path(self, trade_schedule, seed = None):
        # Initialize price path array
        price_path = np.zeros(self.N + 1)
        price_path[0] = self.S0
        total_revenue = 0

        if seed is not None:
            np.random.seed(seed)
        
        # Generate random shocks for the price process
        shocks = np.random.normal(0, np.sqrt(self.dt), self.N)

        current_price = self.S0
        for i in range(self.N):
            u_k = trade_schedule[i]

            drift = -self.gamma * u_k * self.dt
            diffusion = self.sigma * shocks[i]

            next_price = current_price + drift + diffusion

            exec_price = current_price - (self.eta * u_k / self.dt)

            total_revenue += exec_price * u_k
            
            price_path[i + 1] = next_price
            current_price = next_price
        
        initial_value = self.S0 * self.total_shares
        cost = initial_value - total_revenue

        return price_path, cost