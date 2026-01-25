# src/strategy.py
import numpy as np

class Strategy:
    def __init__(self, total_shares, T, N):
        self.X = total_shares
        self.T = T
        self.N = N
        self.dt = T / N

    def get_twap_schedule(self):
        # Strategy 1: Sell equal amonts every step
        trade_schedule = np.ones(self.N) * (self.X / self.N)
        return trade_schedule
    
    
    def get_almgren_chriss_schedule(self, sigma, eta, lmbda):
        # Strategy 2: Almgren-Chriss optimal execution schedule
        if lmbda == 0:
            return self.get_twap_schedule()
        
        kappa = np.sqrt(lmbda * sigma ** 2 / eta)
        t = np.linspace(0, self.T, self.N + 1)

        numerator = np.sinh(kappa * (self.T - t))
        denominator = np.sinh(kappa * self.T)

        x_trajectory = self.X * (numerator / denominator)
        trade_schedule = -np.diff(x_trajectory)

        return trade_schedule
