# src/baseline_ac.py
import numpy as np

class TWAPModel:
    def __init__(self, V, T):
        self.V = V
        self.T = T 
        
    def generate_trajectory(self):
        return np.full(self.T, self.V / self.T)


class AlmgrenChrissModel:
    def __init__(self, V, T, tau, lam, sigma, eta):
        self.V = V
        self.T = T
        self.tau = tau
        self.lam = lam
        self.sigma = sigma
        self.eta = eta
    
    def generate_trajectory(self):
        inside_term = 1.0 + (self.lam * (self.sigma ** 2) * self.tau) / (2.0 * self.eta)
        kappa = np.arccosh(inside_term) / self.tau

        j = np.arange(self.T + 1)
        time_left = (self.T - j) * self.tau
        total_time = self.T * self.tau
        
        numerator = np.sinh(kappa * time_left)
        denominator = np.sinh(kappa * total_time)
        
        x_trajectory = self.V * (numerator / denominator)

        trade_schedule = -np.diff(x_trajectory)

        return trade_schedule

def main():
    baseline = TWAPModel(V=10000, T=8)
    baseline_schedule = baseline.generate_trajectory()
    print(f"Trajectory (shares per interval): {np.round(baseline_schedule, 2)}")
    print(f"Total Shares Scheduled: {np.sum(baseline_schedule):.2f}\n")

    stock_dict = {
        "AAPL": [2157, 0.039263],
        "AMZN": [1704, 0.021858],
        "GOOG": [1247, 0.044573],
        "INTC": [254464, 0.002083],
        "MSFT": [263664, 0.002483],
    }
    global_risk_target = 0.00001

    for stock, params in stock_dict.items():
        vol, sigma = params
        dynamic_lam = global_risk_target / (sigma ** 2)

        ac = AlmgrenChrissModel(
            V=vol,
            T=8,  
            tau=60,
            lam=dynamic_lam,
            sigma=sigma,
            eta=0.1, 
        )
        
        baseline_trades = ac.generate_trajectory()

        print(f"--- {stock} Execution Schedule ---")
        print(f"Trajectory (shares per interval): {np.round(baseline_trades, 2)}")
        print(f"Total Shares Scheduled: {np.sum(baseline_trades):.2f}")
        print(f"Volatility used: {sigma}\n")


if __name__ == "__main__":
    main()