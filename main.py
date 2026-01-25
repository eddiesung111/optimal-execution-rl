# main.py
import numpy as np
import matplotlib.pyplot as plt

from src.market_env import MarketEnvironment
from src.strategy import Strategy
from src.plotting import plot_trading_schedules, plot_efficient_frontier

def run_simulation_batch(env, schedule, num_sims=500):
    costs = []
    for _ in range(num_sims):
        _, cost = env.simulate_price_path(schedule)
        costs.append(cost)
    return np.mean(costs), np.std(costs)


def main():
    S0 = 100        # Initial stock price
    X = 1_000_000   # Total shares to sell
    initial_cash = S0 * X
    T = 1.0         # Total time in days
    N = 50          # Number of time steps
    
    sigma = 0.3     # Volatility in days
    gamma = 2.5e-7  # Permanent impact coefficient
    eta = 2.5e-6    # Temporary impact coefficient
    
    # Setup
    env = MarketEnvironment(S0, X, T, N, sigma, gamma, eta)
    strat = Strategy(X, T, N)


    print("Generating Trading Schedules...")
    twap_sched = strat.get_twap_schedule()
    ac_sched = strat.get_almgren_chriss_schedule(sigma, eta, lmbda=1e-6)
    
    plot_trading_schedules(twap_sched, ac_sched)


    print("\nSimulating Efficient Frontier...")

    scale = 10000 / initial_cash  # Scale to basis points

    lambdas = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    ac_costs = []
    ac_risks = []
    
    persona_map = {
        1e-7:  {"role": "The Algorithm",  "desc": "Risk Neutral (TWAP-like)"},
        5e-5:  {"role": "The Trader",     "desc": "Risk Averse (Balanced)"},
        1e-4:  {"role": "The Panic Seller","desc": "High Urgency (Fast Sell)"}
    }

    print("\n" + "="*85)
    print(f"{'ROLE':<18} | {'DESCRIPTION':<28} | {'LAMBDA':<8} | {'COST':<8} | {'RISK':<8}")
    print("-" * 85)
    for lmbda in lambdas:
        schedule = strat.get_almgren_chriss_schedule(sigma, eta, lmbda)
        avg, std = run_simulation_batch(env, schedule, num_sims=5000)
        
        ac_costs.append(avg)
        ac_risks.append(std)
        
        if lmbda in persona_map:
            info = persona_map[lmbda]
            avg_bps = avg * scale
            std_bps = std * scale
            print(f"{info['role']:<18} | {info['desc']:<28} | {lmbda:.0e}    | {avg_bps:>4.0f} bps | {std_bps:>4.0f} bps")
    print("="*85 + "\n")


    twap_avg, twap_std = run_simulation_batch(env, twap_sched, num_sims=5000)
    plot_efficient_frontier(ac_risks, ac_costs, twap_std, twap_avg)

if __name__ == "__main__":
    main()