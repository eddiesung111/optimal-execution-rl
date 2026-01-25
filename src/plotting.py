# src/plotting.py
import numpy as np
import matplotlib.pyplot as plt

def plot_trading_schedules(twap_schedule, ac_schedule, filename="results/trading_schedule.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(twap_schedule, label='TWAP (Constant)', linestyle='--', color='blue', linewidth=2)
    plt.plot(ac_schedule, label='Almgren-Chriss (Optimal)', color='orange', linewidth=2)
    
    plt.title('Trading Schedule Comparison: Urgency vs. Consistency')
    plt.xlabel('Time Step (Intervals)')
    plt.ylabel('Shares Sold per Interval')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()

def plot_efficient_frontier(ac_risks, ac_costs, twap_risk, twap_cost, filename="results/efficient_frontier.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(ac_risks, ac_costs, 'o-', label='Almgren-Chriss Frontier', color='orange', markersize=6)
    plt.scatter([twap_risk], [twap_cost], color='blue', s=150, label='TWAP Benchmark', zorder=5, edgecolors='black')
    plt.text(twap_risk, twap_cost, '  TWAP', verticalalignment='bottom')

    plt.title('Efficient Frontier of Optimal Execution')
    plt.xlabel('Risk (Standard Deviation of Cost)')
    plt.ylabel('Expected Cost (Implementation Shortfall)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()