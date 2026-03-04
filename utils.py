# utils.py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(agent_name):
    path = f"results/test_history_{agent_name}.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def calculate_metrics(rl_bps, baseline_bps):
    rl = np.array(rl_bps)
    base = np.array(baseline_bps)
    edge = rl - base
    
    prob_win = np.mean(edge > 0) * 100
    gains = edge[edge > 0]
    losses = edge[edge < 0]
    
    avg_gain = np.mean(gains)
    avg_loss = abs(np.mean(losses))
    
    if avg_loss > 1e-8:
        glr = avg_gain / avg_loss
    else:
        glr = float('inf') if avg_gain > 0 else 0.0
        
    return {
        "Avg_RL": np.mean(rl),
        "Mean_Edge": np.mean(edge),
        "Std_Edge": np.std(edge),
        "Prob_Win": prob_win,
        "GLR": glr
    }

def generate_csv_and_plot(baseline="ac"):
    tab_data = load_data("tabular")
    ddqn_data = load_data("ddqn")
    
    if not tab_data or not ddqn_data:
        print("Missing tabular or ddqn data. Please run tests for both models first.")
        return

    stocks = list(tab_data.keys())
    report_rows = []

    for stock in stocks:
        avg_twap = np.mean(tab_data[stock]['twap'])
        avg_ac = np.mean(tab_data[stock]['ac'])
        
        tab_m = calculate_metrics(tab_data[stock]['rl'], tab_data[stock][baseline])
        ddq_m = calculate_metrics(ddqn_data[stock]['rl'], ddqn_data[stock][baseline])
        
        denom = abs(avg_ac) 
        tab_improv = ((tab_m['Avg_RL'] - avg_ac) / denom) * 100
        ddq_improv = ((ddq_m['Avg_RL'] - avg_ac) / denom) * 100

        win_glr = "Tabular" if tab_m['GLR'] > ddq_m['GLR'] else "DDQN"
        win_prob = "Tabular" if tab_m['Prob_Win'] > ddq_m['Prob_Win'] else "DDQN"
        win_mean_rl = "Tabular" if tab_m['Avg_RL'] > ddq_m['Avg_RL'] else "DDQN"

        # --- ROW 1: TABULAR ---
        report_rows.append({
            "Ticker": stock,
            "Model": "Tabular",
            "GLR": f"{tab_m['GLR']:.2f}",
            "P[ΔP&L > 0]": f"{tab_m['Prob_Win']:.1f}%",
            "Std.": f"{tab_m['Std_Edge']:.2f}",
            "Mean RL": f"{tab_m['Avg_RL']:.2f}",
            "Improv. vs AC": f"{tab_improv:+.2f}%",
            "Mean AC": f"{avg_ac:.2f}",
            "Mean TWAP": f"{avg_twap:.2f}"
        })

        # --- ROW 2: DDQN ---
        report_rows.append({
            "Ticker": "",  
            "Model": "DDQN",
            "GLR": f"{ddq_m['GLR']:.2f}",
            "P[ΔP&L > 0]": f"{ddq_m['Prob_Win']:.1f}%",
            "Std.": f"{ddq_m['Std_Edge']:.2f}",
            "Mean RL": f"{ddq_m['Avg_RL']:.2f}",
            "Improv. vs AC": f"{ddq_improv:+.2f}%",
            "Mean AC": "-", 
            "Mean TWAP": "-"
        })
        
        # --- ROW 3: WINNER ---
        report_rows.append({
            "Ticker": "",  
            "Model": "Winner",
            "GLR": win_glr,
            "P[ΔP&L > 0]": win_prob,
            "Std.": "-",
            "Mean RL": "-",
            "Improv. vs AC": win_mean_rl,
            "Mean AC": "-",
            "Mean TWAP": "-"
        })

        report_rows.append({k: "" for k in report_rows[0].keys()})

    report_rows = report_rows[:-1]
    df_report = pd.DataFrame(report_rows)
    csv_path = f"results/performance_report.csv"
    df_report.to_csv(csv_path, index=False)
    
    print("\n" + "="*105)
    print(f"{'RL EXECUTION MODEL COMPARISON REPORT':^105}")
    print("="*105)
    print(df_report.to_string(index=False, justify='center'))
    print("="*105)
    print(f"Grouped CSV saved to: {csv_path}\n")


    fig, axes = plt.subplots(nrows=len(stocks), ncols=1, figsize=(10, 4 * len(stocks)))
    fig.tight_layout(pad=6.0)
    if len(stocks) == 1: axes = [axes]
        
    for i, stock in enumerate(stocks):
        ax = axes[i]
        
        tab_edge = np.array(tab_data[stock]['rl']) - np.array(tab_data[stock][baseline])
        ddqn_edge = np.array(ddqn_data[stock]['rl']) - np.array(ddqn_data[stock][baseline])
        
        sns.histplot(tab_edge, kde=True, color='blue', alpha=0.3, label='Tabular Edge', ax=ax, stat='density', bins=40)
        sns.histplot(ddqn_edge, kde=True, color='orange', alpha=0.3, label='DDQN Edge', ax=ax, stat='density', bins=40)
        
        ax.axvline(np.mean(tab_edge), color='blue', linestyle='--', linewidth=2)
        ax.axvline(np.mean(ddqn_edge), color='orange', linestyle='--', linewidth=2)
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5) 
        
        metrics_text = (
            f"TABULAR:\n"
            f"μ Edge:  {np.mean(tab_edge):+.2f} bps\n"
            f"σ:       {np.std(tab_edge):.2f}\n"
            f"-------------------\n"
            f"DDQN:\n"
            f"μ Edge:  {np.mean(ddqn_edge):+.2f} bps\n"
            f"σ:       {np.std(ddqn_edge):.2f}"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')
        
        ax.set_title(f"{stock}: Performance Edge vs {baseline.upper()}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Edge (BPS) - Positive is Profitable", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc='upper left')

    img_path = f"results/distribution_plots.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {img_path}")
    plt.show()

if __name__ == "__main__":
    generate_csv_and_plot(baseline="ac")