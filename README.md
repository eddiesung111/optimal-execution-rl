# Optimal Trade Execution via Reinforcement Learning: Tabular vs. DDQN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

## 📌 Project Summary
This project builds a custom Reinforcement Learning (RL) environment to solve the Optimal Trade Execution problem. The objective is to autonomously liquidate a large position of shares within a compressed time horizon while minimizing market impact and execution costs (slippage). 

The system leverages real-world historical Limit Order Book (LOB) data across major equities to train and rigorously evaluate two distinct RL agents:
1. **A Tabular Q-Learning Agent** utilizing discrete state-space bucketing.
2. **A Double Deep Q-Network (DDQN)** utilizing continuous, normalized state spaces.

The performance of both models is benchmarked head-to-head against industry-standard execution algorithms: the **Almgren-Chriss (AC)** model and **Time-Weighted Average Price (TWAP)**.

### 🚀 Key Implementations & Technical Highlights
* **LOB Data Pipeline & Microstructure Engineering:** Fetched and processed raw order book data, engineering predictive high-frequency features including **Order Book Imbalance** and **Log-Return Autocorrelation (Momentum)** to give the agents directional awareness before crossing the spread.
* **Rigorous State Normalization:** Engineered a robust state-processing pipeline for the DDQN, mathematically restricting all continuous variables (Time, Inventory, Spread, Imbalance, Momentum) to a strict `[-1.0, 1.0]` range to stabilize deep learning gradients.
* **Custom Reward Shaping & Action Spaces:** Redesigned the reward function logic and action spaces from existing literature to better penalize inventory risk and capture relative execution edge across fundamentally different stocks.
* **Synchronized Evaluation Engine:** Built a deterministic testing framework that forces both RL agents and the baselines (AC and TWAP) to trade the *exact same* historical market slices, ensuring mathematically fair performance comparisons (Gain-Loss Ratio, Win Probability, Mean Edge).

---

## 📊 Key Results & Performance

The models were evaluated strictly on their relative **Execution Edge (in basis points)** against the mathematically optimal AC trajectory and TWAP. A positive improvement percentage indicates the agent successfully out-traded the benchmark, saving execution costs.

*Note: The environment forces a highly compressed 8-step execution timeline to stress-test the agents in a noisy, high-frequency microstructure setting.*

### Model Comparison Matrix

| Ticker | Model   | GLR     | P[ΔP&L > 0] | Std. | Mean RL | Improv. vs AC | Mean AC | Mean TWAP |
|--------|---------|---------|-------------|------|---------|---------------|---------|-----------|
| AAPL   | Tabular | 1.46    | 57.4%       | 0.66 | -0.80   | 16.60%        | -0.95   | -0.90     |
|        | DDQN    | 1.49    | 53.6%       | 0.86 | -0.79   | 17.21%        | -       | -         |
|        | WINNER  | DDQN    | Tabular     | -    | -       | DDQN          | -       | -         |
|        |         |         |             |      |         |               |         |           |
| AMZN   | Tabular | 1.40    | 60.0%       | 0.97 | -1.81   | 11.97%        | -2.06   | -1.99     |
|        | DDQN    | 1.33    | 59.4%       | 1.18 | -1.42   | 30.89%        | -       | -         |
|        | WINNER  | Tabular | Tabular     | -    | -       | DDQN          | -       | -         |
|        |         |         |             |      |         |               |         |           |
| GOOG   | Tabular | 1.36    | 57.8%       | 0.72 | -1.02   | 13.76%        | -1.18   | -1.11     |
|        | DDQN    | 1.25    | 57.8%       | 0.96 | -0.43   | 63.69%        | -       | -         |
|        | WINNER  | Tabular | DDQN        | -    | -       | DDQN          | -       | -         |
|        |         |         |             |      |         |               |         |           |
| INTC   | Tabular | 1.00    | 49.2%       | 1.38 | -2.02   | -0.72%        | -2.00   | -1.91     |
|        | DDQN    | 1.27    | 44.8%       | 1.04 | -2.63   | -31.45%       | -       | -         |
|        | WINNER  | DDQN    | Tabular     | -    | -       | Tabular       | -       | -         |
|        |         |         |             |      |         |               |         |           |
| MSFT   | Tabular | 1.01    | 56.0%       | 1.49 | -2.13   | 5.78%         | -2.26   | -2.17     |
|        | DDQN    | 1.07    | 38.4%       | 1.23 | -2.45   | -8.56%        | -       | -         |
|        | WINNER  | DDQN    | Tabular     | -    | -       | Tabular       | -       | -         |



### Edge Distribution Analysis
![Execution Edge Distributions](results/distribution_plot.png)
*(The graph above illustrates the probability density of the RL agents' outperformance versus the baseline. A right-skewed distribution past the `0.0` break-even line indicates consistent profitability over the benchmark.)*

## 📂 Repository Structure

The codebase is modularized into environment definitions, agent architectures, and execution scripts to ensure easy replication and extension.

```text
├── data/                   # Raw and processed LOB data (CSV/Parquet)
├── src/             
│   ├── agent_ddqn.py     
│   ├── agent_tabular.py    
│   ├── baseline_ac.py      
│   ├── data_loader.py   
│   └── environment.py    
├── models/                # Saved model weights (.pth and .npy files)
├── results/                # Output directory for CSV reports and distribution plots
├── main.py                 # Main execution script for training and testing loops
├── utils.py        # Automated quantitative metrics and plotting engine
├── requirements.txt        # Python package dependencies
└── README.md
```

## 📈 Dataset & Feature Engineering

The models are trained and evaluated on highly granular historical Limit Order Book (LOB) data for five major mega-cap equities: AAPL, AMZN, GOOG, INTC, and MSFT.

To give the RL agents directional awareness before crossing the spread, the raw LOB data was transformed into a set of reactive, high-frequency predictive features:

* **Spread & Volume Percentiles:** Normalized historical trailing percentiles for the bid-ask spread and available queue volumes to detect liquidity droughts.
* **Order Book Imbalance:** Measures the buy/sell pressure at the top of the book, allowing the agent to front-run immediate micro-trend shifts.
* **Log-Return Autocorrelation (Momentum):** Captures short-term price momentum.
    * *The "Double-Lag" Correction:* To prevent the network from trading on stale data, this project computes a fast, reactive 5-minute rolling correlation against a longer 15-minute background mean, ensuring the agent reacts to immediate order flow rather than 30-minute-old signals.

## 🕹️ The RL Execution Environment
The custom environment (ExecutionEnvironment) simulates the mechanics of liquidating a large block of shares. To stress-test the agents in a noisy microstructure setting, the execution horizon is heavily compressed into 8 execution steps over an 8-minute trading window.

### State Space (Zero-Centered & Normalized)
For the DDQN, passing raw financial floats directly into a neural network leads to catastrophic gradient instability. Therefore, the 6-dimensional state space is strictly bounded to a $[-1,1]$. There are six features in total, 'Time Elapsed', 'Inventory Remaining', 'Spread', 'Ask Volume', 'Order Book Imbalance', `Autocorrelation (Z-Score)`

### Action Space
The agent does not output raw share amounts. Instead, the action space is a set of 11 discrete multipliers applied to the standard Time-Weighted Average Price (TWAP) trajectory:
* `Action Space`: $\beta \in [0.5x, 0.6x, ...., 1.4x, 1.5x]$
* An action of `1.0x` exactly matches the TWAP execution for that time step.

### Reward Shaping: Slippage & Inventory Risk
Instead of evaluating the agent solely on absolute implementation shortfall, the reward function is engineered to balance immediate execution costs against ongoing market exposure. The step reward is calculated as the negative sum of two distinct penalties:

1. **Slippage Penalty:** The immediate cost of crossing the spread and absorbing liquidity when executing a slice of the order.
2. **Inventory Penalty:** The market risk of holding unexecuted shares. This is calculated dynamically at each step based on the remaining inventory multiplied by the continuous drift in the asset's mid-price.

**The Strategic Rationale:** Incorporating the *Inventory Penalty* is the exact reason **Autocorrelation (Momentum)** and **Order Book Imbalance** were engineered into the state space. To minimize this penalty, the AI must learn to predict short-term price trends. 
* If the agent's features detect *adverse* momentum (the price is moving away), it learns to accelerate execution (e.g., choosing a `1.5x` multiplier). 
* If it detects *favorable* momentum, it learns to slow down execution (e.g., `0.5x`) to ride the trend and capture better prices later in the episode. 

Finally, the raw step cost is divided by the total ideal arrival cost to normalize the execution reward into basis points (bps) for stable Q-value updates.

## 🧠 Model Architectures

### 1. Tabular Q-Learning Agent
* **State Discretization (5 Features):** Converts continuous financial variables into fixed, discrete buckets to construct a finite Q-Table. The state space consists of five core features: **Spread, Time, Inventory, Volume, and Momentum**.
* **Update Rule:** Instead of a standard forward-stepping epsilon-greedy exploration strategy, this model utilizes **Backward Induction**. Because the optimal execution problem has a fixed, finite time horizon (the trading window strictly ends at $T$), the agent computes the optimal policy by stepping backward from the terminal state, ensuring mathematically rigorous convergence for the discrete state-action pairs.
* **Advantage:** Highly interpretable, entirely deterministic once solved, and immune to the gradient instability that plagues deep learning models in noisy high-frequency environments.

### 2. Double Deep Q-Network (DDQN)
* **Continuous State Space (6 Features):** Unlike the tabular model, the DDQN processes a strictly normalized `[-1.0, 1.0]` continuous state array. It utilizes six features: **Spread, Time, Inventory, Volume, Momentum, AND Autocorrelation**. 
* **Architecture:** Implemented in PyTorch. A lightweight Multi-Layer Perceptron (MLP) utilizing a target network to evaluate the greedy policy dictated by the primary network. This eliminates the maximization bias (overestimation of Q-values) common in standard DQN algorithms.
* **Advantage of the DDQN:** The primary advantage of the DDQN is its ability to natively handle **continuous, high-dimensional state spaces**. While the Tabular agent suffers from the "curse of dimensionality" (adding Autocorrelation would cause the Q-table size to explode exponentially), the DDQN scales effortlessly. By not forcing the data into rigid discrete buckets, the neural network learns a much more generalized, fluid trading policy that adapts smoothly to unseen market micro-states.
---


## ⚙️ Installation & Requirements

This project requires Python 3.8+ and PyTorch. 

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/optimal-trade-execution-rl.git](https://github.com/yourusername/optimal-trade-execution-rl.git)
cd optimal-trade-execution-rl

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run
### Training the Agents
You can train either the Tabular or DDQN agent from scratch using the main execution script.

```bash
# Train the Tabular Q-Learning model
python main.py --agent tabular --mode train

# Train the DDQN model
python main.py --agent ddqn --mode train
```

### Evaluating & Generating Reports
Once the models are trained, run the evaluation suite to benchmark them against the Almgren-Chriss and TWAP baselines across the exact same market trajectories.
```bash
# Test the two models
python main.py --agent tabular --mode test
python main.py --agent ddqn --mode test

# Generate the Gain-Loss Ratio CSV and the distribution plots
python utils.py
```

## 📚 References

This project adapts, modifies, and expands upon the theoretical frameworks and experimental designs presented in the following foundational quantitative finance literature:

1. **Hendricks, D., & Wilcox, D. (2014).** *"A reinforcement learning extension to the Almgren-Chriss framework for optimal trade execution."* 2014 IEEE Conference on Computational Intelligence for Financial Engineering & Economics (CIFEr), pp. 457-464. 
   * [View on IEEE Xplore](https://ieeexplore.ieee.org/document/6924109/) | [View on arXiv](https://arxiv.org/abs/1403.2229)

2. **Ning, B., Lin, F. H. T., & Jaimungal, S. (2020).** *"Double Deep Q-Learning for Optimal Execution."* Applied Mathematical Finance, arXiv:1812.06600. 
   * [View on arXiv](https://arxiv.org/abs/1812.06600)

## ⚠️ Disclaimer
For Educational and Research Purposes Only. The code, models, and data provided in this repository do not constitute financial advice, investment recommendations, or trading signals. Quantitative trading in live markets carries significant financial risk. The models herein were evaluated on historical data and are not guaranteed to perform similarly in live, forward-tested market conditions.
