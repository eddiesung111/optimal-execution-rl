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
![Execution Edge Distributions](results/distribution_plots.png)
*(The graph above illustrates the probability density of the RL agents' outperformance versus the baseline. A right-skewed distribution past the `0.0` break-even line indicates consistent profitability over the benchmark.)*
