# Stochastic-VC-method
This `README.md` is tailored for the **VENTURE-STOCHASTIC-EXIT (VSE)** repository. It emphasizes the "Power Law" and "Survival" logic, making it clear that this tool is specifically for high-risk, high-reward private equity and venture capital analysis.

---

# Venture Stochastic Exit (VSE) Framework
### Probabilistic Valuation for Early-Stage Startups

## Overview
Traditional valuation models (like DCF) often fail for startups because they assume a linear path to stability. In reality, startups face **binary outcomes**: they either fail early due to cash exhaustion or achieve exponential growth leading to a liquidity event.

The **VSE Framework** is a Monte Carlo simulation tool that models the entire startup lifecycle. It calculates the **Multiple on Invested Capital (MOIC)** by simulating thousands of paths involving stochastic revenue growth, monthly burn rates, survival probability, and successive dilution rounds.

## Core Methodology
The framework is built on three pillars of venture math:

1.  **Stochastic Survival (The "Burn" Model)**: Instead of assuming a company reaches year 10, the model checks cash balances monthly. If `Cash < 0`, the simulation is marked as a failure ($0 exit value).
2.  **Performance-Based Dilution**: Subsequent funding rounds (Series B, C, etc.) are modeled as stochastic events. High-growth paths suffer less dilution, while stagnant growth triggers "Down Rounds" with aggressive equity contraction.
3.  **Exit Multiples (The Power Law)**: Revenue at exit is multiplied by industry-specific EV/Revenue multiples. These multiples are sampled from a **Log-Normal distribution** to reflect the "long-tail" nature of venture returns.

## Repository Structure
```text
VENTURE-STOCHASTIC-EXIT/
├── data/
│   └── sector_multiples.json    # Configurable EV/Rev benchmarks
├── src/
│   ├── startup_model.py         # Revenue growth & monthly burn logic
│   ├── cap_table_engine.py      # Series A/B/C dilution & Down Round logic
│   ├── exit_engine.py           # Multiple sampling & MOIC calculation
│   └── utils.py                 # Math helpers
├── app/
│   └── dashboard.py             # Streamlit UI for VC analysis
├── requirements.txt             # Dependencies (numpy, pandas, streamlit)
└── README.md
```

## Mathematical Implementation

### Revenue Growth
Revenue follows a Geometric Brownian Motion (GBM):
$$R_{t} = R_{t-1} \cdot e^{(\mu - 0.5\sigma^2) + \sigma \epsilon}$$
Where $\mu$ represents the aggressive growth targets of a startup and $\sigma$ represents the high volatility of early-stage execution.

### Investor Returns (MOIC)
The Multiple on Invested Capital for each simulation $i$ is calculated as:
$$MOIC_i = \frac{(Revenue_{T,i} \times Multiple_i) \times Ownership_{diluted,i}}{Initial\ Investment}$$
*Note: If the startup runs out of cash before year $T$, $Revenue_{T,i}$ is set to 0.*

## Setup and Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Sector Multiples
You can modify `data/sector_multiples.json` to update the mean and standard deviation of EV/Revenue multiples for different industries (e.g., SaaS, AI, Fintech).

### 3. Launch the Dashboard
```bash
streamlit run app/dashboard.py
```

## Key Features
* **Survival Pie Chart**: Visualizes the percentage of simulations that "defaulted" before reaching an exit.
* **MOIC Histogram**: Shows the distribution of successful exits, highlighting the "Home Run" potential (10x+ returns).
* **Dilution Tracking**: Accounts for the "Option Pool Refresh" and the impact of future venture rounds on the initial cap table.

## Disclaimer
This tool is for educational and analytical purposes. Venture capital involves extreme risk, and no mathematical model can perfectly predict startup survival or market sentiment at the time of exit.

