import numpy as np
import pandas as pd
from typing import Dict, Any

class StartupTrajectoryModel:
    """
    Simulates the operational path of a startup including stochastic revenue 
    growth and cash burn dynamics.
    """

    def __init__(self, 
                 initial_revenue: float, 
                 initial_cash: float, 
                 monthly_burn: float, 
                 growth_mu: float, 
                 growth_sigma: float):
        """
        Args:
            initial_revenue: Current Annual Recurring Revenue (ARR).
            initial_cash: Current cash in bank.
            monthly_burn: Current net cash out per month.
            growth_mu: Expected annual log-growth rate.
            growth_sigma: Volatility of annual growth.
        """
        self.initial_revenue = initial_revenue
        self.initial_cash = initial_cash
        self.monthly_burn = monthly_burn
        self.mu = growth_mu
        self.sigma = growth_sigma

    def simulate_path(self, years: int, n_sims: int = 10000) -> Dict[str, np.ndarray]:
        """
        Generates stochastic paths for revenue and monitors survival.
        
        Returns:
            A dictionary containing revenue paths and a survival mask.
        """
        # 1. Revenue Projection (Geometric Brownian Motion)
        # We use a yearly step for revenue paths
        dt = 1.0
        Z = np.random.standard_normal((n_sims, years))
        
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        
        # Calculate annual growth factors
        growth_factors = np.exp(drift + diffusion)
        
        # Path generation: [n_sims, years + 1]
        revenue_paths = np.zeros((n_sims, years + 1))
        revenue_paths[:, 0] = self.initial_revenue
        
        for t in range(1, years + 1):
            revenue_paths[:, t] = revenue_paths[:, t-1] * growth_factors[:, t-1]

        # 2. Burn and Survival Logic
        # Simplification: Burn rate scales with revenue growth but improves with efficiency.
        # As revenue grows, we assume the 'Efficiency Factor' reduces the relative burn.
        efficiency_improvement = 0.10 # 10% reduction in burn-to-revenue ratio per year
        
        # We simulate month-by-month for cash runway (Total months = years * 12)
        total_months = years * 12
        cash_paths = np.zeros((n_sims, total_months + 1))
        cash_paths[:, 0] = self.initial_cash
        
        survival_mask = np.ones(n_sims, dtype=bool)
        
        # Monthly simulation to check for bankruptcy
        for m in range(1, total_months + 1):
            # Estimate current annual revenue for that month (interpolated)
            year_idx = (m - 1) // 12
            current_annual_rev = revenue_paths[:, year_idx]
            
            # Stochastic monthly burn: base burn adjusted by revenue scale
            # If revenue grows 2x, burn doesn't necessarily grow 2x (Operating Leverage)
            current_monthly_burn = self.monthly_burn * (1 + 0.5 * (current_annual_rev / self.initial_revenue - 1))
            current_monthly_burn *= (1 - efficiency_improvement)**year_idx
            
            # Update cash
            cash_paths[:, m] = cash_paths[:, m-1] - current_monthly_burn
            
            # Check for "Default" (Cash < 0)
            # If a simulation hits 0 cash, it is marked as 'failed'
            failed_this_month = cash_paths[:, m] <= 0
            survival_mask[failed_this_month] = False

        return {
            "revenue_at_exit": revenue_paths[:, -1],
            "full_revenue_paths": revenue_paths,
            "survival_mask": survival_mask,
            "final_cash": cash_paths[:, -1]
        }

# --- Quick Technical Test ---
if __name__ == "__main__":
    # Setup a "Series A" Startup
    model = StartupTrajectoryModel(
        initial_revenue=2_000_000, # $2M ARR
        initial_cash=5_000_000,    # $5M in bank
        monthly_burn=250_000,      # $250k burn/month
        growth_mu=0.50,            # 50% target growth
        growth_sigma=0.30          # High uncertainty
    )
    
    results = model.simulate_path(years=5, n_sims=1000)
    
    survived = np.sum(results["survival_mask"])
    print(f"--- Startup Path Simulation ---")
    print(f"Survival Rate after 5 years: {survived/1000:.1%}")
    print(f"Median Revenue at Exit (Survivors): ${np.median(results['revenue_at_exit'][results['survival_mask']]):,.0f}")
