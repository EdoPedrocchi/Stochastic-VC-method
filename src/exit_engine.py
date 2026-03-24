import numpy as np
import json
import os
from typing import Dict, Any, Optional

class ExitEngine:
    """
    Final Monte Carlo engine that combines operational paths, dilution, 
    and sector multiples to calculate Exit Value and Investor Returns.
    """

    def __init__(self, sector: str, json_path: str = "data/sector_multiples.json"):
        """
        Args:
            sector: The sector key from the JSON (e.g., 'SaaS', 'Fintech').
            json_path: Path to the sector multiples configuration.
        """
        self.sector = sector
        self.multiples_data = self._load_multiples(json_path)

    def _load_multiples(self, path: str) -> Dict[str, float]:
        """Loads mean and std_dev for the chosen sector from JSON."""
        if not os.path.exists(path):
            # Fallback to default if JSON is missing
            return {"mean": 8.0, "std_dev": 2.5}
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        sector_info = data.get("sectors", {}).get(self.sector)
        if not sector_info:
            raise ValueError(f"Sector '{self.sector}' not found in {path}")
            
        return sector_info

    def run_exit_analysis(self, 
                          revenue_at_exit: np.ndarray, 
                          survival_mask: np.ndarray, 
                          final_ownership: np.ndarray, 
                          initial_investment: float) -> Dict[str, Any]:
        """
        Calculates the final distribution of outcomes.
        
        Args:
            revenue_at_exit: Array of projected revenues at Year T.
            survival_mask: Boolean array (True if company survived).
            final_ownership: Array of ownership % after dilution.
            initial_investment: The USD amount invested at Day 0.
            
        Returns:
            Dictionary with distributions of Exit Value, MOIC, and summary stats.
        """
        n_sims = len(revenue_at_exit)
        
        # 1. Generate Stochastic Multiples (EV/Revenue)
        # We use a Log-Normal distribution for multiples because they can't be negative 
        # and tend to have a "long tail" (some exits are massive outliers).
        m_mean = self.multiples_data["mean"]
        m_std = self.multiples_data["std_dev"]
        
        # Convert arithmetic mean/std to log-normal parameters
        mu_log = np.log(m_mean**2 / np.sqrt(m_std**2 + m_mean**2))
        sigma_log = np.sqrt(np.log(1 + (m_std**2 / m_mean**2)))
        
        multiples = np.random.lognormal(mu_log, sigma_log, n_sims)
        
        # 2. Calculate Gross Exit Value (Enterprise Value)
        # EV = Revenue * Multiple
        exit_ev = revenue_at_exit * multiples
        
        # 3. Apply Power Law / Survival Filter
        # If survival_mask is False, the value is $0
        exit_ev = np.where(survival_mask, exit_ev, 0.0)
        
        # 4. Calculate Investor Proceeds
        # Proceeds = Exit EV * Final Ownership %
        investor_proceeds = exit_ev * final_ownership
        
        # 5. Returns Metrics
        # MOIC = Multiple on Invested Capital (Proceeds / Investment)
        moic = investor_proceeds / initial_investment
        
        # Probability of a "Home Run" (MOIC > 10x)
        home_run_prob = np.sum(moic >= 10.0) / n_sims
        
        # Probability of Total Loss (MOIC approx 0)
        loss_prob = np.sum(moic < 0.1) / n_sims

        return {
            "exit_ev_dist": exit_ev,
            "investor_proceeds_dist": investor_proceeds,
            "moic_dist": moic,
            "stats": {
                "mean_moic": np.mean(moic),
                "median_moic": np.median(moic),
                "probability_of_loss": loss_prob,
                "probability_of_homerun": home_run_prob,
                "expected_exit_value_mean": np.mean(exit_ev[survival_mask]) if any(survival_mask) else 0
            }
        }

# --- Quick Technical Test ---
if __name__ == "__main__":
    # Dummy data for testing
    n = 10000
    revs = np.random.normal(10e6, 2e6, n) # $10M Rev
    mask = np.random.choice([True, False], n, p=[0.4, 0.6]) # 60% failure
    own = np.random.uniform(0.05, 0.08, n) # 5-8% ownership
    
    engine = ExitEngine(sector="SaaS")
    results = engine.run_exit_analysis(revs, mask, own, initial_investment=500_000)
    
    print(f"--- VC Exit Engine Results ---")
    print(f"Expected MOIC: {results['stats']['mean_moic']:.2f}x")
    print(f"Probability of Total Loss: {results['stats']['probability_of_loss']:.1%}")
    print(f"Probability of 10x Return: {results['stats']['probability_of_homerun']:.1%}")
