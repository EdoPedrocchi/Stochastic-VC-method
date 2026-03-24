import numpy as np
from typing import Dict, List, Any

class CapTableEngine:
    """
    Simulates the dilution of an initial equity stake through multiple 
    funding rounds (Series B, C, etc.) until exit.
    """

    def __init__(self, initial_ownership: float):
        """
        Args:
            initial_ownership: The percentage of the company owned today (e.g., 0.10 for 10%).
        """
        self.initial_ownership = initial_ownership

    def simulate_dilution(self, 
                          revenue_paths: np.ndarray, 
                          n_sims: int, 
                          years: int) -> np.ndarray:
        """
        Simulates dilution rounds based on revenue milestones and time.
        
        Args:
            revenue_paths: The [n_sims, years] matrix from StartupTrajectoryModel.
            n_sims: Number of simulations.
            years: Projection horizon.
            
        Returns:
            An array [n_sims] representing the final ownership percentage at exit.
        """
        # We start with the current ownership for all simulations
        current_ownership = np.full(n_sims, self.initial_ownership)
        
        # Define typical funding round triggers (Every ~2 years)
        # In a real model, these could be triggered by 'Cash Runway' from StartupModel
        round_years = [2, 4] 
        
        for yr in round_years:
            if yr >= years:
                break
                
            # Stochastic Dilution per round:
            # Usually 15% to 25% of the company is sold in each round.
            # We use a Beta distribution to keep it between 10% and 40%
            # alpha=2, beta=5 gives a mean of ~28% with a right lean
            dilution_per_round = np.random.beta(2, 7, n_sims) * 0.5 + 0.1
            
            # Logic: If revenue growth is poor (Bottom 20%), dilution is higher (Down Round)
            # If revenue growth is great (Top 20%), dilution is lower (High Valuation)
            revenue_growth = revenue_paths[:, yr] / revenue_paths[:, yr-1]
            growth_threshold_low = np.percentile(revenue_growth, 20)
            growth_threshold_high = np.percentile(revenue_growth, 80)
            
            # Penalize poor performers with 1.5x dilution (Down Round / Pay-to-play)
            dilution_per_round[revenue_growth < growth_threshold_low] *= 1.5
            
            # Reward top performers (More leverage, less dilution)
            dilution_per_round[revenue_growth > growth_threshold_high] *= 0.8
            
            # Clip to ensure physical possibility (can't dilute more than 90% in one round)
            dilution_per_round = np.clip(dilution_per_round, 0.05, 0.90)
            
            # Apply dilution: Ownership_new = Ownership_old * (1 - Dilution)
            current_ownership = current_ownership * (1 - dilution_per_round)

        # Final "Option Pool" Refresh: 
        # Usually, right before an exit, the option pool is expanded by ~5%
        exit_refresh = np.random.uniform(0.03, 0.07, n_sims)
        current_ownership = current_ownership * (1 - exit_refresh)

        return current_ownership

# --- Quick Technical Test ---
if __name__ == "__main__":
    # Mock revenue path (1000 sims, 5 years)
    mock_rev = np.ones((1000, 6)) 
    # Simulate some growth
    mock_rev[:, 2] = 1.5 
    mock_rev[:, 4] = 2.5
    
    engine = CapTableEngine(initial_ownership=0.10) # 10% Stake
    final_stakes = engine.simulate_dilution(mock_rev, n_sims=1000, years=5)
    
    print(f"--- Cap Table Dilution Results ---")
    print(f"Initial Ownership: 10.00%")
    print(f"Mean Ownership at Exit: {np.mean(final_stakes):.2%}")
    print(f"Min Ownership (Down-round hell): {np.min(final_stakes):.2%}")
    print(f"Max Ownership (Capital efficient): {np.max(final_stakes):.2%}")
