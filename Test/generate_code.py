import numpy as np
import pandas as pd


def generate_synthetic_projects(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    budget = rng.uniform(80_000, 1_200_000, n_rows)
    duration = rng.integers(2, 30, n_rows)  # months
    labor_cost = budget * rng.uniform(0.2, 0.55, n_rows)
    material_cost = budget * rng.uniform(0.15, 0.5, n_rows)
    overhead = budget * rng.uniform(0.05, 0.2, n_rows)

    delay_pct = np.clip(rng.normal(12, 10, n_rows), 0, 60)
    resource_utilization_pct = np.clip(rng.normal(78, 12, n_rows), 35, 100)

    # Actual cost increases with delay + low utilization
    inefficiency_factor = (delay_pct / 100) * 0.45 + ((100 - resource_utilization_pct) / 100) * 0.30
    baseline_cost = labor_cost + material_cost + overhead
    actual_cost = baseline_cost * (1 + inefficiency_factor) + rng.normal(0, budget * 0.03, n_rows)

    # Target: profit margin (%)
    profit_margin_pct = ((budget - actual_cost) / budget) * 100 + rng.normal(0, 2.2, n_rows)
    profit_margin_pct = np.clip(profit_margin_pct, -35, 55)

    df = pd.DataFrame(
        {
            "budget": budget.round(2),
            "duration_months": duration,
            "labor_cost": labor_cost.round(2),
            "material_cost": material_cost.round(2),
            "overhead": overhead.round(2),
            "actual_cost": actual_cost.round(2),
            "delay_pct": delay_pct.round(2),
            "resource_utilization_pct": resource_utilization_pct.round(2),
            "profit_margin_pct": profit_margin_pct.round(2),
        }
    )
    return df


if __name__ == "__main__":
    data = generate_synthetic_projects(n_rows=1200, seed=42)
    output_path = "project_profitability_data.csv"
    data.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to: {output_path}")
    print(data.head(5))