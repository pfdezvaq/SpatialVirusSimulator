from spatial_virus_simulator import (
    SpatialVirusSimulator,
    simulate_multiple_generations,
    compare_clustering_across_mois,
    analyse_clustering_for_snapshot,
    plot_clustering_analysis
)

import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameters
params = {
    'b': 150,
    'num_cells': 25000,
    'wt_initial': 12500,
    'a_initial': 20500,
    'sigma': 0.4557,
    'gamma': 0.5164,
    'max_time': 48,
    'tau': 12,
    'degradation_rate': 0.1,
    'diffusion_rate': 0.2,
    'plate_diameter': 35,
    'cell_diameter': 12,
    'grid_size': 200,
    'local_retention_rate': 0.6,
    'extracelular_diffusion_rate': 0.1,
    'cell_to_cell_diffusion_rate': 0.3
}

def run_single_simulation():
    simulator = SpatialVirusSimulator(params)
    simulator.run_simulation()
    simulator.plot_spatial_state(
        title="Final simulation state",
        save_path="simulation_result.png"
    )

def run_multiple_generations():
    num_generations = 3
    simulate_multiple_generations(params, num_generations)

def run_clustering_analysis():
    print("=== Single clustering analysis ===")
    simulator = SpatialVirusSimulator(params)
    simulator.run_simulation()

    fixed_time = 15
    clustering_results = simulator.analyse_clustering(time_point=fixed_time)

    if clustering_results is not None:
        plot_clustering_analysis(clustering_results, save_path="clustering_result.png")
    else:
        print("Clustering analysis failed.")

    print("\n=== Clustering comparison by MOI ===")
    moi_values = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1]
    compare_clustering_across_mois(params, moi_values, num_simulations=3)

if __name__ == "__main__":
    # Uncomment the desired analysis to run:
    # run_single_simulation()
    # run_multiple_generations()
    run_clustering_analysis()
