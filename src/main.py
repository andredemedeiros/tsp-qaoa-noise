import time
import numpy as np

from tsp.instance import TSPInstance
from tsp.qubo import TSPtoQUBO
from qaoa.solver import QAOASolver
import noise.models
from analysis.analyzer import NoiseAnalyzer
import config

def main():

    tsp = TSPInstance(config.N_CITIES, seed=config.SEED)
    optimal_route, optimal_cost = tsp.brute_force()

    qubo = TSPtoQUBO(tsp, A=config.A, B=config.B)

    solver = QAOASolver(qubo, p=config.QAOA_P, shots=config.SHOTS)

    results = {}
    results["ideal"] = solver.solve(n_starts=config.N_STARTS)

    noise_type = "bitflip" # parameter to appear on plots, in filenames
    noise_probs = np.linspace(0.0, 1.0, 11)
    sweep_results = []

    for p in noise_probs:
        params = config.noise_params.copy()
        params["p_bf"] = p # change to the appropriate parameter for the noise type being tested

        if p == 0:
            noise_model = None
        else:
            noise_model = noise.models.create_thermal_noise_model(params)

        print("-" * 60)
        print(f"Running QAOA with {noise_type} noise p={p:.2f}...")

        start = time.time()
        res = solver.solve(noise_model=noise_model, verbose=True, n_starts=config.N_STARTS)
        res["elapsed"] = time.time() - start

        print(f"{noise_type.capitalize()} p={p:.2f} | Best Cost: {res['best']['cost']:.4f} | Valid Ratio: {res['valid_ratio']*100:.2f}% | Time: {res['elapsed']:.2f}s")

        sweep_results.append({
            "p": float(p),
            "best_cost": res.get("best", {}).get("cost", np.nan),
            "valid_ratio": res.get("valid_ratio", 0) * 100,
            "elapsed": res["elapsed"],
            "result": res,
        })

    analyzer = NoiseAnalyzer(
        tsp,
        qubo,
        results,
        optimal_cost,
        None,
        config.noise_params,
        config.N_STARTS,
        config.SHOTS,
    )
    analyzer.print_summary()
    analyzer.plot_noise_sweep(
        sweep_results,
        x_key="p",
        x_label=f"{noise_type.capitalize()} Probability (p)",
        title_prefix=f"{noise_type.capitalize()} Sweep",
        output_name=f"{noise_type.replace(' ', '_')}_noise_sweep.png",
    )

if __name__ == "__main__":
    main()