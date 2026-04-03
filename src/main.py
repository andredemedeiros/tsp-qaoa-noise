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
    start = time.time()
    results["ideal"] = solver.solve(noise_model=None, n_starts=config.N_STARTS)
    results["ideal"]["elapsed"] = time.time() - start

    noise_specs = [
        {
            "key": "depolarizing",
            "label": "depolarizing",
            "create_fn": noise.models.create_depolarizing_noise_model,
            "prepare_params": lambda p, params: params.update({"p1q": p, "p2q": p}) or params,
        },
        {
            "key": "bit_flip",
            "label": "bit_flip",
            "create_fn": noise.models.create_bit_flip_noise_model,
            "prepare_params": lambda p, params: params.update({"p_bf": p}) or params,
        },
        {
            "key": "phase_flip",
            "label": "phase_flip",
            "create_fn": noise.models.create_phase_flip_noise_model,
            "prepare_params": lambda p, params: params.update({"p_pf": p}) or params,
        },
        {
            "key": "thermal",
            "label": "thermal",
            "create_fn": noise.models.create_thermal_noise_model,
            "prepare_params": lambda p, params: params.update({
                "T1": params["T1"] / p,
                "T2": params["T2"] / p,
            }) or params,
        },
    ]

    noise_probs = np.linspace(0.1, 1.0, 10)
    all_sweep_results = {}

    for spec in noise_specs:
        sweep_results = [{
                "p": 0.0,
                "best_cost": results["ideal"]["best"]["cost"],
                "valid_ratio": results["ideal"].get("valid_ratio", 0) * 100,
                "elapsed": results["ideal"]["elapsed"],
            }]
        print("\n" + "=" * 60)
        print(f"Running sweep for {spec['label']} noise...")

        for p in noise_probs:
            params = config.noise_params.copy()
            params = spec["prepare_params"](p, params)
            noise_model = spec["create_fn"](params)

            print("-" * 60)
            print(f"Running QAOA with {spec['label']} noise p={p:.2f}...")

            start = time.time()
            res = solver.solve(noise_model=noise_model, verbose=True, n_starts=config.N_STARTS)
            elapsed = time.time() - start
            res["elapsed"] = elapsed

            print(
                f"{spec['label'].replace('_', ' ').capitalize()} p={p:.2f} | Best Cost: {res['best']['cost']:.4f} | "
                f"Valid Ratio: {res['valid_ratio']*100:.2f}% | Time: {elapsed:.2f}s"
            )

            sweep_results.append({
                "p": float(p),
                "best_cost": res.get("best", {}).get("cost", np.nan),
                "valid_ratio": res.get("valid_ratio", 0) * 100,
                "elapsed": elapsed,
                "result": res,
            })

            if p == 0.0:
                results[spec["key"]] = res

        all_sweep_results[spec["key"]] = sweep_results

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
    analyzer.plot_noise_sweep_comparison(
        all_sweep_results,
        x_key="p",
        x_label="Noise Probability (p)",
        title_prefix="Noise Sweep",
        output_name="noise_sweep_comparison.png",
    )

if __name__ == "__main__":
    main()