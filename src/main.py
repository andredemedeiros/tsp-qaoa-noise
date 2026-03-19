from tsp.instance import TSPInstance
from tsp.qubo import TSPtoQUBO
from qaoa.solver import QAOASolver
from noise.models import create_noise_models
from analysis.analyzer import NoiseAnalyzer
import config

def main():

    tsp = TSPInstance(config.N_CITIES, seed=config.SEED)
    optimal_route, optimal_cost = tsp.brute_force()

    qubo = TSPtoQUBO(tsp, A=config.A, B=config.B)

    solver = QAOASolver(qubo, p=config.QAOA_P, shots=config.SHOTS)

    results = {}
    results["ideal"] = solver.solve()
    
    noise_models = create_noise_models()
    for key, model in noise_models.items():
        results[key] = solver.solve(noise_model=model)
    
    analyzer = NoiseAnalyzer(tsp, results, optimal_cost, noise_models)
    analyzer.print_summary()

    analyzer.plot_convergence()
    analyzer.plot_solution_quality()
    analyzer.plot_count_distributions()
    analyzer.plot_summary_table()

if __name__ == "__main__":
    main()