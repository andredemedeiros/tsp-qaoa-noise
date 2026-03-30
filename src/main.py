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
    
    noise_models = create_noise_models(params=config.noise_params)
    for key, model in noise_models.items():
        results[key] = solver.solve(noise_model=model)
    
    analyzer = NoiseAnalyzer(tsp, results, optimal_cost, noise_models, config.noise_params)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()