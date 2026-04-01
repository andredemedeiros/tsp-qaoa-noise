from tsp.instance import TSPInstance
from tsp.qubo import TSPtoQUBO
from qaoa.solver import QAOASolver
from noise.models import create_noise_models
from analysis.analyzer import NoiseAnalyzer
import config
import time

def main():

    tsp = TSPInstance(config.N_CITIES, seed=config.SEED)
    optimal_route, optimal_cost = tsp.brute_force()

    qubo = TSPtoQUBO(tsp, A=config.A, B=config.B)

    solver = QAOASolver(qubo, p=config.QAOA_P, shots=config.SHOTS)


    optimizers = ['trust-constr','COBYLA','L-BFGS-B','POWELL','Nelder-Mead']
    exec_time = []
    for opt in optimizers:
        results = {}
        
        start_time = time.perf_counter()
        results["ideal"] = solver.solve(n_starts=config.N_STARTS,method=opt)
        
        noise_models = create_noise_models(params=config.noise_params)
        for key, model in noise_models.items():
            results[key] = solver.solve(noise_model=model,n_starts=config.N_STARTS,method=opt)
        end_time = time.perf_counter()

        solver_time = end_time - start_time
        exec_time.append((opt,solver_time))

        NoiseAnalyzer(
                tsp, qubo, results, optimal_cost,
                noise_models, config.noise_params,
                config.N_STARTS, config.SHOTS, opt
            ).run_full_analysis()
    
    
    exec_time.sort(key=lambda x: x[1], reverse=True)
    print("\n".join([f"Otimizer: {opt:15} | Time: {dur:.2f}s" for opt, dur in exec_time]))

if __name__ == "__main__":
    main()