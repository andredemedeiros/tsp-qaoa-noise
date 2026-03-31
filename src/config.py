# dont change
SEED = 42       # Random seed for reproducibility
N_CITIES = 3    # Number of cities in the TSP
A = 15.0        # QUBO penalty parameter for constraints, can adjust
B = 1.0         # QUBO weight for cost function, can adjust


# can change
QAOA_P = 3             # QAOA circuit depth (p)
SHOTS = 1024           # Number of measurement shots
N_STARTS = 3           # Number of random initializations for optimization
noise_params = {       # Noise model parameters
    "p1q": 0.005,      # 1-qubit gate depolarizing error probability
    "p2q": 0.02,       # 2-qubit gate depolarizing error probability
    "p_bf": 0.01,      # Probability of bit-flip error (X)
    "p_pf": 0.01,      # Probability of phase-flip error (Z)
    "T1": 50e3,        # Longitudinal relaxation time
    "T2": 30e3,        # Transverse relaxation (dephasing) time
    "t1q": 50,         # 1-qubit gate execution time
    "t2q": 300         # 2-qubit gate execution time for 2-qubit gates (CX/CZ)
}