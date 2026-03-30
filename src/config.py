SEED = 42

N_CITIES = 3

#QAOA_P = 3
QAOA_P = 1 # fast to dev
#HOTS = 1024
SHOTS = 20 # fast to dev

A = 15.0
B = 1.0

noise_params = {
    "p1q": 0.005,      # 1-qubit gate error (depolarizing)
    "p2q": 0.02,       # 2-qubit gate error (depolarizing)
    "p_bf": 0.01,      # Bit Flip probability (X)
    "p_pf": 0.01,      # Phase Flip probability (Z)
    "T1": 50e3,        # Longitudinal relaxation time
    "T2": 30e3,        # Transverse relaxation (dephasing) time
    "t1q": 50,         # 1-qubit gate execution time
    "t2q": 300         # 2-qubit gate execution time (CX/CZ)
}