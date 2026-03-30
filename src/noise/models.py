from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

def create_noise_models(params: dict = None) -> dict:
    """
    Generates a dictionary of Qiskit NoiseModels with customizable parameters.
    
    Args:
        params (dict): Dictionary containing noise parameters. 
                       Example: {"p1q": 0.001, "T1": 100e3, "t2q": 500}
    
    Returns:
        dict: Noise models for 'depolarizing', 'bit_flip', 'phase_flip', and 'thermal'.
    """
    
    # --- 1. Default Parameter Configuration ---
    # Values are typically in nanoseconds (ns) for times (T1, T2, gate times)
    p = {
        "p1q": 0.005,      # 1-qubit gate error (depolarizing)
        "p2q": 0.02,       # 2-qubit gate error (depolarizing)
        "p_bf": 0.01,      # Bit Flip probability (X)
        "p_pf": 0.01,      # Phase Flip probability (Z)
        "T1": 50e3,        # Longitudinal relaxation time
        "T2": 30e3,        # Transverse relaxation (dephasing) time
        "t1q": 50,         # 1-qubit gate execution time
        "t2q": 300         # 2-qubit gate execution time (CX/CZ)
    }
    
    # Update with values passed from main, if any
    if params:
        p.update(params)

    # --- 2. Physical Consistency Validation (Choi-Jamiolkowski Criterion) ---
    # Physically, T2 cannot exceed 2*T1.
    if p["T2"] > 2 * p["T1"]:
        p["T2"] = 2 * p["T1"]
        print(f"Warning: T2 adjusted to {p['T2']} to maintain physical consistency (T2 <= 2*T1).")

    models = {}
    
    # Standard gate sets for noise application
    gates_1q = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "id"]
    gates_2q = ["cx", "cz"]

    # --- 3. Model: Depolarizing ---
    nm_dep = NoiseModel()
    err_dep_1q = depolarizing_error(p["p1q"], 1)
    err_dep_2q = depolarizing_error(p["p2q"], 2)
    
    nm_dep.add_all_qubit_quantum_error(err_dep_1q, gates_1q)
    nm_dep.add_all_qubit_quantum_error(err_dep_2q, gates_2q)
    models["depolarizing"] = nm_dep

    # --- 4. Model: Bit Flip (X) ---
    nm_bf = NoiseModel()
    err_bf_1q = pauli_error([("X", p["p_bf"]), ("I", 1 - p["p_bf"])])
    err_bf_2q = err_bf_1q.expand(err_bf_1q) # Independent error on both qubits
    
    nm_bf.add_all_qubit_quantum_error(err_bf_1q, gates_1q)
    nm_bf.add_all_qubit_quantum_error(err_bf_2q, gates_2q)
    models["bit_flip"] = nm_bf

    # --- 5. Model: Phase Flip (Z) ---
    nm_pf = NoiseModel()
    err_pf_1q = pauli_error([("Z", p["p_pf"]), ("I", 1 - p["p_pf"])])
    err_pf_2q = err_pf_1q.expand(err_pf_1q)
    
    nm_pf.add_all_qubit_quantum_error(err_pf_1q, gates_1q)
    nm_pf.add_all_qubit_quantum_error(err_pf_2q, gates_2q)
    models["phase_flip"] = nm_pf

    # --- 6. Model: Thermal Relaxation ---
    nm_th = NoiseModel()
    # 1-qubit error based on T1, T2, and gate time
    err_th_1q = thermal_relaxation_error(p["T1"], p["T2"], p["t1q"])
    
    # 2-qubit error: assumes relaxation occurs on both qubits during t2q
    err_th_q_individual = thermal_relaxation_error(p["T1"], p["T2"], p["t2q"])
    err_th_2q = err_th_q_individual.expand(err_th_q_individual)
    
    nm_th.add_all_qubit_quantum_error(err_th_1q, gates_1q)
    nm_th.add_all_qubit_quantum_error(err_th_2q, gates_2q)
    models["thermal"] = nm_th

    return models

# from qiskit_aer.noise import (
#     NoiseModel,
#     depolarizing_error,
#     pauli_error,
#     thermal_relaxation_error,
# )


# def create_noise_models() -> dict:
#     """
#     Returns dictionary of NoiseModel objects.

#     {
#         "depolarizing": NoiseModel,
#         "bit_flip": NoiseModel,
#         "phase_flip": NoiseModel,
#         "thermal": NoiseModel
#     }
#     """
#     models = {}

#     # ---------------- Depolarizing ----------------
#     nm_dep = NoiseModel()
#     p1q = 0.005
#     p2q = 0.02

#     nm_dep.add_all_qubit_quantum_error(
#         depolarizing_error(p1q, 1),
#         ["u1", "u2", "u3", "rx", "ry", "rz", "h"]
#     )
#     nm_dep.add_all_qubit_quantum_error(
#         depolarizing_error(p2q, 2),
#         ["cx", "cz"]
#     )

#     models["depolarizing"] = nm_dep

#     # ---------------- Bit Flip ----------------
#     nm_bf = NoiseModel()
#     p_bf = 0.01

#     err_bf_1q = pauli_error([("X", p_bf), ("I", 1 - p_bf)])
#     err_bf_2q = err_bf_1q.expand(err_bf_1q)

#     nm_bf.add_all_qubit_quantum_error(
#         err_bf_1q,
#         ["u1", "u2", "u3", "rx", "ry", "rz", "h"]
#     )
#     nm_bf.add_all_qubit_quantum_error(
#         err_bf_2q,
#         ["cx", "cz"]
#     )

#     models["bit_flip"] = nm_bf

#     # ---------------- Phase Flip ----------------
#     nm_pf = NoiseModel()
#     p_pf = 0.01

#     err_pf_1q = pauli_error([("Z", p_pf), ("I", 1 - p_pf)])
#     err_pf_2q = err_pf_1q.expand(err_pf_1q)

#     nm_pf.add_all_qubit_quantum_error(
#         err_pf_1q,
#         ["u1", "u2", "u3", "rx", "ry", "rz", "h"]
#     )
#     nm_pf.add_all_qubit_quantum_error(
#         err_pf_2q,
#         ["cx", "cz"]
#     )

#     models["phase_flip"] = nm_pf

#     # ---------------- Thermal ----------------
#     nm_th = NoiseModel()

#     T1 = 50e3
#     T2 = 30e3
#     t1q = 50
#     t2q = 300

#     err_1q = thermal_relaxation_error(T1, T2, t1q)
#     err_2q = thermal_relaxation_error(T1, T2, t2q).expand(
#         thermal_relaxation_error(T1, T2, t2q)
#     )

#     nm_th.add_all_qubit_quantum_error(
#         err_1q,
#         ["u1", "u2", "u3", "rx", "ry", "rz", "h"]
#     )
#     nm_th.add_all_qubit_quantum_error(err_2q, ["cx"])

#     models["thermal"] = nm_th

#     return models