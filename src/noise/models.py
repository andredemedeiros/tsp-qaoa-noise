from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

def _default_noise_params(params: dict = None) -> dict:
    """Return the default noise parameters updated with the given overrides."""
    defaults = {
        "p1q": 0.005,      # 1-qubit gate error (depolarizing)
        "p2q": 0.02,       # 2-qubit gate error (depolarizing)
        "p_bf": 0.01,      # Bit Flip probability (X)
        "p_pf": 0.01,      # Phase Flip probability (Z)
        "T1": 50e3,        # Longitudinal relaxation time
        "T2": 30e3,        # Transverse relaxation (dephasing) time
        "t1q": 50,         # 1-qubit gate execution time
        "t2q": 300         # 2-qubit gate execution time (CX/CZ)
    }
    if params:
        defaults.update(params)

    if defaults["T2"] > 2 * defaults["T1"]:
        defaults["T2"] = 2 * defaults["T1"]
        print(f"Warning: T2 adjusted to {defaults['T2']} to maintain physical consistency (T2 <= 2*T1).")

    return defaults


def create_depolarizing_noise_model(params: dict = None) -> NoiseModel:
    """Returns a depolarizing noise model."""
    p = _default_noise_params(params)
    gates_1q = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "id"]
    gates_2q = ["cx", "cz"]

    nm_dep = NoiseModel()
    err_dep_1q = depolarizing_error(p["p1q"], 1)
    err_dep_2q = depolarizing_error(p["p2q"], 2)

    nm_dep.add_all_qubit_quantum_error(err_dep_1q, gates_1q)
    nm_dep.add_all_qubit_quantum_error(err_dep_2q, gates_2q)

    return nm_dep


def create_bit_flip_noise_model(params: dict = None) -> NoiseModel:
    """Returns a bit flip noise model."""
    p = _default_noise_params(params)
    gates_1q = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "id"]
    gates_2q = ["cx", "cz"]

    nm_bf = NoiseModel()
    err_bf_1q = pauli_error([("X", p["p_bf"]), ("I", 1 - p["p_bf"])])
    err_bf_2q = err_bf_1q.expand(err_bf_1q)

    nm_bf.add_all_qubit_quantum_error(err_bf_1q, gates_1q)
    nm_bf.add_all_qubit_quantum_error(err_bf_2q, gates_2q)

    return nm_bf


def create_phase_flip_noise_model(params: dict = None) -> NoiseModel:
    """Returns a phase flip noise model."""
    p = _default_noise_params(params)
    gates_1q = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "id"]
    gates_2q = ["cx", "cz"]

    nm_pf = NoiseModel()
    err_pf_1q = pauli_error([("Z", p["p_pf"]), ("I", 1 - p["p_pf"])])
    err_pf_2q = err_pf_1q.expand(err_pf_1q)

    nm_pf.add_all_qubit_quantum_error(err_pf_1q, gates_1q)
    nm_pf.add_all_qubit_quantum_error(err_pf_2q, gates_2q)

    return nm_pf


def create_thermal_noise_model(params: dict = None) -> NoiseModel:
    """Returns a thermal relaxation noise model."""
    p = _default_noise_params(params)
    gates_1q = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "id"]
    gates_2q = ["cx", "cz"]

    nm_th = NoiseModel()
    err_th_1q = thermal_relaxation_error(p["T1"], p["T2"], p["t1q"])
    err_th_q_individual = thermal_relaxation_error(p["T1"], p["T2"], p["t2q"])
    err_th_2q = err_th_q_individual.expand(err_th_q_individual)

    nm_th.add_all_qubit_quantum_error(err_th_1q, gates_1q)
    nm_th.add_all_qubit_quantum_error(err_th_2q, gates_2q)

    return nm_th


def create_noise_models(params: dict = None) -> dict:
    """Returns the default set of noise models."""
    return {
        "depolarizing": create_depolarizing_noise_model(params),
        "bit_flip": create_bit_flip_noise_model(params),
        "phase_flip": create_phase_flip_noise_model(params),
        "thermal": create_thermal_noise_model(params),
    }

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