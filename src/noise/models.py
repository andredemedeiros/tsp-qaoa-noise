from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)


def create_noise_models() -> dict:
    """
    Returns dictionary of NoiseModel objects.

    {
        "depolarizing": NoiseModel,
        "bit_flip": NoiseModel,
        "phase_flip": NoiseModel,
        "thermal": NoiseModel
    }
    """
    models = {}

    # ---------------- Depolarizing ----------------
    nm_dep = NoiseModel()
    p1q = 0.005
    p2q = 0.02

    nm_dep.add_all_qubit_quantum_error(
        depolarizing_error(p1q, 1),
        ["u1", "u2", "u3", "rx", "ry", "rz", "h"]
    )
    nm_dep.add_all_qubit_quantum_error(
        depolarizing_error(p2q, 2),
        ["cx", "cz"]
    )

    models["depolarizing"] = nm_dep

    # ---------------- Bit Flip ----------------
    nm_bf = NoiseModel()
    p_bf = 0.01

    err_bf_1q = pauli_error([("X", p_bf), ("I", 1 - p_bf)])
    err_bf_2q = err_bf_1q.expand(err_bf_1q)

    nm_bf.add_all_qubit_quantum_error(
        err_bf_1q,
        ["u1", "u2", "u3", "rx", "ry", "rz", "h"]
    )
    nm_bf.add_all_qubit_quantum_error(
        err_bf_2q,
        ["cx", "cz"]
    )

    models["bit_flip"] = nm_bf

    # ---------------- Phase Flip ----------------
    nm_pf = NoiseModel()
    p_pf = 0.01

    err_pf_1q = pauli_error([("Z", p_pf), ("I", 1 - p_pf)])
    err_pf_2q = err_pf_1q.expand(err_pf_1q)

    nm_pf.add_all_qubit_quantum_error(
        err_pf_1q,
        ["u1", "u2", "u3", "rx", "ry", "rz", "h"]
    )
    nm_pf.add_all_qubit_quantum_error(
        err_pf_2q,
        ["cx", "cz"]
    )

    models["phase_flip"] = nm_pf

    # ---------------- Thermal ----------------
    nm_th = NoiseModel()

    T1 = 50e3
    T2 = 30e3
    t1q = 50
    t2q = 300

    err_1q = thermal_relaxation_error(T1, T2, t1q)
    err_2q = thermal_relaxation_error(T1, T2, t2q).expand(
        thermal_relaxation_error(T1, T2, t2q)
    )

    nm_th.add_all_qubit_quantum_error(
        err_1q,
        ["u1", "u2", "u3", "rx", "ry", "rz", "h"]
    )
    nm_th.add_all_qubit_quantum_error(err_2q, ["cx"])

    models["thermal"] = nm_th

    return models