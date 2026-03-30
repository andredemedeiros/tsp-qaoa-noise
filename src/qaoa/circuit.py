import numpy as np
from qiskit import QuantumCircuit


def build_qaoa_circuit(cost_op, params: np.ndarray, p: int, n_qubits: int) -> QuantumCircuit:

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    gammas = params[:p]
    betas  = params[p:]

    for layer in range(p):
        gam = gammas[layer]
        bet = betas[layer]

        # Cost operator
        for term, coeff in zip(cost_op.paulis, cost_op.coeffs):
            coeff_val = float(np.real(coeff))

            qubits_in_term = list(np.where(term.z)[0]) 

            if not qubits_in_term:
                continue

            assert len(qubits_in_term) <= 2, (
                f"Term with {len(qubits_in_term)} qubits Z — max hamiltoniano Ising: ZZ"
            )

            if len(qubits_in_term) == 1:
                qc.rz(2 * gam * coeff_val, qubits_in_term[0])
            else:
                q0, q1 = qubits_in_term[0], qubits_in_term[1]
                qc.cx(q0, q1)
                qc.rz(2 * gam * coeff_val, q1)
                qc.cx(q0, q1)

        # Mixer
        qc.rx(2 * bet, range(n_qubits))

    qc.measure_all()
    return qc