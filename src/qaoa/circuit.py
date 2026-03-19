import numpy as np
from qiskit import QuantumCircuit


def build_qaoa_circuit(cost_op, params: np.ndarray, p: int, n_qubits: int) -> QuantumCircuit:

    qc = QuantumCircuit(n_qubits)

    qc.h(range(n_qubits))

    gammas = params[:p]
    betas = params[p:]

    for layer in range(p):
        gam = gammas[layer]
        bet = betas[layer]

        # Cost operator
        for term, coeff in zip(cost_op.paulis, cost_op.coeffs):

            coeff_val = float(np.real(coeff))
            term_str = str(term)

            # 🔴 CRÍTICO: inverter para alinhar com Qiskit
            qubits_in_term = [
                q for q, p_char in enumerate(term_str[::-1]) if p_char != "I"
            ]

            if not qubits_in_term:
                continue

            if len(qubits_in_term) == 1:
                qc.rz(2 * gam * coeff_val, qubits_in_term[0])
            else:
                for k in range(len(qubits_in_term) - 1):
                    qc.cx(qubits_in_term[k], qubits_in_term[k + 1])

                qc.rz(2 * gam * coeff_val, qubits_in_term[-1])

                for k in reversed(range(len(qubits_in_term) - 1)):
                    qc.cx(qubits_in_term[k], qubits_in_term[k + 1])

        # Mixer
        qc.rx(2 * bet, range(n_qubits))

    qc.measure_all()
    return qc