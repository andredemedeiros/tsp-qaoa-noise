import numpy as np
from tsp.instance import TSPInstance
from qiskit.quantum_info import SparsePauliOp


class TSPtoQUBO:

    def __init__(self, tsp: TSPInstance, A: float = 5.0, B: float = 1.0):
        self.tsp = tsp
        self.n = tsp.n
        self.A = A
        self.B = B
        self.n_qubits = self.n ** 2

    def var_index(self, city: int, pos: int) -> int:
        return city * self.n + pos

    def build_qubo(self) -> np.ndarray:
        n = self.n
        Q = np.zeros((self.n_qubits, self.n_qubits))

        # Constraint 1
        for i in range(n):
            for t in range(n):
                vi = self.var_index(i, t)
                for s in range(t + 1, n):
                    vj = self.var_index(i, s)
                    Q[vi, vj] += 2 * self.A
                Q[vi, vi] += -2 * self.A

        # Constraint 2
        for t in range(n):
            for i in range(n):
                vi = self.var_index(i, t)
                for j in range(i + 1, n):
                    vj = self.var_index(j, t)
                    Q[vi, vj] += 2 * self.A
                Q[vi, vi] += -2 * self.A

        # Objective (SEM duplicação)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for t in range(n):
                    vi = self.var_index(i, t)
                    vj = self.var_index(j, (t + 1) % n)

                    if vi < vj:
                        Q[vi, vj] += self.B * self.tsp.dist[i][j]

        return Q

    def qubo_to_ising(self, Q: np.ndarray):
        n = self.n_qubits
        h = np.zeros(n)
        J = np.zeros((n, n))
        offset = 0.0

        for i in range(n):
            h[i] -= Q[i, i] / 2
            offset += Q[i, i] / 4

            for j in range(i + 1, n):
                Qij = Q[i, j]
                J[i, j] += Qij / 4
                h[i] -= Qij / 4
                h[j] -= Qij / 4
                offset += Qij / 4

        return h, J, offset

    def build_cost_operator(self) -> SparsePauliOp:
        Q = self.build_qubo()
        h, J, offset = self.qubo_to_ising(Q)
        n = self.n_qubits

        pauli_list = []
        coeffs = []

        for i in range(n):
            if abs(h[i]) > 1e-10:
                label = ["I"] * n
                label[i] = "Z"

                # 🔴 CRÍTICO: Qiskit usa ordem invertida
                pauli_list.append("".join(reversed(label)))
                coeffs.append(h[i])

        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-10:
                    label = ["I"] * n
                    label[i] = "Z"
                    label[j] = "Z"

                    pauli_list.append("".join(reversed(label)))
                    coeffs.append(J[i, j])

        return SparsePauliOp(pauli_list, coeffs=coeffs)

    def decode_bitstring(self, bitstring: str):
        n = self.n

        bits = [int(b) for b in bitstring]  # SEM reverse

        x = np.array(bits).reshape(n, n)

        if not (np.all(x.sum(axis=1) == 1) and np.all(x.sum(axis=0) == 1)):
            return None

        route = [0] * n
        for i in range(n):
            for t in range(n):
                if x[i, t] == 1:
                    route[t] = i

        return route