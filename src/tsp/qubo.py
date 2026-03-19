import numpy as np
from tsp.instance import TSPInstance
from qiskit.quantum_info import SparsePauliOp

class TSPtoQUBO:
    """
    Converts TSP to a QUBO problem and then into an Ising operator (Hamiltonian).

    Binary variable: x_{i,t} = 1 if city i is visited at position t.

    QUBO Objective Function:
        H = A * H_constraint + B * H_distance
    where:
        H_constraint penalizes invalid solutions (each city visited once,
                     each position occupied once).
        H_distance   is the route cost.
    """

    def __init__(self, tsp: TSPInstance, A: float = 5.0, B: float = 1.0):
        self.tsp = tsp
        self.n = tsp.n
        self.A = A
        self.B = B
        self.n_qubits = self.n ** 2

    def var_index(self, city: int, pos: int) -> int:
        """Qubit index for x_{city, pos}"""
        return city * self.n + pos

    def build_qubo(self) -> np.ndarray:
        """Builds the QUBO matrix Q rigorously using exact expansion of constraints."""
        n = self.n
        Q = np.zeros((self.n_qubits, self.n_qubits))

        # -- Constraint 1: each city visited exactly once ----------------------
        # H_city = (Σ_t x_{i,t} - 1)^2
        for i in range(n):
            # Quadratic terms
            for t in range(n):
                vi = self.var_index(i, t)
                for s in range(t + 1, n):
                    vj = self.var_index(i, s)
                    Q[vi, vj] += 2 * self.A  # cross term

            # Linear terms
            for t in range(n):
                vi = self.var_index(i, t)
                Q[vi, vi] += -1 * 2 * self.A  # -2*A from expansion

        # -- Constraint 2: each position occupied by exactly one city ----------
        # H_pos = (Σ_i x_{i,t} - 1)^2
        for t in range(n):
            # Quadratic terms
            for i in range(n):
                vi = self.var_index(i, t)
                for j in range(i + 1, n):
                    vj = self.var_index(j, t)
                    Q[vi, vj] += 2 * self.A

            # Linear terms
            for i in range(n):
                vi = self.var_index(i, t)
                Q[vi, vi] += -2 * self.A

        # -- Objective: minimize total distance --------------------------------
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for t in range(n):
                    vi = self.var_index(i, t)
                    vj = self.var_index(j, (t + 1) % n)
                    Q[vi, vj] += self.B * self.tsp.dist[i][j]
                    Q[vj, vi] += self.B * self.tsp.dist[i][j]  # garante simetria

        return Q

    def qubo_to_ising(self, Q: np.ndarray):
        """
        Converts QUBO -> Ising: x_i = (1 - z_i) / 2, z_i ∈ {-1, +1}
        Returns (h, J, offset) where H_ising = Σ h_i z_i + Σ J_ij z_i z_j
        """
        n = self.n_qubits
        h = np.zeros(n)
        J = np.zeros((n, n))

        # Diagonal -> local field h
        # Off-diagonal -> coupling J
        offset = 0.0
        for i in range(n):
            h[i] -= Q[i, i] / 2
            offset += Q[i, i] / 4
            for j in range(i + 1, n):
                Qij = Q[i, j] + Q[j, i]
                J[i, j] += Qij / 4
                h[i] -= Qij / 4
                h[j] -= Qij / 4
                offset += Qij / 4

        return h, J, offset

    def build_cost_operator(self) -> SparsePauliOp:
        """Builds the cost operator Hc as SparsePauliOp"""
        Q = self.build_qubo()
        h, J, offset = self.qubo_to_ising(Q)
        n = self.n_qubits

        pauli_list = []
        coeffs = []

        # Field terms (Z_i)
        for i in range(n):
            if abs(h[i]) > 1e-10:
                label = ["I"] * n
                label[i] = "Z"
                pauli_list.append("".join(reversed(label)))
                coeffs.append(h[i])

        # Coupling terms (Z_i Z_j)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-10:
                    label = ["I"] * n
                    label[i] = "Z"
                    label[j] = "Z"
                    pauli_list.append("".join(reversed(label)))
                    coeffs.append(J[i, j])

        if not pauli_list:
            pauli_list = ["I" * n]
            coeffs = [0.0]

        return SparsePauliOp(pauli_list, coeffs=coeffs)

    def decode_bitstring(self, bitstring: str):
        """
        Decodes an n²-bit bitstring into a route
        Returns None if the solution is invalid
        """
        n = self.n
        bits = [int(b) for b in bitstring]
        x = np.array(bits).reshape(n, n)

        # Check validity: each row and column sums to 1
        if not (np.all(x.sum(axis=1) == 1) and np.all(x.sum(axis=0) == 1)):
            return None

        route = [0] * n
        for i in range(n):
            for t in range(n):
                if x[i, t] == 1:
                    route[t] = i
        return route
