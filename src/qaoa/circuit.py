import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from tsp_qaoa.qaoa.circuit import build_qaoa_circuit


class QAOASolver:
    def __init__(self, qubo, p: int = 2, shots: int = 2048):
        self.qubo = qubo
        self.p = p
        self.shots = shots
        self.cost_op = qubo.build_cost_operator()
        self.n_qubits = qubo.n_qubits

    # ------------------------------------------------------------------
    # Expectation value
    # ------------------------------------------------------------------
    def _expectation_value(self, params, simulator):
        qc = build_qaoa_circuit(
            cost_op=self.cost_op,
            params=params,
            p=self.p,
            n_qubits=self.n_qubits
        )

        qc_t = transpile(qc, simulator)
        result = simulator.run(qc_t, shots=self.shots).result()
        counts = result.get_counts()

        energy = 0.0
        total = sum(counts.values())

        for bitstring, count in counts.items():
            bits = np.array([int(b) for b in reversed(bitstring)])
            z = 1 - 2 * bits

            ev = 0.0
            for term, coeff in zip(self.cost_op.paulis, self.cost_op.coeffs):
                term_str = str(term)
                val = float(np.real(coeff))

                for q, p_char in enumerate(reversed(term_str)):
                    if p_char == "Z":
                        val *= z[q]

                ev += val

            energy += (count / total) * ev

        return energy

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, noise_model=None, verbose=True):

        simulator = (
            AerSimulator(noise_model=noise_model)
            if noise_model else AerSimulator()
        )

        best_result = None
        best_fun = float("inf")

        N_STARTS = 3

        for trial in range(N_STARTS):

            np.random.seed(trial * 7)
            init_params = np.random.uniform(0, np.pi, size=2 * self.p)

            def objective(params):
                return self._expectation_value(params, simulator)

            res = minimize(
                objective,
                init_params,
                method="COBYLA",
                options={"maxiter": 200}
            )

            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res

        # ---------------- FINAL SAMPLING ----------------
        qc = build_qaoa_circuit(
            cost_op=self.cost_op,
            params=best_result.x,
            p=self.p,
            n_qubits=self.n_qubits
        )

        qc_t = transpile(qc, simulator)
        result = simulator.run(qc_t, shots=self.shots * 4).result()
        counts = result.get_counts()

        # Decode solutions
        solutions = []
        total = sum(counts.values())

        for bitstring, count in counts.items():
            route = self.qubo.decode_bitstring(bitstring)
            if route is not None:
                cost = self.qubo.tsp.route_cost(route)
                solutions.append({
                    "route": route,
                    "cost": cost,
                    "frequency": count / total
                })

        best = min(solutions, key=lambda x: x["cost"]) if solutions else None

        return {
            "optimal_params": best_result.x,
            "final_energy": best_result.fun,
            "counts": counts,
            "solutions": solutions,
            "best": best
        }