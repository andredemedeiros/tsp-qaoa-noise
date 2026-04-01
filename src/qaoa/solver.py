# solver_complete.py
import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from qaoa.circuit import build_qaoa_circuit

class QAOASolver:
    """
    QAOA solver for TSP (QUBO formulation), with noise and full metrics.
    """

    def __init__(self, qubo, p: int = 2, shots: int = 2048):
        self.qubo = qubo
        self.p = p
        self.shots = shots
        self.cost_op = qubo.build_cost_operator()
        self.n_qubits = qubo.n_qubits
        self.param_vector = ParameterVector("theta", length=2 * self.p)
        self.parameterized_circuit = build_qaoa_circuit(
            cost_op=self.cost_op,
            params=self.param_vector,
            p=self.p,
            n_qubits=self.n_qubits,
        )

        # Validate operator consistency
        self.check_consistency(self.cost_op, self.n_qubits)

    # ------------------------------------------------------------------
    # Compute expectation value (with optional energy history)
    # ------------------------------------------------------------------
    def _expectation_value(self, params, simulator, history_list=None):
        bound = {self.param_vector[i]: float(params[i]) for i in range(2 * self.p)}
        qc = self.transpiled_circuit.assign_parameters(bound)
        result = simulator.run(qc, shots=self.shots).result()
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
                z_mask = term.z  # array bool size of n_qubits
                for q in np.where(z_mask)[0]:
                    val *= z[q]
                ev += val

            energy += (count / total) * ev

        if history_list is not None:
            history_list.append(energy)

        return energy

    # ------------------------------------------------------------------
    # Solve QAOA problem
    # ------------------------------------------------------------------
    def solve(self, noise_model=None, verbose=True, n_starts=3):
        simulator = AerSimulator(noise_model=noise_model, seed_simulator=42)
        self.transpiled_circuit = transpile(self.parameterized_circuit, simulator)

        best_result = None
        best_fun = float("inf")
        all_history = []
        trial_histories = []

        for trial in range(n_starts):
            np.random.seed(trial * 7)
            init_params = np.random.uniform(0, np.pi, size=2 * self.p)
            trial_history = []

            def objective(params):
                return self._expectation_value(params, simulator, history_list=trial_history)

            res = minimize(
                objective,
                init_params,
                method="COBYLA",
                options={"maxiter": 200}
            )

            trial_histories.append({
                "trial": trial,
                "init_params": init_params.copy(),
                "history": trial_history,
                "final_energy": res.fun
            })


            all_history.extend(trial_history)

            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res

            if verbose:
                print(f"[Trial {trial}] Energy = {res.fun:.4f}, iterations = {len(trial_history)}")

        # ---------------- FINAL SAMPLING ----------------
        bound = {self.param_vector[i]: float(best_result.x[i]) for i in range(2 * self.p)}
        qc = self.transpiled_circuit.assign_parameters(bound)
        result = simulator.run(qc, shots=self.shots * 4).result()
        counts = result.get_counts()

        # ---------------- VALID SOLUTIONS ----------------
        solutions = []
        total = sum(counts.values())
        valid_counts = 0

        for bitstring, count in counts.items():
            route = self.qubo.decode_bitstring(bitstring)
            if route is not None:
                valid_counts += count
                cost = self.qubo.tsp.route_cost(route)
                solutions.append({
                    "route": route,
                    "cost": cost,
                    "frequency_raw": count / total
                })

        # Normalize frequencies only for valid solutions
        for sol in solutions:
            sol["frequency"] = sol["frequency_raw"] * total / valid_counts if valid_counts > 0 else 0

        valid_ratio = valid_counts / total if total > 0 else 0
        best = min(solutions, key=lambda x: x["cost"]) if solutions else None

        return {
            "optimal_params": best_result.x,
            "final_energy": best_result.fun,
            "history": all_history,
            "trial_histories": trial_histories,
            "counts": counts,
            "solutions": solutions,
            "best": best,
            "valid_ratio": valid_ratio,
            "n_iter": len(all_history)
        }

    # ------------------------------------------------------------------
    # Consistency check for cost operator
    # ------------------------------------------------------------------
    def check_consistency(self, cost_op, n_qubits):
        for term, coeff in zip(cost_op.paulis, cost_op.coeffs):
            assert len(str(term)) == n_qubits, f"Comprimento inconsistente: {term}"
            assert abs(np.imag(coeff)) < 1e-10, f"Coeficiente complexo inesperado: {coeff}"