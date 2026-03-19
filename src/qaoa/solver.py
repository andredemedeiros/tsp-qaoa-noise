import numpy as np
from tsp.qubo import TSPtoQUBO

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
    ReadoutError,
)

from qiskit import transpile
from scipy.optimize import minimize

class QAOASolver:
    """
    Solves TSP-QUBO with QAOA using Qiskit.
    Supports execution with and without noise models.
    """

    def __init__(self, qubo: TSPtoQUBO, p: int = 2, shots: int = 2048):
        self.qubo = qubo
        self.p = p          # QAOA depth
        self.shots = shots
        self.cost_op = qubo.build_cost_operator()
        self.n_qubits = qubo.n_qubits

    def _expectation_value(self, params: np.ndarray,
                           simulator: AerSimulator) -> float:
        """Calculates ⟨Hc⟩ via sampling."""
        qc = self._build_qaoa_circuit(params)
        qc_t = transpile(qc, simulator)
        result = simulator.run(qc_t, shots=self.shots).result()
        counts = result.get_counts()

        energy = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            # Convert bitstring to Z-spin vector z ∈ {-1, +1}
            bits = np.array([int(b) for b in reversed(bitstring)])
            z = 1 - 2 * bits  # x=0 → z=+1, x=1 → z=-1
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

    def solve(self, noise_model: NoiseModel = None,
              verbose: bool = True) -> dict:
        """
        Runs QAOA optimization.
        Returns a dictionary with results.
        """
        if noise_model:
            simulator = AerSimulator(noise_model=noise_model)
        else:
            simulator = AerSimulator()  # ideal simulator

        if verbose:
            noise_label = "WITH noise" if noise_model else "WITHOUT noise"
            print(f"\n{chr(8212)*50}")
            print(f"  QAOA p={self.p} | {noise_label} | shots={self.shots}")
            print(f"{chr(8212)*50}")

        # Multi-start: tests N initial points; keeps the best result
        N_STARTS = 3
        best_result = None
        best_fun = float("inf")
        all_history = []

        for trial in range(N_STARTS):
            np.random.seed(trial * 7)
            init_params = np.random.uniform(0, np.pi, size=2 * self.p)
            trial_history = []
            iter_count = [0]

            def objective(params, _th=trial_history, _ic=iter_count):
                val = self._expectation_value(params, simulator)
                _th.append(val)
                _ic[0] += 1
                if verbose and _ic[0] % 15 == 0:
                    print(f"  [start {trial+1}/{N_STARTS}] "
                          f"Iter {_ic[0]:3d} | ⟨Hc⟩ = {val:.4f}")
                return val

            res = minimize(objective, init_params,
                           method="COBYLA",
                           options={"maxiter": 200, "rhobeg": 0.5})
            all_history.extend(trial_history)
            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res

        # Final sampling with optimal parameters (more shots for statistics)
        final_qc = self._build_qaoa_circuit(best_result.x)
        final_qc_t = transpile(final_qc, simulator)
        final_result = simulator.run(final_qc_t,
                                     shots=self.shots * 8).result()
        counts = final_result.get_counts()
        total_shots = sum(counts.values())

        # Decodes top-50 bitstrings (valid and invalid)
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        solutions = []
        invalid_best = None

        for bitstring, count in sorted_counts[:50]:
            route = self.qubo.decode_bitstring(bitstring)
            freq = count / total_shots
            if route is not None:
                cost = self.qubo.tsp.route_cost(route)
                solutions.append({
                    "bitstring": bitstring,
                    "route": route,
                    "cost": cost,
                    "frequency": freq,
                })
            elif invalid_best is None:
                invalid_best = {
                    "bitstring": bitstring,
                    "route": None,
                    "cost": float("nan"),
                    "frequency": freq,
                    "note": "invalid (constraints violated)",
                }

        best = min(solutions, key=lambda s: s["cost"]) if solutions else None
        valid_freq_total = sum(s["frequency"] for s in solutions)

        print(f"  → Valid: {len(solutions)} types | "
              f"{valid_freq_total*100:.1f}% of samples")
        if best:
            print(f"  → Best cost: {best['cost']:.4f} | Route: {best['route']}")
        else:
            print("  → No valid solution. Noise too high for this p/A.")

        return {
            "optimal_params": best_result.x,
            "final_energy": best_result.fun,
            "history": all_history,
            "counts": counts,
            "solutions": solutions,
            "invalid_best": invalid_best,
            "best": best,
            "valid_freq": valid_freq_total,
            "n_iter": len(all_history),
        }