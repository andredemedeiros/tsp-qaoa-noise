"""
=============================================================================
TSP implementation via QAOA (Quantum Approximate Optimization Algorithm)
with the addition of noise models and comparative analysis of their effects.

Code structure:
  1. TSP formulation as QUBO
  2. Noise-free QAOA (ideal simulator)
  3. QAOA with noise models (depolarizing, bit-flip, phase-flip, thermal)
  4. Analysis and visualization of noise effects
=============================================================================
"""

# -----------------------------------------------------------------------------
# 1. IMPORTS
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import permutations
import warnings
warnings.filterwarnings("ignore")

# Qiskit core
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector

# Qiskit Aer (simulator)
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

# Primitives and optimizers
from qiskit.primitives import Sampler, StatevectorSampler
from qiskit_aer.primitives import SamplerV2 as AerSampler

# Operators and QAOA
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize


# -----------------------------------------------------------------------------
# 2. TSP PROBLEM DEFINITION
# -----------------------------------------------------------------------------

class TSPInstance:
    """Defines a TSP instance with n cities, [0,1]²"""

    def __init__(self, n_cities: int, seed: int = 42):
        self.n = n_cities
        np.random.seed(seed)
        self.coords = np.random.rand(n_cities, 2)
        self.dist = np.array([
            [np.linalg.norm(self.coords[i] - self.coords[j])
             for j in range(n_cities)]
            for i in range(n_cities)
        ])

    def brute_force(self):
        """Solves by brute force for validation (small instances only)"""
        cities = list(range(self.n))
        best_cost = float("inf")
        best_route = None
        for perm in permutations(cities[1:]):
            route = [0] + list(perm)
            cost = self.route_cost(route)
            if cost < best_cost:
                best_cost = cost
                best_route = route[:]
        return best_route, best_cost

    def route_cost(self, route):
        """Calculates the cost of a route."""
        return sum(self.dist[route[i]][route[(i+1) % self.n]]
                   for i in range(self.n))

    def plot(self, routes_dict: dict = None, title: str = "TSP"):
        """Plots the cities and provided routes."""
        fig, axes = plt.subplots(1, max(1, len(routes_dict or {})),
                                 figsize=(5 * max(1, len(routes_dict or {})), 4))
        if routes_dict is None:
            routes_dict = {"Cities": None}
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for ax, (label, route) in zip(axes, routes_dict.items()):
            ax.scatter(self.coords[:, 0], self.coords[:, 1],
                       s=150, c="steelblue", zorder=5)
            for i, (x, y) in enumerate(self.coords):
                ax.annotate(f"C{i}", (x, y), textcoords="offset points",
                            xytext=(6, 6), fontsize=10, fontweight="bold")
            if route is not None:
                r = list(route) + [route[0]]
                xs = [self.coords[c][0] for c in r]
                ys = [self.coords[c][1] for c in r]
                ax.plot(xs, ys, "o-", color="tomato", lw=2)
                cost = self.route_cost(route)
                ax.set_title(f"{label}\nCost: {cost:.4f}", fontsize=11)
            else:
                ax.set_title(label, fontsize=11)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig


# -----------------------------------------------------------------------------
# 3. TSP QUBO FORMULATION
# -----------------------------------------------------------------------------

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
        """Qubit index for x_{city, pos}."""
        return city * self.n + pos

    def build_qubo(self) -> np.ndarray:
        """Builds the QUBO matrix Q such that cost = x^T Q x."""
        n = self.n
        Q = np.zeros((self.n_qubits, self.n_qubits))

        # -- Constraint 1: each city visited exactly once ----------------------
        for i in range(n):
            for t in range(n):
                vi = self.var_index(i, t)
                Q[vi, vi] -= self.A          # linear term (-A * x)
                for s in range(t + 1, n):
                    vj = self.var_index(i, s)
                    Q[vi, vj] += 2 * self.A  # quadratic term

        # -- Constraint 2: each position occupied by exactly one city ----------
        for t in range(n):
            for i in range(n):
                vi = self.var_index(i, t)
                Q[vi, vi] -= self.A
                for j in range(i + 1, n):
                    vj = self.var_index(j, t)
                    Q[vi, vj] += 2 * self.A

        # -- Objective: minimize total distance --------------------------------
        for i in range(n):
            for j in range(n):
                if i != j:
                    for t in range(n):
                        vi = self.var_index(i, t)
                        vj = self.var_index(j, (t + 1) % n)
                        Q[vi, vj] += self.B * self.tsp.dist[i][j]

        return Q

    def qubo_to_ising(self, Q: np.ndarray):
        """
        Converts QUBO -> Ising: x_i = (1 - z_i) / 2, z_i ∈ {-1, +1}.
        Returns (h, J, offset) where H_ising = Σ h_i z_i + Σ J_ij z_i z_j.
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
        """Builds the cost operator Hc as SparsePauliOp."""
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
        Decodes an n²-bit bitstring into a route.
        Returns None if the solution is invalid.
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


# -----------------------------------------------------------------------------
# 4. QAOA SOLVER
# -----------------------------------------------------------------------------

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

    def _build_qaoa_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """
        Builds the QAOA circuit manually for full control.
        params = [γ_1, β_1, γ_2, β_2, ...] with length 2*p.
        """
        n = self.n_qubits
        qc = QuantumCircuit(n)

        # Initial state: uniform superposition
        qc.h(range(n))

        gammas = params[:self.p]
        betas = params[self.p:]

        for layer in range(self.p):
            γ = gammas[layer]
            β = betas[layer]

            # -- Cost operator: e^{-i γ Hc} -------------------------------------
            for term, coeff in zip(self.cost_op.paulis, self.cost_op.coeffs):
                coeff_val = float(np.real(coeff))
                qubits_in_term = [q for q, p in enumerate(str(term)) if p != "I"]
                if not qubits_in_term:
                    continue
                # Apply CNOT ladder + RZ
                if len(qubits_in_term) == 1:
                    qc.rz(2 * γ * coeff_val, qubits_in_term[0])
                else:
                    for k in range(len(qubits_in_term) - 1):
                        qc.cx(qubits_in_term[k], qubits_in_term[k + 1])
                    qc.rz(2 * γ * coeff_val, qubits_in_term[-1])
                    for k in reversed(range(len(qubits_in_term) - 1)):
                        qc.cx(qubits_in_term[k], qubits_in_term[k + 1])

            # -- Mixer operator: e^{-i β Hm} with Hm = Σ X_i --------------------
            qc.rx(2 * β, range(n))

        qc.measure_all()
        return qc

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


# -----------------------------------------------------------------------------
# 5. NOISE MODELS
# -----------------------------------------------------------------------------

def create_noise_models() -> dict:
    """
    Creates four distinct noise models for comparative analysis.

    Returns dict with:
      - depolarizing  : depolarizing noise (common in real hardware)
      - bit_flip       : bit-flip (classical qubit error)
      - phase_flip     : phase-flip (phase decoherence)
      - thermal        : thermal relaxation (T1/T2)
    """
    models = {}

    # -- 1. Depolarizing Noise -------------------------------------------------
    # Applies random error with probability p on 1 and 2 qubit gates.
    # Models generic hardware noise.
    nm_dep = NoiseModel()
    p1q = 0.005   # 0.5% per 1-qubit gate
    p2q = 0.02    # 2.0% per 2-qubit gate
    nm_dep.add_all_qubit_quantum_error(
        depolarizing_error(p1q, 1), ["u1", "u2", "u3", "rx", "ry", "rz", "h"])
    nm_dep.add_all_qubit_quantum_error(
        depolarizing_error(p2q, 2), ["cx", "cz"])
    models["depolarizing"] = {
        "model": nm_dep,
        "label": "Depolarizing",
        "color": "tomato",
        "desc": f"p_1q={p1q}, p_2q={p2q}",
    }

    # -- 2. Bit-Flip Noise -----------------------------------------------------
    # |0⟩ ↔ |1⟩ with probability p. Analogous to classical error.
    nm_bf = NoiseModel()
    p_bf = 0.01
    err_bf_1q = pauli_error([("X", p_bf), ("I", 1 - p_bf)])
    err_bf_2q = err_bf_1q.expand(err_bf_1q)   # independent error on each qubit
    nm_bf.add_all_qubit_quantum_error(
        err_bf_1q, ["u1", "u2", "u3", "rx", "ry", "rz", "h"])
    nm_bf.add_all_qubit_quantum_error(
        err_bf_2q, ["cx", "cz"])
    models["bit_flip"] = {
        "model": nm_bf,
        "label": "Bit-Flip",
        "color": "mediumorchid",
        "desc": f"p_flip={p_bf}",
    }

    # -- 3. Phase-Flip Noise ---------------------------------------------------
    # Applies Z (phase swap) with probability p. Causes decoherence.
    nm_pf = NoiseModel()
    p_pf = 0.01
    err_pf_1q = pauli_error([("Z", p_pf), ("I", 1 - p_pf)])
    err_pf_2q = err_pf_1q.expand(err_pf_1q)
    nm_pf.add_all_qubit_quantum_error(
        err_pf_1q, ["u1", "u2", "u3", "rx", "ry", "rz", "h"])
    nm_pf.add_all_qubit_quantum_error(
        err_pf_2q, ["cx", "cz"])
    models["phase_flip"] = {
        "model": nm_pf,
        "label": "Phase-Flip",
        "color": "darkorange",
        "desc": f"p_flip={p_pf}",
    }

    # -- 4. Thermal Relaxation (T1/T2) -----------------------------------------
    # Simulates real decoherence: T1 = amplitude relaxation time,
    # T2 = phase relaxation time. Typical superconducting hardware values.
    nm_th = NoiseModel()
    T1 = 50e3   # 50 µs
    T2 = 30e3   # 30 µs
    t_gate_1q = 50    # 50 ns per 1-qubit gate
    t_gate_2q = 300   # 300 ns per 2-qubit gate
    err_1q = thermal_relaxation_error(T1, T2, t_gate_1q)
    err_2q = thermal_relaxation_error(T1, T2, t_gate_2q).expand(
              thermal_relaxation_error(T1, T2, t_gate_2q))
    nm_th.add_all_qubit_quantum_error(
        err_1q, ["u1", "u2", "u3", "rx", "ry", "rz", "h"])
    nm_th.add_all_qubit_quantum_error(err_2q, ["cx"])
    models["thermal"] = {
        "model": nm_th,
        "label": "Thermal Relaxation",
        "color": "steelblue",
        "desc": f"T1={T1/1e3:.0f}µs, T2={T2/1e3:.0f}µs",
    }

    return models


# -----------------------------------------------------------------------------
# 6. ANALYSIS AND VISUALIZATION
# -----------------------------------------------------------------------------

class NoiseAnalyzer:
    """Analyzes and visualizes the effects of different noise models."""

    def __init__(self, tsp: TSPInstance, results: dict,
                 optimal_cost: float, noise_meta: dict):
        self.tsp = tsp
        self.results = results          # {label: result_dict}
        self.optimal_cost = optimal_cost
        self.noise_meta = noise_meta    # noise model metadata

    # -- 6.1 Energy Convergence ------------------------------------------------
    def plot_convergence(self):
        fig, ax = plt.subplots(figsize=(10, 5))

        for key, res in self.results.items():
            meta = self.noise_meta.get(key, {"label": key, "color": "gray"})
            ax.plot(res["history"], label=meta["label"],
                    color=meta["color"], lw=1.8, alpha=0.85)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("⟨Hc⟩ (Hamiltonian Expectation Value)", fontsize=12)
        ax.set_title("QAOA Energy Convergence by Noise Model",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # -- 6.2 Solution Quality --------------------------------------------------
    def plot_solution_quality(self):
        labels, costs, valid_pcts = [], [], []

        for key, res in self.results.items():
            meta = self.noise_meta.get(key, {"label": key})
            labels.append(meta["label"])
            best = res.get("best")
            costs.append(best["cost"] if best else float("nan"))

            sols = res.get("solutions", [])
            total_freq = res.get("valid_freq",
                                  sum(s["frequency"] for s in sols))
            valid_pcts.append(total_freq * 100)

        x = np.arange(len(labels))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Best solution cost found
        colors = [self.noise_meta.get(k, {}).get("color", "gray")
                  for k in self.results.keys()]
        bars = ax1.bar(x, costs, color=colors, alpha=0.85, edgecolor="black")
        ax1.axhline(self.optimal_cost, color="green", lw=2,
                    linestyle="--", label=f"Opt BF ({self.optimal_cost:.4f})")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
        ax1.set_ylabel("Best Route Cost Found", fontsize=11)
        ax1.set_title("Solution Quality by Noise Model",
                      fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, cost in zip(bars, costs):
            if not np.isnan(cost):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.001,
                         f"{cost:.4f}", ha="center", va="bottom", fontsize=9)

        # Frequency of valid solutions
        ax2.bar(x, valid_pcts, color=colors, alpha=0.85, edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
        ax2.set_ylabel("% of Samples with Valid Solution", fontsize=11)
        ax2.set_title("Valid Solution Rate by Noise Model",
                      fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, pct in zip(ax2.patches, valid_pcts):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        return fig

    # -- 6.3 Count Distributions -----------------------------------------------
    def plot_count_distributions(self):
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for ax, (key, res) in zip(axes, self.results.items()):
            meta = self.noise_meta.get(key, {"label": key, "color": "gray"})
            counts = res["counts"]
            sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:15]
            labels_c = [bs[:6] + "…" for bs, _ in sorted_counts]
            values = [c for _, c in sorted_counts]

            ax.barh(range(len(labels_c)), values,
                    color=meta["color"], alpha=0.8, edgecolor="black")
            ax.set_yticks(range(len(labels_c)))
            ax.set_yticklabels(labels_c, fontsize=8)
            ax.set_xlabel("Count", fontsize=10)
            ax.set_title(f"{meta['label']}\n({meta.get('desc','')})",
                         fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

        plt.suptitle("Top-15 Bitstrings by Noise Model",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    # -- 6.4 Summary Table -----------------------------------------------------
    def plot_summary_table(self):
        rows = []
        for key, res in self.results.items():
            meta = self.noise_meta.get(key, {"label": key})
            best = res.get("best")
            best_cost = f"{best['cost']:.4f}" if best else "N/A"
            approx_ratio = (best["cost"] / self.optimal_cost
                            if best else float("nan"))
            valid_pct = res.get("valid_freq", 0)
            rows.append([
                meta["label"],
                meta.get("desc", "—"),
                best_cost,
                f"{approx_ratio:.3f}",
                f"{valid_pct*100:.1f}%",
                str(res["n_iter"]),
            ])

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.axis("off")
        col_labels = ["Model", "Parameters", "Best Cost",
                      "Approx Ratio", "% Valid", "Iterations"]
        table = ax.table(cellText=rows, colLabels=col_labels,
                         cellLoc="center", loc="center",
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#ecf0f1")
        ax.set_title("Comparative Summary — Noise Model Effects",
                     fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        return fig

    def print_summary(self):
        """Prints textual summary to the terminal."""
        print("\n" + "═" * 65)
        print("  SUMMARY — EFFECTS OF NOISE MODELS ON TSP-QAOA")
        print("═" * 65)
        print(f"  Optimal (Brute Force): {self.optimal_cost:.4f}\n")

        for key, res in self.results.items():
            meta = self.noise_meta.get(key, {"label": key})
            best = res.get("best")
            print(f"  [{meta['label']}]  ({meta.get('desc', '')})")
            valid_pct = res.get("valid_freq", 0)
            if best:
                ratio = best["cost"] / self.optimal_cost
                print(f"    Best cost     : {best['cost']:.4f}  "
                      f"(approx ratio = {ratio:.3f})")
                print(f"    Best route    : {best['route']}")
                print(f"    % Valid sols  : {valid_pct*100:.1f}%")
                print(f"    Final energy  : {res['final_energy']:.4f}")
            else:
                inv = res.get("invalid_best")
                print(f"    No valid solution found.")
                print(f"    % Valid sols  : {valid_pct*100:.1f}%  "
                      f"(noise breaks QUBO constraints)")
                if inv:
                    print(f"    Most freq bit : {inv['bitstring'][:12]}... "
                          f"({inv['frequency']*100:.1f}%) [{inv['note']}]")
                print(f"    Final energy  : {res['final_energy']:.4f}")
            print()
        print("═" * 65)


# -----------------------------------------------------------------------------
# 7. MAIN — COMPLETE PIPELINE
# -----------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  TSP with QAOA + Noise Analysis — Qiskit")
    print("=" * 65)

    # -- Configurations --------------------------------------------------------
    N_CITIES = 3      # 3 cities -> 9 qubits (manageable in simulation)
    QAOA_P   = 1      # QAOA depth
    SHOTS    = 1024
    SEED     = 42

    # -- Create TSP Instance ---------------------------------------------------
    print(f"\n[1/5] Creating TSP instance with {N_CITIES} cities...")
    tsp = TSPInstance(N_CITIES, seed=SEED)
    optimal_route, optimal_cost = tsp.brute_force()
    print(f"      Optimal route (Brute Force): {optimal_route}")
    print(f"      Optimal cost: {optimal_cost:.4f}")

    # -- Build QUBO Formulation ------------------------------------------------
    print("\n[2/5] Building QUBO formulation / Ising Hamiltonian...")
    qubo = TSPtoQUBO(tsp, A=15.0, B=1.0)
    cost_op = qubo.build_cost_operator()
    print(f"      Number of qubits: {qubo.n_qubits}")
    print(f"      Hamiltonian terms: {len(cost_op)}")

    # -- Create Noise Models ---------------------------------------------------
    print("\n[3/5] Creating noise models...")
    noise_models = create_noise_models()
    for k, v in noise_models.items():
        print(f"      • {v['label']}: {v['desc']}")

    # -- Run QAOA --------------------------------------------------------------
    print("\n[4/5] Running QAOA...")
    solver = QAOASolver(qubo, p=QAOA_P, shots=SHOTS)

    # Prepare results dictionary: ideal + each noise model
    noise_meta = {"ideal": {"label": "Ideal (No Noise)",
                             "color": "seagreen",
                             "desc": "error-free"}}
    noise_meta.update({k: v for k, v in noise_models.items()})

    results = {}

    # Ideal simulation
    results["ideal"] = solver.solve(noise_model=None, verbose=True)

    # With each noise model
    for key, meta in noise_models.items():
        results[key] = solver.solve(noise_model=meta["model"], verbose=True)

    # -- Analysis and Visualization --------------------------------------------
    print("\n[5/5] Generating analysis and figures...")
    analyzer = NoiseAnalyzer(tsp, results, optimal_cost, noise_meta)
    analyzer.print_summary()

    # Figures
    fig1 = tsp.plot(
        routes_dict={
            "Opt (BF)": optimal_route,
            **{noise_meta[k]["label"]: (results[k]["best"]["route"]
                                        if results[k]["best"] else None)
               for k in results}
        },
        title="TSP — Routes Found by Noise Model"
    )

    fig2 = analyzer.plot_convergence()
    fig3 = analyzer.plot_solution_quality()
    fig4 = analyzer.plot_count_distributions()
    fig5 = analyzer.plot_summary_table()

    # Save figures
    fig1.savefig("tsp_routes.png", dpi=120, bbox_inches="tight")
    fig2.savefig("convergence.png", dpi=120, bbox_inches="tight")
    fig3.savefig("solution_quality.png", dpi=120, bbox_inches="tight")
    fig4.savefig("count_distributions.png", dpi=120, bbox_inches="tight")
    fig5.savefig("summary_table.png", dpi=120, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()