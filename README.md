# TSP-QAOA-Noise

Implementation of the Traveling Salesman Problem (TSP) solved via the Quantum Approximate Optimization Algorithm (QAOA), with a comparative analysis of different quantum noise models.

This project investigates how realistic noise affects the performance of QAOA when applied to combinatorial optimization problems.


## Overview

The workflow implemented in this repository follows:
**TSP instance** → **QUBO formulation** → **QAOA optimization** → **Noise simulation** → **Analysis**

**Main goals:**
* Solve TSP using QAOA.
* Introduce realistic quantum noise models.
* Compare performance degradation under noise.
* Provide a modular and reproducible scientific pipeline.


## Theoretical Background

### 🔹 Traveling Salesman Problem (TSP)
The Traveling Salesman Problem consists of finding the shortest route visiting all cities exactly once and returning to the origin.

### 🔹 QUBO Formulation
The problem is mapped into a **Quadratic Unconstrained Binary Optimization** model, suitable for quantum optimization algorithms. 


### 🔹 QAOA
The **Quantum Approximate Optimization Algorithm** is a hybrid quantum-classical algorithm designed to solve combinatorial optimization problems. It uses a parameterized ansatz:
$$|\psi(\gamma, \beta)\rangle = e^{-i\beta_p H_B} e^{-i\gamma_p H_C} \dots e^{-i\beta_1 H_B} e^{-i\gamma_1 H_C} |+\rangle^{\otimes n}$$

### 🔹 Noise Models
We simulate realistic quantum hardware effects such as:
* **Depolarizing noise:** General decoherence.
* **Bit-flip errors:** Flips $|0\rangle$ to $|1\rangle$ and vice-versa.
* **Phase-flip errors:** Alters the phase of the qubit.
* **Thermal relaxation ($T_1/T_2$):** Models energy loss and dephasing over time.


## Project Structure

```text
tsp-qaoa-noise/
│
├── src/
│   ├── tsp/
│   │   ├── instance.py       # TSP instance generation
│   │   └── qubo.py           # QUBO formulation
│   │
│   ├── qaoa/
│   │   └── solver.py         # QAOA solver
│   │   └── circuit.py        # Circuit creation
│   ├── noise/
│   │   └── models.py         # Noise models
│   │
│   ├── analysis/
│   │   └── analyzer.py       # Results analysis
│   │
│   ├── config.py             # Global parameters
│   └── main.py               # Main execution pipeline
│
├── requirements.txt
└── README.md
```
## Installation

**Clone the repository:**
```bash
git clone https://github.com/andredemedeiros/tsp-qaoa-noise.git
cd tsp-qaoa-noise
```

**Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

##  Usage

**Run the main experiment:**
```bash
python src/main.py
```


##  Configuration

All parameters can be adjusted in `src/config.py`.

**Example setup:**
```python
N_CITIES = 3
SEED = 42
P = 1         # QAOA layers
SHOTS = 1024
```

##  Output

The program generates a summary of the performance under different conditions:
```plaintext
SUMMARY — EFFECTS OF NOISE MODELS ON TSP-QAOA
═══════════════════════════════════════════════
Optimal (Brute Force): 2.0524

[ideal]
  Best cost     : 2.0524  (approx ratio = 1.000)

[depolarizing]
  Best cost     : 2.4150  (approx ratio = 0.850)

[bit_flip]
  Best cost     : 2.1892  (approx ratio = 0.937)
```

**Metrics calculated:**
- Best route found.
- Cost comparison with optimal solution.
- Approximation ratio ($\frac{E_{QAOA}}{E_{optimal}}$).


##  Noise Models Implemented

| Model | Description |
|---|---|
| Depolarizing | Randomizes qubit state completely |
| Bit-flip | Flips qubit state ($X$ error) |
| Phase-flip | Alters phase ($Z$ error) |
| Thermal | Models decoherence ($T_1/T_2$ relaxation) |


## Analysis

The analysis module:
- Compares noisy vs ideal results.
- Computes approximation ratios.
- Evaluates robustness of QAOA.


##  Key Contributions

- Modular implementation of TSP → QUBO → QAOA.
- Integration of multiple noise models for robustness testing.
- Reproducible experimental pipeline for quantum optimization research.
- Clean architecture for scientific use.

##  Limitations

- **Exponential complexity:** TSP brute force is used for validation (limited to small $N$).
- **Scale:** Small instances only (typically $N \le 5$ due to qubit requirements).
- **Environment:** Simulator-based (not real quantum hardware).


##  Future Work

- Scaling to larger instances using decomposition.
- Integration with real quantum devices (e.g., IBM Quantum).
- Testing advanced optimizers (SPSA, COBYLA) for QAOA.
- Implementation of Error Mitigation techniques (ZNE, PEC).


## References

- Farhi, E. et al. (2014) – *A Quantum Approximate Optimization Algorithm.*
- Lucas, A. (2014) – *Ising formulations of many NP problems.*
- Qiskit Documentation – https://qiskit.org


## Author

**André Medeiros** 

**Mikita Szimonenko**


## License

This project is for academic and research purposes.