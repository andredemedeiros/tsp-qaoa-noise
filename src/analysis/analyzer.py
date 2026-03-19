from tsp.instance import TSPInstance

class NoiseAnalyzer:
    """Analyzes and visualizes the effects of different noise models."""

    def __init__(self, tsp: TSPInstance, results: dict,
                 optimal_cost: float, noise_meta: dict):
        self.tsp = tsp
        self.results = results          # {label: result_dict}
        self.optimal_cost = optimal_cost
        self.noise_meta = noise_meta    # noise model metadata

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