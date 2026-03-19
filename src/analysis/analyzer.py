import os
import numpy as np
import matplotlib.pyplot as plt
from tsp.instance import TSPInstance


class NoiseAnalyzer:
    """Analyzes and visualizes the effects of different noise models."""

    def __init__(self, tsp: TSPInstance, results: dict, optimal_cost: float,
                 noise_meta: dict, output_dir: str = "figures"):
        self.tsp = tsp
        self.results = results
        self.optimal_cost = optimal_cost
        self.noise_meta = noise_meta
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Meta handler (robust: dict / object / None)
    # ------------------------------------------------------------------
    def _get_meta(self, key):
        meta = self.noise_meta.get(key, None)
        default = {"label": key, "desc": "", "color": "gray"}

        if meta is None:
            return default
        if isinstance(meta, dict):
            return {
                "label": meta.get("label", key),
                "desc":  meta.get("desc", ""),
                "color": meta.get("color", "gray"),
            }
        # fallback: object (e.g., NoiseModel)
        return {
            "label": getattr(meta, "label", key),
            "desc":  getattr(meta, "description", ""),
            "color": getattr(meta, "color", "gray"),
        }

    # ------------------------------------------------------------------
    # Save figure helper
    # ------------------------------------------------------------------
    def _save_fig(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {path}")

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    def print_summary(self):
        print("\n" + "═" * 65)
        print(" SUMMARY — EFFECTS OF NOISE MODELS ON TSP-QAOA")
        print("═" * 65)
        print(f" Optimal (Brute Force): {self.optimal_cost:.4f}\n")

        for key, res in self.results.items():
            meta = self._get_meta(key)
            print(f" [{meta['label']}] ({meta['desc']})")
            best = res.get("best")
            valid_pct = res.get("valid_ratio", 0)   # FIX: chave correta

            if best:
                ratio = best["cost"] / self.optimal_cost
                print(f"   Best cost    : {best['cost']:.4f} (approx ratio = {ratio:.3f})")
                print(f"   Best route   : {best['route']}")
                print(f"   % Valid sols : {valid_pct * 100:.1f}%")
                print(f"   Final energy : {res['final_energy']:.4f}")
            else:
                inv = res.get("invalid_best")
                print(f"   No valid solution found.")
                print(f"   % Valid sols : {valid_pct * 100:.1f}%")
                if inv:
                    print(
                        f"   Most freq bit: {inv['bitstring'][:12]}..."
                        f" ({inv['frequency'] * 100:.1f}%) [{inv['note']}]"
                    )
                print(f"   Final energy : {res['final_energy']:.4f}")
            print()

        print("═" * 65)

    # ------------------------------------------------------------------
    # 6.1 Energy Convergence
    # ------------------------------------------------------------------
    def plot_convergence(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = False

        for key, res in self.results.items():
            meta = self._get_meta(key)
            history = res.get("history", None)
            if history is None:
                print(f"[Warning] No history for {key}, skipping.")
                continue
            ax.plot(history, label=meta["label"], color=meta["color"],
                    lw=1.8, alpha=0.85)
            plotted = True

        if not plotted:
            print("[Warning] No convergence data available.")
            return

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("⟨Hc⟩ (Hamiltonian Expectation Value)", fontsize=12)
        ax.set_title("QAOA Energy Convergence by Noise Model",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_fig(fig, "convergence.png")

    # ------------------------------------------------------------------
    # 6.2 Solution Quality
    # ------------------------------------------------------------------
    def plot_solution_quality(self):
        labels, costs, valid_pcts = [], [], []
        colors = []

        for key, res in self.results.items():
            meta = self._get_meta(key)
            labels.append(meta["label"])
            colors.append(meta["color"])

            best = res.get("best")
            costs.append(best["cost"] if best else float("nan"))

            # FIX: usar valid_ratio (proporção bruta count/total do solver),
            # não somar frequencies normalizadas que sempre somam ~1.0
            valid_pcts.append(res.get("valid_ratio", 0) * 100)

        x = np.arange(len(labels))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # --- Best cost ---
        bars = ax1.bar(x, costs, color=colors, alpha=0.85, edgecolor="black")
        ax1.axhline(self.optimal_cost, color="green", lw=2, linestyle="--",
                    label=f"Opt BF ({self.optimal_cost:.4f})")
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

        # --- Valid % ---
        bars2 = ax2.bar(x, valid_pcts, color=colors, alpha=0.85, edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
        ax2.set_ylabel("% of Samples with Valid Solution", fontsize=11)
        ax2.set_title("Valid Solution Rate by Noise Model",
                      fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, pct in zip(bars2, valid_pcts):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        self._save_fig(fig, "solution_quality.png")

    # ------------------------------------------------------------------
    # 6.3 Count Distributions
    # ------------------------------------------------------------------
    def plot_count_distributions(self):
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for ax, (key, res) in zip(axes, self.results.items()):
            meta = self._get_meta(key)
            counts = res["counts"]
            sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:15]
            labels_c = [bs[:6] + "…" for bs, _ in sorted_counts]
            values   = [c for _, c in sorted_counts]

            ax.barh(range(len(labels_c)), values,
                    color=meta["color"], alpha=0.8, edgecolor="black")
            ax.set_yticks(range(len(labels_c)))
            ax.set_yticklabels(labels_c, fontsize=8)
            ax.set_xlabel("Count", fontsize=10)
            ax.set_title(f"{meta['label']}\n({meta['desc']})",
                         fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

        plt.suptitle("Top-15 Bitstrings by Noise Model",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_fig(fig, "counts.png")

    # ------------------------------------------------------------------
    # 6.4 Summary Table
    # ------------------------------------------------------------------
    def plot_summary_table(self):
        rows = []
        for key, res in self.results.items():
            meta = self._get_meta(key)
            best = res.get("best")
            best_cost = f"{best['cost']:.4f}" if best else "N/A"
            approx_ratio = (
                best["cost"] / self.optimal_cost if best else float("nan")
            )

            # FIX: chave correta é valid_ratio, não valid_freq
            valid_pct = res.get("valid_ratio", 0)

            rows.append([
                meta["label"],
                meta["desc"] if meta["desc"] else "—",
                best_cost,
                f"{approx_ratio:.3f}" if not np.isnan(approx_ratio) else "N/A",
                f"{valid_pct * 100:.1f}%",
                str(res.get("n_iter", "—")),
            ])

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.axis("off")

        col_labels = ["Model", "Parameters", "Best Cost",
                      "Approx Ratio", "% Valid", "Iterations"]

        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]
        )
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
        self._save_fig(fig, "summary_table.png")