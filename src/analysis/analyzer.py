import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tsp.instance import TSPInstance

class NoiseAnalyzer:
    """Analyzes and visualizes the effects of different noise models."""

    def __init__(self, tsp: TSPInstance, results: dict, optimal_cost: float,
                 noise_meta: dict, noise_params: dict, output_dir: str = "figures"):
        self.tsp = tsp
        self.results = results
        self.optimal_cost = optimal_cost
        self.noise_meta = noise_meta
        self.noise_params = noise_params
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Meta handler (robust: dict / object / None)
    # ------------------------------------------------------------------
    def _get_meta(self, key):
        meta = self.noise_meta.get(key, None)
        default = {"label": key, "desc": "", "color": None}

        if meta is None:
            return default
        if isinstance(meta, dict):
            return {
                "label": meta.get("label", key),
                "desc":  meta.get("desc", ""),
                "color": meta.get("color", None),
            }
        # fallback: object (e.g., bare NoiseModel)
        return {
            "label": getattr(meta, "label", key),
            "desc":  getattr(meta, "description", ""),
            "color": getattr(meta, "color", None),
        }

    def _color(self, key):
        return self._color_map[key]

    # ------------------------------------------------------------------
    # Save figure helper
    # ------------------------------------------------------------------
    def _save_fig(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"[Saved] {path}")

    def run_full_analysis(self):
        """
        Executes the complete analysis pipeline:
        1. Prints text summary to console.
        2. Generates all plots (convergence, quality, distributions).
        3. Creates the final summary table image.
        """
        print(f"\n[Starting Full Analysis] Output directory: {self.output_dir}")
        
        self.print_summary()
        self.plot_convergence()
        self.plot_solution_quality()
        self.plot_count_distributions()
        self.plot_summary_table()
        
        print(f"\n[Analysis Complete] All figures saved to '{self.output_dir}'.")
        
    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    def print_summary(self):
        print("\n" + "═" * 65)
        print(" SUMMARY — EFFECTS OF NOISE MODELS ON TSP-QAOA")
        print("═" * 65)
        print(f" Optimal (Brute Force): {self.optimal_cost:.4f}\n")

        noise_params = self.noise_params

        for key, res in self.results.items():
            meta = self._get_meta(key)
            best      = res.get("best")
            valid_pct = res.get("valid_ratio", 0)

            if meta['label'] == 'ideal':
                params_str = "No noise"
            elif meta['label'] == 'depolarizing':
                params_str = f"p1q={noise_params['p1q']}, p2q={noise_params['p2q']}"
            elif meta['label'] == 'bit_flip':
                params_str = f"p_bf={noise_params['p_bf']}"
            elif meta['label'] == 'phase_flip':
                params_str = f"p_pf={noise_params['p_pf']}"
            else:
                params_str = f"T1={noise_params['T1']}, T2={noise_params['T2']} \n t1q={noise_params['t1q']}, t2q={noise_params['t2q']}"

            print(f" [{meta['label']}] ({params_str})")

            if best:
                ratio = best["cost"] / self.optimal_cost
                print(f"   Best cost    : {best['cost']:.4f}  (approx ratio = {ratio:.3f})")
                print(f"   Best route   : {best['route']}")
                print(f"   % Valid sols : {valid_pct * 100:.1f}%")
                print(f"   Final energy : {res['final_energy']:.4f}")

        print("═" * 65)

    # ------------------------------------------------------------------
    # 6.1 Energy Convergence
    #     FIX: each model gets its own color AND line style so they are
    #     distinguishable even without color (e.g. printed grayscale).
    # ------------------------------------------------------------------
    def plot_convergence(self):
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False

        for i, (key, res) in enumerate(self.results.items()):
            meta    = self._get_meta(key)
            history = res.get("history")
            if history is None:
                print(f"[Warning] No history for {key}, skipping.")
                continue

            ax.plot(
                history,
                label=meta["label"],
                color=self._color(key),                      # distinct color per model
                linestyle=_LINE_STYLES[i % len(_LINE_STYLES)],  # distinct dash per model
                lw=1.6,
                alpha=0.9,
            )
            plotted = True

        if not plotted:
            print("[Warning] No convergence data available.")
            plt.close(fig)
            return

        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\langle H_C \rangle$")
        ax.set_title("QAOA Energy Convergence by Noise Model")
        ax.legend(loc="upper right")
        fig.tight_layout()
        self._save_fig(fig, "convergence.png")

    # ------------------------------------------------------------------
    # 6.2 Solution Quality
    # ------------------------------------------------------------------
    def plot_solution_quality(self):
        keys      = list(self.results.keys())
        labels    = [self._get_meta(k)["label"] for k in keys]
        colors    = [self._color(k) for k in keys]
        costs     = [
            self.results[k]["best"]["cost"]
            if self.results[k].get("best") else float("nan")
            for k in keys
        ]
        valid_pcts = [self.results[k].get("valid_ratio", 0) * 100 for k in keys]

        x = np.arange(len(keys))
        w = 0.5
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # --- Best cost ---
        bars1 = ax1.bar(x, costs, width=w, color=colors,
                        alpha=0.85, edgecolor="black", linewidth=0.6)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20, ha="right")
        ax1.set_ylabel("Best Route Cost")
        ax1.set_title("Solution Quality")
        ax1.legend()
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        for bar, cost in zip(bars1, costs):
            if not np.isnan(cost):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.0005,
                         f"{cost:.4f}", ha="center", va="bottom", fontsize=8)

        # --- Valid % ---
        bars2 = ax2.bar(x, valid_pcts, width=w, color=colors,
                        alpha=0.85, edgecolor="black", linewidth=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=20, ha="right")
        ax2.set_ylabel("Valid Samples (%)")
        ax2.set_title("Valid Solution Rate")
        ax2.set_ylim(0, max(valid_pcts) * 1.3 + 1)
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
        for bar, pct in zip(bars2, valid_pcts):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        self._save_fig(fig, "solution_quality.png")

    # ------------------------------------------------------------------
    # 6.3 Count Distributions
    # ------------------------------------------------------------------
    def plot_count_distributions(self):
        keys     = list(self.results.keys())
        n_models = len(keys)
        fig, axes = plt.subplots(1, n_models,
                                 figsize=(4.5 * n_models, 4.5),
                                 sharey=False)
        if n_models == 1:
            axes = [axes]

        for ax, key in zip(axes, keys):
            meta   = self._get_meta(key)
            counts = self.results[key]["counts"]
            top    = sorted(counts.items(), key=lambda x: -x[1])[:12]
            bs_labels = [bs[:8] for bs, _ in top]
            values    = [c for _, c in top]

            ax.barh(range(len(bs_labels)), values,
                    color=self._color(key), alpha=0.82,
                    edgecolor="black", linewidth=0.5)
            ax.set_yticks(range(len(bs_labels)))
            ax.set_yticklabels(bs_labels, fontsize=7.5, family="monospace")
            ax.set_xlabel("Counts")
            ax.set_title(f"{meta['label']}\n{meta['desc']}", fontsize=9.5)
            ax.invert_yaxis()

        fig.suptitle("Top-12 Bitstring Frequencies by Noise Model",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        self._save_fig(fig, "counts.png")

    # ------------------------------------------------------------------
    # 6.4 Summary Table — minimal scientific style
    # ------------------------------------------------------------------
    def plot_summary_table(self):
        rows = []
        for key, res in self.results.items():
            meta  = self._get_meta(key)
            best  = res.get("best")
            valid = res.get("valid_ratio", 0)

            best_cost    = f"{best['cost']:.4f}" if best else "—"
            approx_ratio = (f"{best['cost'] / self.optimal_cost:.3f}"
                            if best else "—")
            valid_str    = f"{valid * 100:.1f}%"
            n_iter       = str(res.get("n_iter", "—"))
            desc         = meta["desc"] if meta["desc"] else "—"

            rows.append([meta["label"], desc, best_cost,
                         approx_ratio, valid_str, n_iter])

        col_labels = ["Noise Model", "Parameters", "Best Cost",
                      "Approx. Ratio", "Valid (%)", "Iterations"]

        n_rows  = len(rows)
        fig_h   = max(0.55 * (n_rows + 2), 2.5)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        ax.axis("off")

        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)

        header_bg  = "#1a1a2e"
        row_colors = ["#ffffff", "#f4f4f8"]

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.4)
            cell.set_edgecolor("#aaaaaa")
            if row == 0:
                cell.set_facecolor(header_bg)
                cell.set_text_props(color="white", fontweight="bold",
                                    fontsize=9.5)
            else:
                cell.set_facecolor(row_colors[(row - 1) % 2])
                cell.set_text_props(color="#1a1a1a", fontsize=9.5)
                # Parameters column — muted italic
                if col == 1:
                    cell.set_text_props(color="#555555", fontstyle="italic",
                                        fontsize=9)

        ax.set_title(
            "Comparative Summary — Noise Model Effects on TSP-QAOA",
            fontsize=11, fontweight="bold", pad=10, loc="left",
        )
        fig.tight_layout()
        self._save_fig(fig, "summary_table.png")