import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from tsp.instance import TSPInstance
from tsp.qubo import TSPtoQUBO

class NoiseAnalyzer:
    """Analyzes and visualizes the effects of different noise models."""

    def __init__(self, tsp: TSPInstance, qubo:TSPtoQUBO, results: dict, optimal_cost: float,
                 noise_meta: dict, noise_params: dict, n_starts : int, n_shots: int, output_dir: str = "figures"):
        self.tsp = tsp
        self.qubo = qubo
        self.results = results
        self.optimal_cost = optimal_cost
        self.noise_meta = noise_meta
        self.noise_params = noise_params
        self.n_starts = n_starts
        self.n_shots = n_shots
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_fig(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"[Saved] {path}")

    def run_full_analysis(self):
        """
        Executes the complete analysis pipeline:
        Creates the final summary table.
        Generates all plots (convergence, quality, distributions).
        """
        
        self.print_summary()
        self.plot_convergence()
        self.plot_solution_quality()
        self.plot_count_distributions()
        self.plot_summary_table()
        
    def print_summary(self):
        print("\n" + "═" * 65)
        print(" SUMMARY — EFFECTS OF NOISE MODELS ON TSP-QAOA")
        print("═" * 65)
        print(f" Optimal (Brute Force): {self.optimal_cost:.4f}\n")

        noise_params = self.noise_params

        for key, res in self.results.items():
            best      = res.get("best")
            valid_pct = res.get("valid_ratio", 0)

            if key == 'ideal':
                params_str = "No noise"
            elif key == 'depolarizing':
                params_str = f"p1q={noise_params['p1q']}, p2q={noise_params['p2q']}"
            elif key == 'bit_flip':
                params_str = f"p_bf={noise_params['p_bf']}"
            elif key == 'phase_flip':
                params_str = f"p_pf={noise_params['p_pf']}"
            else:
                params_str = f"T1={noise_params['T1']}, T2={noise_params['T2']} \n t1q={noise_params['t1q']}, t2q={noise_params['t2q']}"

            print(f" [{key}] ({params_str})")

            if best:
                ratio = best["cost"] / self.optimal_cost
                print(f"   Best cost    : {best['cost']:.4f}  (approx ratio = {ratio:.3f})")
                print(f"   Best route   : {best['route']}")
                print(f"   % Valid sols : {valid_pct * 100:.1f}%")
                print(f"   Final energy : {res['final_energy']:.4f}")

        print("═" * 65)

    def plot_convergence(self):
        n_trials = self.n_starts
        fig, axes = plt.subplots(
            n_trials,
            1,
            figsize=(7, 3 * n_trials),
            sharex=True,
            sharey=True
        )

        if n_trials == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            for key, res in self.results.items():
                trial_histories = res.get("trial_histories", [])
                if i < len(trial_histories):
                    history = trial_histories[i]["history"]
                    ax.plot(history, label=f"{key} (trial {i})")

            ax.set_ylabel(r"$\langle H_C \rangle$")
            ax.set_title(f"Trial {i}")

            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True)

        axes[-1].set_xlabel("Iteration")
        fig.suptitle("QAOA Energy Convergence by Trial and Model")
        fig.tight_layout(rect=[0, 0, 1, 1])
        self._save_fig(fig, "convergence.png")
        
    def plot_solution_quality(self):
        keys      = list(self.results.keys())
        costs     = [
            self.results[k]["best"]["cost"]
            if self.results[k].get("best") else float("nan")
            for k in keys
        ]
        valid_pcts = [self.results[k].get("valid_ratio", 0) * 100 for k in keys]

        x = np.arange(len(keys))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        bars1 = ax1.bar(x, costs)
        ax1.set_xticks(x)
        ax1.set_xticklabels(keys)
        ax1.set_ylabel("Best Route Cost by Model")
        ax1.set_title("Solution Quality by Model")
        for bar, cost in zip(bars1, costs):
            if not np.isnan(cost):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.0005,
                         f"{cost:.4f}", ha="center", va="bottom", fontsize=8)
                ax1.grid(True)

        bars2 = ax2.bar(x, valid_pcts)
        ax2.set_xticks(x)
        ax2.set_xticklabels(keys)
        ax2.set_ylabel("Valid Samples (%)")
        ax2.set_title("Valid Solution Rate by Model")
        for bar, pct in zip(bars2, valid_pcts):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.0005,
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
            ax2.grid(True)

        fig.tight_layout()
        self._save_fig(fig, "solution_quality.png")

    def plot_count_distributions(self):
        keys     = list(self.results.keys())
        fig, axes = plt.subplots(2, 3,
                                 figsize=(4.5 * 3, 4.5 * 2),
                                 sharex=True)
                
        axes = axes.flatten()

        for ax, key in zip(axes, keys):
            counts = self.results[key]["counts"]
            top    = sorted(counts.items(), key=lambda x: -x[1])[:10]
            values    = [c for _, c in top]

            colors = []
            bs_labels = []
            for bs, _ in top:
                route = self.qubo.decode_bitstring(bs)
                if route is not None:
                    bs_labels.append(f"{bs}\n{route}")
                    colors.append("tab:blue")   # valid bitstring
                else:
                    bs_labels.append(f"{bs}\n—")
                    colors.append("tab:gray")   # invalid bitstring

            ax.barh(range(len(bs_labels)), values, color=colors)
            ax.set_yticks(range(len(bs_labels)))
            ax.set_yticklabels(bs_labels)
            ax.set_xlabel("Counts")
            ax.set_title(key)
            ax.invert_yaxis()
            ax.grid(True)

        legend_ax = axes[5]
        legend_ax.axis("off")
        legend_ax.legend(
            handles=[
                Patch(facecolor="tab:blue", label="Valid solution"),
                Patch(facecolor="tab:gray", label="Invalid solution"),
            ],
            loc="center",
            frameon=False,
            fontsize=12,
        )

        fig.suptitle(f"Top-10 Final State Distributions ({4 * self.n_shots} shots) by Model")
        fig.tight_layout()
        self._save_fig(fig, "counts.png")

    def plot_summary_table(self):
        rows = []
        for key, res in self.results.items():
            best  = res.get("best")
            valid = res.get("valid_ratio", 0)

            best_cost    = f"{best['cost']:.4f}" if best else "—"
            approx_ratio = (f"{best['cost'] / self.optimal_cost:.3f}"
                            if best else "—")
            valid_str    = f"{valid * 100:.1f}%"
            n_iter       = str(res.get("n_iter", "—"))

            noise_params = self.noise_params

            if key == 'ideal':
                params_str = "No noise"
            elif key == 'depolarizing':
                params_str = f"p1q={noise_params['p1q']}, \n p2q={noise_params['p2q']}"
            elif key == 'bit_flip':
                params_str = f"p_bf={noise_params['p_bf']}"
            elif key == 'phase_flip':
                params_str = f"p_pf={noise_params['p_pf']}"
            else:
                params_str = f"T1={noise_params['T1']}, \n T2={noise_params['T2']} \n t1q={noise_params['t1q']}, \n t2q={noise_params['t2q']}"

            desc         = params_str

            rows.append([key, desc, best_cost,
                         approx_ratio, valid_str, n_iter])

        col_labels = ["Model", "Parameters", "Best Cost",
                      "Approx. Ratio", "Valid (%)", "Iterations"]

        n_rows  = len(rows)
        fig_h   = max(0.55 * (n_rows + 2), 2.5)
        fig, ax = plt.subplots(figsize=(7,4))
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

        #header_bg  = "#1a1a2e"
        #row_colors = ["#ffffff", "#f4f4f8"]

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.4)
            #cell.set_edgecolor("#aaaaaa")
            if row == 0:
                #cell.set_facecolor(header_bg)
                cell.set_text_props(color="#1a1a1a", fontweight="bold",
                                    fontsize=9.5)
            else:
                #cell.set_facecolor(row_colors[(row - 1) % 2])
                cell.set_text_props(color="#1a1a1a", fontsize=9.5)
                # Parameters column — muted italic
                if col == 1:
                    cell.set_text_props(color="#1a1a1a", fontsize=8)

        ax.set_title(
            "Comparative Summary — Noise Model Effects on TSP-QAOA",
            fontsize=11,
        )
        fig.tight_layout()
        self._save_fig(fig, "summary_table.png")


    def plot_noise_sweep(self,
                         sweep_results,
                         x_key,
                         x_label,
                         title_prefix,
                         output_name="noise_sweep.png"):
        """Plot a generic noise sweep for best cost, valid rate, and elapsed time."""
        x_vals = [entry[x_key] for entry in sweep_results]
        costs = [entry["best_cost"] for entry in sweep_results]
        valids = [entry["valid_ratio"] for entry in sweep_results]
        times = [entry["elapsed"] for entry in sweep_results]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        self._plot_sweep_axis(
            axes[0],
            x_vals,
            {"Best Cost": costs},
            x_label,
            "Best Cost",
            f"{title_prefix} — Best Cost vs Noise Probability",
        )
        self._plot_sweep_axis(
            axes[1],
            x_vals,
            {"Valid Rate (%)": valids},
            x_label,
            "Valid Rate (%)",
            f"{title_prefix} — Valid Rate vs Noise Probability",
        )
        self._plot_sweep_axis(
            axes[2],
            x_vals,
            {"Elapsed Time (s)": times},
            x_label,
            "Elapsed Time (s)",
            f"{title_prefix} — Elapsed Time vs Noise Probability",
        )

        fig.tight_layout()
        self._save_fig(fig, output_name)

    def plot_noise_sweep_comparison(self,
                                    all_sweep_results,
                                    x_key,
                                    x_label,
                                    title_prefix,
                                    output_name="noise_sweep_comparison.png"):
        """Plot a comparison of noise sweeps for multiple models using grouped bar charts."""
        x_vals = None
        costs = {}
        valids = {}
        times = {}

        for model, sweep in all_sweep_results.items():
            if x_vals is None:
                x_vals = [entry[x_key] for entry in sweep]

            costs[model] = [entry["best_cost"] for entry in sweep]
            valids[model] = [entry["valid_ratio"] for entry in sweep]
            times[model] = [entry["elapsed"] for entry in sweep]

        models = list(all_sweep_results.keys())
        n_models = len(models)
        x = np.arange(len(x_vals))
        width = 0.75 / n_models

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        self._plot_grouped_bar_axis(
            axes[0],
            x,
            x_vals,
            models,
            costs,
            width,
            x_label,
            "Best Cost",
            f"{title_prefix} — Best Cost vs Noise Probability",
        )
        self._plot_grouped_bar_axis(
            axes[1],
            x,
            x_vals,
            models,
            valids,
            width,
            x_label,
            "Valid Rate (%)",
            f"{title_prefix} — Valid Rate vs Noise Probability",
        )
        self._plot_grouped_bar_axis(
            axes[2],
            x,
            x_vals,
            models,
            times,
            width,
            x_label,
            "Elapsed Time (s)",
            f"{title_prefix} — Elapsed Time vs Noise Probability",
        )

        fig.tight_layout()
        self._save_fig(fig, output_name)

    def _plot_grouped_bar_axis(self, ax, x_positions, x_labels, models, series, width,
                               xlabel, ylabel, title):
        for i, model in enumerate(models):
            offsets = x_positions + (i - len(models) / 2 + 0.5) * width
            values = series[model]
            bars = ax.bar(offsets, values, width, label=model)
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{val:.1f}" for val in x_labels])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.legend()

    def _plot_sweep_axis(self, ax, x_values, series, xlabel, ylabel, title):
        for label, values in series.items():
            ax.plot(x_values, values, marker="o", linestyle="-", label=label)
            for x, y in zip(x_values, values):
                ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        if len(series) > 1:
            ax.legend()
