# # -- 6.1 Energy Convergence ------------------------------------------------
# def plot_convergence(self):
#     fig, ax = plt.subplots(figsize=(10, 5))

#     for key, res in self.results.items():
#         meta = self.noise_meta.get(key, {"label": key, "color": "gray"})
#         ax.plot(res["history"], label=meta["label"],
#                 color=meta["color"], lw=1.8, alpha=0.85)

#     ax.set_xlabel("Iteration", fontsize=12)
#     ax.set_ylabel("⟨Hc⟩ (Hamiltonian Expectation Value)", fontsize=12)
#     ax.set_title("QAOA Energy Convergence by Noise Model",
#                     fontsize=13, fontweight="bold")
#     ax.legend(fontsize=10)
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     return fig

# # -- 6.2 Solution Quality --------------------------------------------------
# def plot_solution_quality(self):
#     labels, costs, valid_pcts = [], [], []

#     for key, res in self.results.items():
#         meta = self.noise_meta.get(key, {"label": key})
#         labels.append(meta["label"])
#         best = res.get("best")
#         costs.append(best["cost"] if best else float("nan"))

#         sols = res.get("solutions", [])
#         total_freq = res.get("valid_freq",
#                                 sum(s["frequency"] for s in sols))
#         valid_pcts.append(total_freq * 100)

#     x = np.arange(len(labels))
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

#     # Best solution cost found
#     colors = [self.noise_meta.get(k, {}).get("color", "gray")
#                 for k in self.results.keys()]
#     bars = ax1.bar(x, costs, color=colors, alpha=0.85, edgecolor="black")
#     ax1.axhline(self.optimal_cost, color="green", lw=2,
#                 linestyle="--", label=f"Opt BF ({self.optimal_cost:.4f})")
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
#     ax1.set_ylabel("Best Route Cost Found", fontsize=11)
#     ax1.set_title("Solution Quality by Noise Model",
#                     fontsize=12, fontweight="bold")
#     ax1.legend()
#     ax1.grid(True, alpha=0.3, axis="y")
#     for bar, cost in zip(bars, costs):
#         if not np.isnan(cost):
#             ax1.text(bar.get_x() + bar.get_width() / 2,
#                         bar.get_height() + 0.001,
#                         f"{cost:.4f}", ha="center", va="bottom", fontsize=9)

#     # Frequency of valid solutions
#     ax2.bar(x, valid_pcts, color=colors, alpha=0.85, edgecolor="black")
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
#     ax2.set_ylabel("% of Samples with Valid Solution", fontsize=11)
#     ax2.set_title("Valid Solution Rate by Noise Model",
#                     fontsize=12, fontweight="bold")
#     ax2.set_ylim(0, 105)
#     ax2.grid(True, alpha=0.3, axis="y")
#     for bar, pct in zip(ax2.patches, valid_pcts):
#         ax2.text(bar.get_x() + bar.get_width() / 2,
#                     bar.get_height() + 1,
#                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

#     plt.tight_layout()
#     return fig

# # -- 6.3 Count Distributions -----------------------------------------------
# def plot_count_distributions(self):
#     n_models = len(self.results)
#     fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
#     if not isinstance(axes, np.ndarray):
#         axes = [axes]

#     for ax, (key, res) in zip(axes, self.results.items()):
#         meta = self.noise_meta.get(key, {"label": key, "color": "gray"})
#         counts = res["counts"]
#         sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:15]
#         labels_c = [bs[:6] + "…" for bs, _ in sorted_counts]
#         values = [c for _, c in sorted_counts]

#         ax.barh(range(len(labels_c)), values,
#                 color=meta["color"], alpha=0.8, edgecolor="black")
#         ax.set_yticks(range(len(labels_c)))
#         ax.set_yticklabels(labels_c, fontsize=8)
#         ax.set_xlabel("Count", fontsize=10)
#         ax.set_title(f"{meta['label']}\n({meta.get('desc','')})",
#                         fontsize=10, fontweight="bold")
#         ax.grid(True, alpha=0.3, axis="x")

#     plt.suptitle("Top-15 Bitstrings by Noise Model",
#                     fontsize=13, fontweight="bold")
#     plt.tight_layout()
#     return fig

# # -- 6.4 Summary Table -----------------------------------------------------
# def plot_summary_table(self):
#     rows = []
#     for key, res in self.results.items():
#         meta = self.noise_meta.get(key, {"label": key})
#         best = res.get("best")
#         best_cost = f"{best['cost']:.4f}" if best else "N/A"
#         approx_ratio = (best["cost"] / self.optimal_cost
#                         if best else float("nan"))
#         valid_pct = res.get("valid_freq", 0)
#         rows.append([
#             meta["label"],
#             meta.get("desc", "—"),
#             best_cost,
#             f"{approx_ratio:.3f}",
#             f"{valid_pct*100:.1f}%",
#             str(res["n_iter"]),
#         ])

#     fig, ax = plt.subplots(figsize=(14, 3))
#     ax.axis("off")
#     col_labels = ["Model", "Parameters", "Best Cost",
#                     "Approx Ratio", "% Valid", "Iterations"]
#     table = ax.table(cellText=rows, colLabels=col_labels,
#                         cellLoc="center", loc="center",
#                         bbox=[0, 0, 1, 1])
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     for (row, col), cell in table.get_celld().items():
#         if row == 0:
#             cell.set_facecolor("#2c3e50")
#             cell.set_text_props(color="white", fontweight="bold")
#         elif row % 2 == 0:
#             cell.set_facecolor("#ecf0f1")
#     ax.set_title("Comparative Summary — Noise Model Effects",
#                     fontsize=13, fontweight="bold", pad=15)
#     plt.tight_layout()
#     return fig

