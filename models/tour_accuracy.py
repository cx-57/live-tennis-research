"""Render Figure 4 from the saved ATP/WTA result table."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def plot_atp_wta_results(results):
    """Render Figure 4 from the ATP/WTA accuracy results table."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    series = [
        ("markov_accuracy", "Elo-asymmetric Markov", {"marker": "o", "linewidth": 3}),
        ("serve_shrink_accuracy", "Serve-shrink Markov", {"marker": "o", "linewidth": 3}),
        ("trace_accuracy", "Trace", {"marker": "o", "linewidth": 3}),
    ]

    for axis, tour in zip(axes, ("ATP", "WTA")):
        tour_results = results[results["tour"] == tour].sort_values("percent")
        for column, label, style in series:
            axis.plot(tour_results["percent"], 100 * tour_results[column], label=label, **style)
        axis.set_title(tour, fontsize=22)
        axis.set_xlabel("Match Progress (%)", fontsize=20)
        axis.set_xlim(0, 100)
        axis.set_ylim(65, 100)
        axis.tick_params(axis="both", labelsize=16)

    axes[0].set_ylabel("Accuracy (%)", fontsize=20)
    axes[1].legend(fontsize=16, loc="lower right")
    fig.suptitle("ATP vs WTA Accuracy by Match Progress", fontsize=24)
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "figure_4_tour_accuracy.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot_atp_wta_results(pd.read_csv(ROOT / "results" / "atp_wta_match_fraction_accuracy.csv"))
