"""Plot ATP and WTA model accuracy across match progress from saved results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images"
CSV_PATH = IMAGE_DIR / "atp_wta_match_fraction_accuracy.csv"
PNG_PATH = IMAGE_DIR / "atp_wta_match_fraction_accuracy.png"
PDF_PATH = IMAGE_DIR / "atp_wta_match_fraction_accuracy.pdf"

SERIES = [
    ("markov_accuracy", "Asymmetric Markov"),
    ("serve_shrink_accuracy", "Serve-shrink Markov"),
    ("residual_accuracy", "Markov Ensemble"),
]


def plot_accuracy():
    results = pd.read_csv(CSV_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    for axis, tour in zip(axes, ("ATP", "WTA")):
        tour_results = results[results["tour"] == tour].sort_values("percent")
        for column, label in SERIES:
            axis.plot(
                tour_results["percent"],
                100 * tour_results[column],
                marker="o",
                markersize=8,
                linewidth=3,
                label=label,
            )

        axis.set_title(tour, fontsize=22)
        axis.set_xlabel("Match Progress (%)", fontsize=20)
        axis.set_xlim(0, 100)
        axis.set_ylim(65, 100)
        axis.tick_params(axis="both", labelsize=16)

    axes[0].set_ylabel("Accuracy (%)", fontsize=20)
    axes[1].legend(fontsize=16, loc="lower right")
    fig.suptitle("ATP vs WTA Accuracy by Match Progress", fontsize=24)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"saved graph to {PNG_PATH}")
    print(f"saved PDF to {PDF_PATH}")


if __name__ == "__main__":
    plot_accuracy()
