
import os

import matplotlib.pyplot as plt
import pandas as pd

# Load the completed comparison results
results = pd.read_csv(
    "results/recommender_comparison_results.csv"
)

os.makedirs("results", exist_ok=True)

# Calculate mean and standard deviation across seeds
summary = (
    results.groupby(["availability", "method"])["mean_ndcg"]
    .agg(["mean", "std"])
    .reset_index()
)

method_labels = {
    "collaborative": "Collaborative (SVD)",
    "content": "Content-based (TF-IDF)",
    "hybrid": "Hybrid (50/50)"
}

# Figure 1: Comparison of all three recommendation methods
plt.figure(figsize=(10, 6))

for method, label in method_labels.items():
    method_data = summary[
        summary["method"] == method
    ].sort_values("availability")

    plt.plot(
        method_data["availability"],
        method_data["mean"],
        marker="o",
        linewidth=2,
        label=label
    )

plt.xlabel("Training Data Availability (%)")
plt.ylabel("Mean NDCG@10")
plt.title("Recommendation Performance Across Data Availability Levels")
plt.xticks([20, 40, 60, 80, 100])
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/recommender_performance_comparison.png",
    dpi=300
)
plt.close()

# Figure 2: Variability across random seeds
plt.figure(figsize=(10, 6))

for method, label in method_labels.items():
    method_data = summary[
        summary["method"] == method
    ].sort_values("availability")

    x = method_data["availability"].to_numpy()
    mean = method_data["mean"].to_numpy()
    std = method_data["std"].fillna(0).to_numpy()

    plt.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        label=label
    )

    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.15
    )

plt.xlabel("Training Data Availability (%)")
plt.ylabel("Mean NDCG@10")
plt.title("Recommendation Performance and Variation Across Seeds")
plt.xticks([20, 40, 60, 80, 100])
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/recommender_performance_variability.png",
    dpi=300
)
plt.close()

print("Saved results/recommender_performance_comparison.png")
print("Saved results/recommender_performance_variability.png")
