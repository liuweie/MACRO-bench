import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Configuration
# =========================

EXP_DIR = "./trace_output/exp_repeat_10"
OUTPUT_DIR = "./trace_output/stability_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = {
    "u_task_success_rate": "avg_u_task_success_rate",
    "s_task_success_rate": "avg_s_task_success_rate",
    "u_task_complete_rate": "avg_u_task_complete_rate",
    "s_task_complete_rate": "avg_s_task_complete_rate",
    "execution_efficiency": "avg_execution_efficiency",
    "clarification_efficiency": "avg_clarification_efficiency",
    "agent_routing_accuracy": "avg_agent_routing_accuracy",
}

# =========================
# 2. Read all summary.json files
# =========================

summary_files = sorted(
    glob.glob(os.path.join(EXP_DIR, "*_summary.json"))
)

if len(summary_files) == 0:
    raise RuntimeError("No summary.json files found.")

records = []

for run_idx, filepath in enumerate(summary_files, start=1):
    with open(filepath, "r") as f:
        data = json.load(f)

    row = {"run": run_idx}
    avg_scores = data["summary"]["average_scores"]

    for metric, key in METRICS.items():
        row[metric] = avg_scores[key]

    records.append(row)

df = pd.DataFrame(records)
df.set_index("run", inplace=True)

print("\n=== Raw metric values per run ===")
print(df)

# =========================
# 3. Stability stats (Mean / Std / CV / Min / Max)
# =========================

stats = []

for metric in df.columns:
    values = df[metric].values
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    cv = std / mean if mean != 0 else 0.0
    min_v = np.min(values)
    max_v = np.max(values)

    stats.append({
        "metric": metric,
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": min_v,
        "max": max_v,
    })

stability_df = pd.DataFrame(stats).sort_values("cv")

print("\n=== Stability statistics ===")
print(stability_df)

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, "evaluation_stability_stats.csv")
stability_df.to_csv(csv_path, index=False)

# =========================
# 4. Line chart: save as image
# =========================

# plt.figure(figsize=(12, 6))

# for metric in df.columns:
#     plt.plot(
#         df.index,
#         df[metric],
#         marker="o",
#         linewidth=2,
#         label=metric
#     )

# plt.xlabel("Run Index")
# plt.ylabel("Metric Value")
# plt.title("Stability of Evaluation Metrics Across 10 Repeated Runs")
# plt.xticks(df.index)
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend(
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5),
#     borderaxespad=0.0,
#     fontsize=9
# )
# plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

# figure_path = os.path.join(
#     OUTPUT_DIR,
#     "evaluation_metrics_stability_across_runs.png"
# )

# plt.savefig(figure_path, dpi=300)
# plt.close()
plt.figure(figsize=(12, 6))

# Compute CV for each metric first
metric_cvs = {}
for metric in df.columns:
    values = df[metric].values
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    cv = (std / mean * 100) if mean != 0 else 0.0  # convert to percentage
    metric_cvs[metric] = cv

# Add CV to each chart label
for metric in df.columns:
    plt.plot(
        df.index,
        df[metric],
        marker="o",
        linewidth=2,
        label=f"{metric} (CV={metric_cvs[metric]:.1f}%)"
    )

plt.xlabel("Run Index")
plt.ylabel("Metric Value")
plt.title("Stability of Evaluation Metrics Across 10 Repeated Runs")
plt.xticks(df.index)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0,
    fontsize=9
)
plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

figure_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_metrics_stability_across_runs.png"
)

plt.savefig(figure_path, dpi=300)
plt.close()

print(f"\nFigure saved to: {figure_path}")
print(f"CSV saved to: {csv_path}")
