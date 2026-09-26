"""
Regression-GNN Ablation Study
================================

Runs the same ablation comparison as run_ablation_experiment.py, but using
the REGRESSION-based GNN (gnn_regression.py) instead of the classification
version. This generates the CSV files that back Table 3 in the paper,
which were previously only reported in chat/conversation, not saved to a
file.

Produces:
  - ablation_results_regression.csv
  - ablation_significance_results_regression.csv

HOW TO RUN:
    python run_ablation_regression.py

Uses the SAME seed range (3000+trial) as the original ablation study, so
this is a fair, direct comparison against the classification-GNN ablation
results already saved in ablation_results.csv.
"""

import random
import numpy as np
import pandas as pd
from scipy import stats

from main_simulation_clean import (
    create_wsn_network,
    apply_disaster_scenario,
    assign_failure_probabilities,
    train_q_learning_routing_agent,
    find_rl_route,
    build_route_result,
    SINK_NODE,
    CRITICAL_PACKET_PRIORITY,
    DISASTER_SCENARIOS,
)

from gnn_regression import (
    train_gnn_failure_prediction_model,
    assign_gnn_failure_probabilities,
)


N_TRIALS = 30
SEVERITY_LEVELS = list(DISASTER_SCENARIOS.keys())  # ["Low", "Medium", "High"]


def get_weighted_rl_results_silent(graph):
    """Ablation condition 1: RL agent using the weighted-formula risk directly, no GNN."""
    assign_failure_probabilities(graph)
    q_table = train_q_learning_routing_agent(graph, SINK_NODE, packet_priority=CRITICAL_PACKET_PRIORITY)

    results = []
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    for patient_node in patient_nodes:
        path = find_rl_route(graph, q_table, patient_node, SINK_NODE)
        results.append(build_route_result(graph, "RL + Weighted-Formula (No GNN)", patient_node, path))
    return results


def get_gnn_rl_results_silent(graph):
    """Ablation condition 2: RL agent using the REGRESSION-trained GNN's risk prediction."""
    train_gnn_failure_prediction_model(graph)
    assign_gnn_failure_probabilities(graph)
    q_table = train_q_learning_routing_agent(graph, SINK_NODE, packet_priority=CRITICAL_PACKET_PRIORITY)

    results = []
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    for patient_node in patient_nodes:
        path = find_rl_route(graph, q_table, patient_node, SINK_NODE)
        results.append(build_route_result(graph, "GNN + RL (Regression, Proposed)", patient_node, path))
    return results


def summarize_trial(results):
    delivered = [r for r in results if r.get("delivered")]
    if not results:
        return None
    if not delivered:
        return {"route_delivery_ratio": 0.0, "average_latency": None, "average_hop_count": None,
                "average_energy_cost": None, "average_disaster_risk": None, "average_high_risk_nodes": None}
    return {
        "route_delivery_ratio": len(delivered) / len(results),
        "average_latency": sum(r["latency"] for r in delivered) / len(delivered),
        "average_hop_count": sum(r["hop_count"] for r in delivered) / len(delivered),
        "average_energy_cost": sum(r["energy_cost"] for r in delivered) / len(delivered),
        "average_disaster_risk": sum(r["average_risk"] for r in delivered) / len(delivered),
        "average_high_risk_nodes": sum(r["high_risk_nodes"] for r in delivered) / len(delivered),
    }


def run_ablation_regression():
    all_summary_rows = []
    significance_rows = []

    for severity in SEVERITY_LEVELS:
        print(f"\n===================== SEVERITY: {severity} =====================")

        trial_records = {
            "RL + Weighted-Formula (No GNN)": [],
            "GNN + RL (Regression, Proposed)": [],
        }

        for trial in range(N_TRIALS):
            print(f"\n--- {severity} | Trial {trial + 1}/{N_TRIALS} ---")

            # Same seed used for both conditions within a trial -- both see
            # the EXACT same network layout; only the risk source differs.
            random.seed(3000 + trial)
            np.random.seed(3000 + trial)
            base_graph_a = create_wsn_network()
            graph_a = apply_disaster_scenario(base_graph_a, severity)
            weighted_results = get_weighted_rl_results_silent(graph_a)

            random.seed(3000 + trial)
            np.random.seed(3000 + trial)
            base_graph_b = create_wsn_network()
            graph_b = apply_disaster_scenario(base_graph_b, severity)
            gnn_results = get_gnn_rl_results_silent(graph_b)

            trial_records["RL + Weighted-Formula (No GNN)"].append(summarize_trial(weighted_results))
            trial_records["GNN + RL (Regression, Proposed)"].append(summarize_trial(gnn_results))

        metrics = ["route_delivery_ratio", "average_latency", "average_hop_count",
                   "average_energy_cost", "average_disaster_risk", "average_high_risk_nodes"]

        for model_name, records in trial_records.items():
            row = {"severity": severity, "model": model_name}
            for metric in metrics:
                values = [r[metric] for r in records if r is not None and r.get(metric) is not None]
                if values:
                    row[f"{metric}_mean"] = round(float(np.mean(values)), 4)
                    row[f"{metric}_std"] = round(float(np.std(values, ddof=1)), 4)
                else:
                    row[f"{metric}_mean"] = None
                    row[f"{metric}_std"] = None
            all_summary_rows.append(row)

        print(f"\n--- {severity}: Paired t-tests (No-GNN vs Regression-GNN) ---")
        for metric in metrics:
            no_gnn_vals = [r[metric] for r in trial_records["RL + Weighted-Formula (No GNN)"] if r and r.get(metric) is not None]
            gnn_vals = [r[metric] for r in trial_records["GNN + RL (Regression, Proposed)"] if r and r.get(metric) is not None]

            if len(no_gnn_vals) == len(gnn_vals) and len(no_gnn_vals) > 1:
                t_stat, p_value = stats.ttest_rel(no_gnn_vals, gnn_vals)
                print(f"  {metric}: No-GNN mean={np.mean(no_gnn_vals):.4f}, "
                      f"GNN mean={np.mean(gnn_vals):.4f}, p={p_value:.4f}")
                significance_rows.append({
                    "severity": severity, "metric": metric,
                    "no_gnn_mean": round(float(np.mean(no_gnn_vals)), 4),
                    "gnn_mean": round(float(np.mean(gnn_vals)), 4),
                    "p_value": round(float(p_value), 4),
                    "significant_at_0.05": bool(p_value < 0.05)
                })

    results_df = pd.DataFrame(all_summary_rows)
    results_df.to_csv("ablation_results_regression.csv", index=False)

    significance_df = pd.DataFrame(significance_rows)
    significance_df.to_csv("ablation_significance_results_regression.csv", index=False)

    print("\n\n===================== ABLATION RESULTS (REGRESSION GNN) =====================")
    print(results_df.to_string())
    print("\n\n===================== SIGNIFICANCE TESTS =====================")
    print(significance_df.to_string())

    print("\nSaved: ablation_results_regression.csv")
    print("Saved: ablation_significance_results_regression.csv")

    return results_df, significance_df


if __name__ == "__main__":
    run_ablation_regression()
