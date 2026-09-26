"""
Extreme Severity Experiment — REGRESSION GNN VERSION
========================================================

Re-runs the Extreme-severity experiment using the regression-based GNN
(gnn_regression.py) instead of the original classification-based GNN,
to confirm whether the earlier significant delivery-ratio finding
(95.25% vs 89.62%, p=0.03) holds up with the improved model that fixed
the ablation-study gap.

Uses 100 trials (matching the trial count that produced the original
significant result), with the SAME seed range (2000+trial) as the
original run, so this is a fair, direct comparison against that result.

HOW TO RUN:
    python run_extreme_experiment_regression.py
"""

import random
import numpy as np
import pandas as pd
from scipy import stats

from main_simulation_clean import (
    create_wsn_network,
    apply_disaster_scenario,
    get_shortest_path_results_silent,
    get_energy_aware_results_silent,
    get_weighted_failure_aware_results_silent,
    get_lstm_failure_aware_results_silent,
    get_lstm_rl_results_silent,
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


# ------------------------------------------------------------------
# Add "Extreme" severity, scaled up from "High"
# High:   temp+0.28, loss+0.15, delay+35, energy-18, load+10, rssi-10
# Extreme: roughly another 1.4-1.6x step beyond High
# ------------------------------------------------------------------
DISASTER_SCENARIOS["Extreme"] = {
    "temperature_add": 0.42,
    "packet_loss_add": 0.24,
    "delay_add": 55,
    "energy_drain": 28,
    "load_add": 16,
    "rssi_drop": 16
}

N_TRIALS = 100
SEVERITY_LEVELS = ["Extreme"]  # only run the new level


def get_gnn_rl_results_silent(graph):
    train_gnn_failure_prediction_model(graph)
    assign_gnn_failure_probabilities(graph)
    q_table = train_q_learning_routing_agent(graph, SINK_NODE, packet_priority=CRITICAL_PACKET_PRIORITY)
    results = []
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    for patient_node in patient_nodes:
        path = find_rl_route(graph, q_table, patient_node, SINK_NODE)
        results.append(build_route_result(graph, "GNN + RL Self-Healing", patient_node, path))
    return results


def summarize_trial(results, trial_number, model_name, failure_log):
    """Same as before, but also logs which specific patient nodes failed
    to find a route, for a given trial/model, so we can investigate
    WHY failures happen rather than just how often."""
    delivered = [r for r in results if r.get("delivered")]
    failed = [r for r in results if not r.get("delivered")]

    if failed:
        for r in failed:
            failure_log.append({
                "trial": trial_number,
                "model": model_name,
                "patient_node": r["patient_node"]
            })

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


def run_extreme_experiment():
    all_summary_rows = []
    significance_rows = []
    failure_log = []

    for severity in SEVERITY_LEVELS:
        print(f"\n===================== SEVERITY: {severity} =====================")

        trial_records = {
            "Shortest-Path": [], "Energy-Aware": [], "Weighted-Risk-Aware": [],
            "LSTM": [], "LSTM-RL": [], "GNN-RL": []
        }

        for trial in range(N_TRIALS):
            print(f"\n--- {severity} | Trial {trial + 1}/{N_TRIALS} ---")

            random.seed(2000 + trial)  # different seed range than the main experiment
            np.random.seed(2000 + trial)

            base_graph = create_wsn_network()
            graph = apply_disaster_scenario(base_graph, severity)

            sp_results = get_shortest_path_results_silent(graph)
            ea_results = get_energy_aware_results_silent(graph)
            wf_results = get_weighted_failure_aware_results_silent(graph)
            lstm_results = get_lstm_failure_aware_results_silent(graph)
            lstm_rl_results = get_lstm_rl_results_silent(graph)
            gnn_rl_results = get_gnn_rl_results_silent(graph)

            trial_records["Shortest-Path"].append(summarize_trial(sp_results, trial, "Shortest-Path", failure_log))
            trial_records["Energy-Aware"].append(summarize_trial(ea_results, trial, "Energy-Aware", failure_log))
            trial_records["Weighted-Risk-Aware"].append(summarize_trial(wf_results, trial, "Weighted-Risk-Aware", failure_log))
            trial_records["LSTM"].append(summarize_trial(lstm_results, trial, "LSTM", failure_log))
            trial_records["LSTM-RL"].append(summarize_trial(lstm_rl_results, trial, "LSTM-RL", failure_log))
            trial_records["GNN-RL"].append(summarize_trial(gnn_rl_results, trial, "GNN-RL", failure_log))

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

        print(f"\n--- {severity}: Paired t-tests (LSTM-RL vs GNN-RL) ---")
        for metric in ["route_delivery_ratio", "average_latency", "average_hop_count",
                       "average_energy_cost", "average_disaster_risk", "average_high_risk_nodes"]:
            lstm_rl_vals = [r[metric] for r in trial_records["LSTM-RL"] if r and r.get(metric) is not None]
            gnn_rl_vals = [r[metric] for r in trial_records["GNN-RL"] if r and r.get(metric) is not None]

            if len(lstm_rl_vals) == len(gnn_rl_vals) and len(lstm_rl_vals) > 1:
                t_stat, p_value = stats.ttest_rel(lstm_rl_vals, gnn_rl_vals)
                print(f"  {metric}: LSTM-RL mean={np.mean(lstm_rl_vals):.4f}, "
                      f"GNN-RL mean={np.mean(gnn_rl_vals):.4f}, p={p_value:.4f}")
                significance_rows.append({
                    "severity": severity, "metric": metric,
                    "lstm_rl_mean": round(float(np.mean(lstm_rl_vals)), 4),
                    "gnn_rl_mean": round(float(np.mean(gnn_rl_vals)), 4),
                    "p_value": round(float(p_value), 4),
                    "significant_at_0.05": bool(p_value < 0.05)
                })

    results_df = pd.DataFrame(all_summary_rows)
    results_df.to_csv("extreme_experiment_results_regression.csv", index=False)

    significance_df = pd.DataFrame(significance_rows)
    significance_df.to_csv("extreme_significance_results_regression.csv", index=False)

    failure_df = pd.DataFrame(failure_log)
    failure_df.to_csv("extreme_failure_log_regression.csv", index=False)

    print("\n\n===================== EXTREME RESULTS =====================")
    print(results_df.to_string())

    print("\n\n===================== SIGNIFICANCE TESTS =====================")
    print(significance_df.to_string())

    print("\n\n===================== ROUTE-FINDING FAILURES =====================")
    if len(failure_df) > 0:
        print(failure_df.to_string())
        print(f"\nTotal failures: {len(failure_df)}")
        print("\nFailures by model:")
        print(failure_df["model"].value_counts())
    else:
        print("No route-finding failures occurred in any trial/model.")

    print("\nSaved: extreme_experiment_results_regression.csv")
    print("Saved: extreme_significance_results_regression.csv")
    print("Saved: extreme_failure_log_regression.csv")

    return results_df, significance_df, failure_df


if __name__ == "__main__":
    run_extreme_experiment()
