"""
Full 30-Trial, Multi-Severity Experiment
=========================================

Runs all six routing models (Shortest-Path, Energy-Aware, Weighted
Failure-Risk-Aware, LSTM, LSTM-RL, GNN-RL) across Low/Medium/High disaster
severity, 30 independent trials each, with a DIFFERENT random seed per
trial (critical -- otherwise every "trial" would be identical).

Produces:
  - full_experiment_results.csv   (mean +/- std per model per severity)
  - A printed paired t-test (LSTM-RL vs GNN-RL) for each severity level

HOW TO RUN:
  python run_full_experiment.py

This will take a while (30 trials x 3 severities x ~1500 RL training
episodes per trial, twice per trial for LSTM-RL and GNN-RL). Consider
running it and leaving it, rather than watching it live.

NOTE ON GNN TRAINING: the GNN is trained ONCE per trial's network graph
(not once total), so that its topology-aware predictions are re-learned
for each newly generated network -- this matches how the paper says the
GNN adapts to the network it's deployed on, and keeps the comparison to
LSTM (which is also re-run per trial via assign_lstm_failure_probabilities,
using the single pre-trained LSTM model) as fair as possible.
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
    calculate_path_latency,
    calculate_path_failure_risk,
    count_predicted_high_risk_nodes_in_path,
    calculate_path_average_energy,
    calculate_path_energy_cost,
    calculate_path_risk,
    count_high_risk_nodes_in_path,
    build_route_result,
    SINK_NODE,
    CRITICAL_PACKET_PRIORITY,
    DISASTER_SCENARIOS,
)

from gnn_failure_prediction import (
    train_gnn_failure_prediction_model,
    assign_gnn_failure_probabilities,
)


N_TRIALS = 30
SEVERITY_LEVELS = list(DISASTER_SCENARIOS.keys())  # ["Low", "Medium", "High"]


def get_gnn_rl_results_silent(graph):
    """
    Run GNN + RL self-healing routing and return results in the same
    silent-result format as the other get_*_results_silent() functions,
    so it can be summarized the same way.
    """
    train_gnn_failure_prediction_model(graph)
    assign_gnn_failure_probabilities(graph)

    q_table = train_q_learning_routing_agent(
        graph, SINK_NODE, packet_priority=CRITICAL_PACKET_PRIORITY
    )

    results = []
    patient_nodes = [
        node for node, data in graph.nodes(data=True)
        if data["node_type"] == "patient_sensor"
    ]

    for patient_node in patient_nodes:
        path = find_rl_route(graph, q_table, patient_node, SINK_NODE)
        results.append(build_route_result(graph, "GNN + RL Self-Healing", patient_node, path))

    return results


def summarize_trial(results):
    """
    Compute simple per-trial averages (delivery ratio, latency, hop count,
    energy, disaster risk, high-risk nodes) from one routing_results list.
    Returns None for a metric if no packets were delivered.
    """
    delivered = [r for r in results if r.get("delivered")]

    if not results:
        return None

    if not delivered:
        return {
            "route_delivery_ratio": 0.0,
            "average_latency": None,
            "average_hop_count": None,
            "average_energy_cost": None,
            "average_disaster_risk": None,
            "average_high_risk_nodes": None,
        }

    return {
        "route_delivery_ratio": len(delivered) / len(results),
        "average_latency": sum(r["latency"] for r in delivered) / len(delivered),
        "average_hop_count": sum(r["hop_count"] for r in delivered) / len(delivered),
        "average_energy_cost": sum(r["energy_cost"] for r in delivered) / len(delivered),
        "average_disaster_risk": sum(r["average_risk"] for r in delivered) / len(delivered),
        "average_high_risk_nodes": sum(r["high_risk_nodes"] for r in delivered) / len(delivered),
    }


def run_full_experiment():
    all_summary_rows = []
    significance_rows = []

    for severity in SEVERITY_LEVELS:
        print(f"\n===================== SEVERITY: {severity} =====================")

        trial_records = {
            "Shortest-Path": [], "Energy-Aware": [], "Weighted-Risk-Aware": [],
            "LSTM": [], "LSTM-RL": [], "GNN-RL": []
        }

        for trial in range(N_TRIALS):
            print(f"\n--- {severity} | Trial {trial + 1}/{N_TRIALS} ---")

            # IMPORTANT: different seed each trial, so trials are genuinely
            # independent network realizations, not 30 copies of the same one.
            random.seed(1000 + trial)
            np.random.seed(1000 + trial)

            base_graph = create_wsn_network()
            graph = apply_disaster_scenario(base_graph, severity)

            sp_results = get_shortest_path_results_silent(graph)
            ea_results = get_energy_aware_results_silent(graph)
            wf_results = get_weighted_failure_aware_results_silent(graph)
            lstm_results = get_lstm_failure_aware_results_silent(graph)
            lstm_rl_results = get_lstm_rl_results_silent(graph)
            gnn_rl_results = get_gnn_rl_results_silent(graph)

            trial_records["Shortest-Path"].append(summarize_trial(sp_results))
            trial_records["Energy-Aware"].append(summarize_trial(ea_results))
            trial_records["Weighted-Risk-Aware"].append(summarize_trial(wf_results))
            trial_records["LSTM"].append(summarize_trial(lstm_results))
            trial_records["LSTM-RL"].append(summarize_trial(lstm_rl_results))
            trial_records["GNN-RL"].append(summarize_trial(gnn_rl_results))

        # --- Compute mean +/- std across the 30 trials, per model ---
        metrics = [
            "route_delivery_ratio", "average_latency", "average_hop_count",
            "average_energy_cost", "average_disaster_risk", "average_high_risk_nodes"
        ]

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

        # --- Paired t-test: LSTM-RL vs GNN-RL, per metric, for this severity ---
        print(f"\n--- {severity}: Paired t-tests (LSTM-RL vs GNN-RL) ---")
        for metric in ["average_latency", "average_hop_count", "average_energy_cost",
                       "average_disaster_risk", "average_high_risk_nodes"]:
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
            else:
                print(f"  {metric}: insufficient paired data for t-test")

    # --- Save results ---
    results_df = pd.DataFrame(all_summary_rows)
    results_df.to_csv("full_experiment_results.csv", index=False)

    significance_df = pd.DataFrame(significance_rows)
    significance_df.to_csv("significance_test_results.csv", index=False)

    print("\n\n===================== FULL RESULTS =====================")
    print(results_df.to_string())

    print("\n\n===================== SIGNIFICANCE TESTS =====================")
    print(significance_df.to_string())

    print("\nSaved: full_experiment_results.csv")
    print("Saved: significance_test_results.csv")

    return results_df, significance_df


if __name__ == "__main__":
    run_full_experiment()
