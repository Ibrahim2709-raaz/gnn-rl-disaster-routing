"""
GNN-Based Failure Prediction + GNN-RL Self-Healing Routing
============================================================

This module implements the GNN component described in the paper
(Section 3.4 / 4.7) which was missing from the shared codebase
(only an LSTM + RL implementation was found).

Architecture matches paper specifications:
  - 2 graph aggregation (message-passing) layers
  - 6 input features (residual_energy, temperature_risk, packet_loss_rate,
    rssi, delay, forwarding_load)
  - 32-dimensional hidden layer
  - Trained for 250 epochs, Adam optimizer, learning rate 0.01

HOW TO USE:
  1. Place this file in the same folder as your main WSN simulation script
     (the one with create_wsn_network, FEATURE_COLUMNS, etc.)
  2. Import the functions you need, e.g.:
       from gnn_failure_prediction import (
           train_gnn_failure_prediction_model,
           assign_gnn_failure_probabilities,
           test_gnn_rl_self_healing_routing
       )
  3. Call train_gnn_failure_prediction_model(graph) once to train and save
     the GNN, matching the same pattern as train_lstm_failure_prediction_model()
     in your existing code.
  4. Use assign_gnn_failure_probabilities(graph) the same way you use
     assign_lstm_failure_probabilities(graph) — it fills in
     graph.nodes[node]["failure_probability"] and ["risk_status"] using
     the GNN instead of the LSTM.
  5. Use test_gnn_rl_self_healing_routing(graph) the same way you use
     test_lstm_rl_self_healing_routing(graph) — this is your proposed
     GNN-RL model, directly comparable to the existing LSTM-RL baseline.

IMPORTANT: This reuses your existing Q-learning RL agent code
(train_q_learning_routing_agent, find_rl_route, calculate_rl_step_reward)
so the ONLY thing that changes between LSTM-RL and GNN-RL is the source
of the failure_probability values feeding into routing decisions —
which is exactly the controlled comparison the paper's central
hypothesis depends on.
"""

import os
import random
import numpy as np
import networkx as nx

PYTORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    PYTORCH_AVAILABLE = False
    print("\nPyTorch is not installed.")
    print("Install it using: py -m pip install torch")


# ============================================================
# CONFIGURATION (matches paper Section 4.7)
# ============================================================

GNN_INPUT_FEATURES = 6          # residual_energy, temperature_risk, packet_loss_rate, rssi, delay, forwarding_load
GNN_HIDDEN_DIM = 32              # matches paper: 32-dimensional hidden layer
GNN_NUM_LAYERS = 2                # matches paper: 2 graph aggregation stages
GNN_EPOCHS = 250                 # matches paper: 250 epochs
GNN_LEARNING_RATE = 0.01          # matches paper: Adam, lr=0.01

GNN_RISK_THRESHOLD = 0.55         # kept consistent with the LSTM threshold in the shared code
MODEL_DIR_NAME = "models"
GNN_MODEL_FILENAME = "gnn_failure_prediction_model_pytorch.pt"

FEATURE_COLUMNS = [
    "residual_energy",
    "temperature_risk",
    "packet_loss_rate",
    "rssi",
    "delay",
    "forwarding_load"
]


# ============================================================
# DIRECTORY HELPERS (mirrors the pattern in the shared LSTM code)
# ============================================================

def get_safe_model_dir():
    """Create a safe model folder, matching the pattern used for the LSTM model."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    preferred_dir = os.path.join(script_dir, MODEL_DIR_NAME)

    try:
        os.makedirs(preferred_dir, exist_ok=True)
        return preferred_dir
    except PermissionError:
        fallback_dir = os.path.join(os.path.expanduser("~"), "wsn_ai_routing_models")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


def find_gnn_model_file():
    """Find the saved GNN model file."""
    possible_paths = []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    possible_paths.append(os.path.join(script_dir, MODEL_DIR_NAME, GNN_MODEL_FILENAME))
    possible_paths.append(os.path.join(os.getcwd(), MODEL_DIR_NAME, GNN_MODEL_FILENAME))
    possible_paths.append(os.path.join(os.path.expanduser("~"), "wsn_ai_routing_models", GNN_MODEL_FILENAME))

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


# ============================================================
# GRAPH DATA PREPARATION
# ============================================================

def build_adjacency_matrix(graph):
    """
    Build a normalized adjacency matrix (with self-loops) from the WSN graph,
    ordered consistently with a fixed node-id list.

    Returns:
        node_ids: list of node ids in the order used by the matrix/features
        A_norm: normalized adjacency matrix (numpy array), shape [N, N]
    """
    node_ids = list(graph.nodes())
    n = len(node_ids)
    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    A = np.zeros((n, n), dtype=np.float32)

    for node_a, node_b in graph.edges():
        i, j = id_to_idx[node_a], id_to_idx[node_b]
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Add self-loops (each node aggregates its own previous representation too)
    A = A + np.eye(n, dtype=np.float32)

    # Symmetric degree normalization: D^-1/2 * A * D^-1/2 (standard GCN-style normalization)
    degrees = A.sum(axis=1)
    degrees[degrees == 0] = 1.0  # avoid division by zero for isolated nodes
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return node_ids, A_norm.astype(np.float32)


def build_node_feature_matrix(graph, node_ids):
    """
    Build the node feature matrix X, shape [N, 6], in the same order as node_ids.
    Applies the same normalization ranges as the LSTM feature normalization
    in the shared code, for consistency across models.
    """
    X = np.zeros((len(node_ids), GNN_INPUT_FEATURES), dtype=np.float32)

    for i, node_id in enumerate(node_ids):
        data = graph.nodes[node_id]

        residual_energy = data.get("residual_energy", 100.0) / 100.0
        temperature_risk = data.get("temperature_risk", 0.0)  # already 0-1
        packet_loss_rate = data.get("packet_loss_rate", 0.0)  # already 0-1
        rssi = (data.get("rssi", -40) + 95) / 60.0             # -95..-35 -> 0..1
        delay = data.get("delay", 0.0) / 150.0                  # 0..150 -> 0..1
        forwarding_load = data.get("forwarding_load", 0.0) / 40.0  # 0..40 -> 0..1

        X[i] = [
            residual_energy,
            temperature_risk,
            packet_loss_rate,
            rssi,
            delay,
            forwarding_load
        ]

    return X


def calculate_pseudo_label_failure_probability(residual_energy_norm, temperature_risk,
                                                packet_loss_rate, rssi_norm,
                                                delay_norm, forwarding_load_norm):
    """
    Weighted failure-risk scoring mechanism used to generate pseudo-labels
    for GNN training, consistent with the weighted-risk formula already
    used elsewhere in the codebase (0.25 energy, 0.25 disaster risk,
    0.20 packet loss, 0.10 rssi, 0.10 delay, 0.10 load).

    NOTE: inputs here are already normalized to [0,1], matching the
    normalized feature matrix used for GNN training.
    """
    energy_risk = 1 - residual_energy_norm

    failure_probability = (
        0.25 * energy_risk +
        0.25 * temperature_risk +
        0.20 * packet_loss_rate +
        0.10 * rssi_norm +
        0.10 * delay_norm +
        0.10 * forwarding_load_norm
    )

    return min(max(failure_probability, 0.0), 1.0)


def build_pseudo_labels(X, node_ids, graph):
    """
    Build pseudo-labels (y) for GNN training using the weighted
    failure-risk scoring mechanism, for every non-sink node.
    Sink node label is fixed at 0 (never at risk).
    """
    y = np.zeros((len(node_ids), 1), dtype=np.float32)

    for i, node_id in enumerate(node_ids):
        if graph.nodes[node_id].get("node_type") == "sink":
            y[i] = 0.0
            continue

        residual_energy_norm, temperature_risk, packet_loss_rate, rssi_norm, delay_norm, forwarding_load_norm = X[i]

        failure_probability = calculate_pseudo_label_failure_probability(
            residual_energy_norm, temperature_risk, packet_loss_rate,
            rssi_norm, delay_norm, forwarding_load_norm
        )

        # Binary pseudo-label using the same risk threshold as other models
        y[i] = 1.0 if failure_probability >= GNN_RISK_THRESHOLD else 0.0

    return y


# ============================================================
# GNN MODEL DEFINITION
# ============================================================

if PYTORCH_AVAILABLE:

    class GraphConvLayer(nn.Module):
        """
        A single graph-convolution (message-passing) layer.
        Implements: H' = sigma( A_norm @ H @ W + b )
        This is the standard GCN-style propagation rule (Kipf & Welling, 2017).
        """
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, X, A_norm):
            # X: [N, in_features], A_norm: [N, N]
            aggregated = torch.matmul(A_norm, X)      # aggregate neighbor features
            out = self.linear(aggregated)               # apply learnable transform
            return out


    class GNNFailurePredictor(nn.Module):
        """
        GNN failure-risk prediction model matching paper Section 3.4:
          - 2 graph aggregation (message-passing) layers
          - 6 input features -> 32-dim hidden layer -> failure probability output
        """
        def __init__(self, input_size=GNN_INPUT_FEATURES, hidden_dim=GNN_HIDDEN_DIM):
            super().__init__()
            self.layer1 = GraphConvLayer(input_size, hidden_dim)
            self.layer2 = GraphConvLayer(hidden_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, X, A_norm):
            h = self.relu(self.layer1(X, A_norm))
            h = self.relu(self.layer2(h, A_norm))
            out = self.sigmoid(self.output_layer(h))
            return out


# ============================================================
# TRAINING
# ============================================================

def train_gnn_failure_prediction_model(graph):
    """
    Train the GNN failure prediction model on the given WSN graph,
    using pseudo-labels generated from the weighted failure-risk formula
    (since real-world failure-labeled data is unavailable — Section 4.7).

    Mirrors the structure of train_lstm_failure_prediction_model() in the
    existing codebase, for a fair, consistent comparison.
    """
    if not PYTORCH_AVAILABLE:
        print("\nCannot train GNN because PyTorch is not installed.")
        return None

    node_ids, A_norm = build_adjacency_matrix(graph)
    X = build_node_feature_matrix(graph, node_ids)
    y = build_pseudo_labels(X, node_ids, graph)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = GNNFailurePredictor()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE)

    print("\n========== GNN TRAINING STARTED ==========")
    print(model)
    print(f"Nodes: {len(node_ids)}, Epochs: {GNN_EPOCHS}, LR: {GNN_LEARNING_RATE}")
    print("===========================================\n")

    model.train()
    for epoch in range(GNN_EPOCHS):
        optimizer.zero_grad()
        predictions = model(X_tensor, A_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                predicted_labels = (predictions >= 0.5).float()
                accuracy = (predicted_labels == y_tensor).float().mean().item()
            print(f"Epoch {epoch + 1}/{GNN_EPOCHS} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

    print("\n========== GNN TRAINING COMPLETE ==========")
    print("============================================\n")

    model_dir = get_safe_model_dir()
    model_path = os.path.join(model_dir, GNN_MODEL_FILENAME)
    torch.save(model.state_dict(), model_path)

    print(f"GNN model saved to: {model_path}")

    return model


def load_trained_gnn_model():
    """Load a previously trained GNN model."""
    if not PYTORCH_AVAILABLE:
        print("\nCannot load GNN model because PyTorch is not installed.")
        return None

    model_path = find_gnn_model_file()
    if model_path is None:
        print("\nCould not find trained GNN model. Run train_gnn_failure_prediction_model() first.")
        return None

    model = GNNFailurePredictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    print(f"\nGNN model loaded from: {model_path}")
    return model


# ============================================================
# INFERENCE: ASSIGN FAILURE PROBABILITIES TO THE GRAPH
# ============================================================

def assign_gnn_failure_probabilities(graph, model=None):
    """
    Use the trained GNN to assign a topology-aware failure_probability
    and risk_status to every node in the graph — the direct analog of
    assign_lstm_failure_probabilities() in the existing codebase, but
    using graph connectivity instead of a per-node temporal sequence.
    """
    if model is None:
        model = load_trained_gnn_model()

    if model is None:
        print("\nGNN model could not be loaded. Cannot assign predictions.")
        return

    node_ids, A_norm = build_adjacency_matrix(graph)
    X = build_node_feature_matrix(graph, node_ids)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A_norm, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor, A_tensor).numpy().flatten()

    for i, node_id in enumerate(node_ids):
        if graph.nodes[node_id]["node_type"] == "sink":
            graph.nodes[node_id]["failure_probability"] = 0.0
            graph.nodes[node_id]["risk_status"] = "Sink"
            graph.nodes[node_id]["prediction_source"] = "sink"
            continue

        gnn_probability = float(predictions[i])
        graph.nodes[node_id]["failure_probability"] = gnn_probability
        graph.nodes[node_id]["prediction_source"] = "GNN"

        if gnn_probability >= GNN_RISK_THRESHOLD:
            graph.nodes[node_id]["risk_status"] = "High-risk"
        else:
            graph.nodes[node_id]["risk_status"] = "Normal"


def print_gnn_failure_risk_summary(graph):
    """Print GNN-based failure probability summary (mirrors the LSTM version)."""
    high_risk_nodes = [
        node for node, data in graph.nodes(data=True)
        if data.get("risk_status") == "High-risk"
    ]
    normal_nodes = [
        node for node, data in graph.nodes(data=True)
        if data.get("risk_status") == "Normal"
    ]

    print("\n========== GNN-BASED FAILURE-RISK MODEL ==========")
    print(f"GNN risk threshold: P(failure) >= {GNN_RISK_THRESHOLD}")
    print(f"Normal nodes: {len(normal_nodes)}")
    print(f"High-risk nodes: {len(high_risk_nodes)}")
    print(f"High-risk node IDs: {high_risk_nodes}")
    print("===================================================\n")


# ============================================================
# GNN + RL SELF-HEALING ROUTING
# ============================================================
#
# IMPORTANT: This reuses the EXISTING Q-learning RL agent code from your
# main script (train_q_learning_routing_agent, find_rl_route,
# calculate_rl_step_reward, build_route_result). Those functions are NOT
# redefined here — they must be imported from your main WSN script.
#
# This keeps the RL side of GNN-RL and LSTM-RL IDENTICAL, so the only
# difference between the two models is the source of failure_probability
# (GNN topology-aware prediction vs. LSTM temporal prediction) — exactly
# the controlled comparison the paper's hypothesis depends on.


def test_gnn_rl_self_healing_routing(graph, train_q_learning_routing_agent_fn,
                                      find_rl_route_fn, sink_node,
                                      calculate_path_latency_fn, calculate_path_failure_risk_fn,
                                      count_predicted_high_risk_nodes_in_path_fn,
                                      calculate_path_average_energy_fn, calculate_path_energy_cost_fn,
                                      calculate_path_risk_fn, count_high_risk_nodes_in_path_fn,
                                      packet_priority=1):
    """
    Test GNN + RL self-healing routing — the proposed model.

    This mirrors test_lstm_rl_self_healing_routing() from the existing
    codebase, but assigns GNN-based (topology-aware) failure probabilities
    instead of LSTM-based (temporal) ones before training/running the RL agent.

    Pass in your existing functions (from the main script) as arguments so
    this file does not need to duplicate that logic.
    """
    assign_gnn_failure_probabilities(graph)

    q_table = train_q_learning_routing_agent_fn(
        graph,
        sink_node,
        packet_priority=packet_priority
    )

    patient_nodes = [
        node for node, data in graph.nodes(data=True)
        if data["node_type"] == "patient_sensor"
    ]

    print("\n========== GNN + RL SELF-HEALING ROUTING ==========")

    routing_results = []

    for patient_node in patient_nodes:
        path = find_rl_route_fn(graph, q_table, patient_node, sink_node)

        if path is None:
            print(f"\nPatient Node {patient_node} -> Sink {sink_node}")
            print("No RL route found.")

            routing_results.append({
                "model": "GNN + RL Self-Healing",
                "patient_node": patient_node,
                "path": None,
                "delivered": False
            })
        else:
            hop_count = len(path) - 1
            latency = calculate_path_latency_fn(graph, path)
            average_failure_risk = calculate_path_failure_risk_fn(graph, path)
            predicted_high_risk_nodes = count_predicted_high_risk_nodes_in_path_fn(graph, path)
            average_energy = calculate_path_average_energy_fn(graph, path)
            energy_cost = calculate_path_energy_cost_fn(graph, path)
            average_risk = calculate_path_risk_fn(graph, path)
            high_risk_nodes = count_high_risk_nodes_in_path_fn(graph, path)

            print(f"\nPatient Node {patient_node} -> Sink {sink_node}")
            print(f"GNN + RL Path: {path}")
            print(f"Hop Count: {hop_count}")
            print(f"Total Latency: {latency} ms")
            print(f"Average GNN Failure Risk: {average_failure_risk}")
            print(f"GNN Predicted High-Risk Nodes in Path: {predicted_high_risk_nodes}")
            print(f"Average Path Energy: {average_energy}%")
            print(f"Estimated Energy Cost: {energy_cost}")
            print(f"Average Disaster Risk: {average_risk}")
            print(f"High Disaster-Risk Nodes in Path: {high_risk_nodes}")

            routing_results.append({
                "model": "GNN + RL Self-Healing",
                "patient_node": patient_node,
                "path": path,
                "delivered": True,
                "hop_count": hop_count,
                "latency": latency,
                "average_failure_risk": average_failure_risk,
                "predicted_high_risk_nodes": predicted_high_risk_nodes,
                "average_energy": average_energy,
                "energy_cost": energy_cost,
                "average_risk": average_risk,
                "high_risk_nodes": high_risk_nodes
            })

    print("\n====================================================\n")

    return routing_results


# ============================================================
# EXAMPLE USAGE (run standalone for a quick sanity check)
# ============================================================

if __name__ == "__main__":
    print("This module is designed to be imported alongside your main WSN")
    print("simulation script. Example usage:\n")
    print("  from your_main_script import create_wsn_network")
    print("  from gnn_failure_prediction import (")
    print("      train_gnn_failure_prediction_model,")
    print("      assign_gnn_failure_probabilities,")
    print("      print_gnn_failure_risk_summary,")
    print("      test_gnn_rl_self_healing_routing")
    print("  )")
    print("\n  graph = create_wsn_network()")
    print("  train_gnn_failure_prediction_model(graph)")
    print("  assign_gnn_failure_probabilities(graph)")
    print("  print_gnn_failure_risk_summary(graph)")
