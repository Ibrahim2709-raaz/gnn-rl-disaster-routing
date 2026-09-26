"""
Regression-Based GNN: Predicting Continuous Risk, Not Binary Class
======================================================================

DIAGNOSIS FROM PRIOR ABLATIONS: Both the GCN and GAT versions were trained
via binary classification (BCE loss against a thresholded 0/1 label). But
the RL agent's reward function uses the CONTINUOUS probability output
(-150 * failure_risk), not the binary class. A model can be "100% accurate"
at classification while still outputting continuous values that drift from
the formula's actual risk score -- and that drift is pure noise from the
RL agent's perspective, since there's no additional ground truth beyond
the formula for the model to learn.

FIX: train via MSE regression directly against the formula's continuous
risk score (0.0-1.0), not BCE against a thresholded label. This should
make the GNN's output track the formula much more precisely, removing
this specific source of reward-signal noise while keeping topology
awareness in the architecture.

Uses the same GCN-style aggregation as the original module (simplest,
already confirmed to work mechanically) -- the change here is entirely
in the loss function and target, not the aggregation mechanism, since
we've now tested that aggregation choice (GCN vs GAT) wasn't the issue.

HOW TO USE: same interface as the other two GNN modules --
  train_gnn_failure_prediction_model(graph)
  assign_gnn_failure_probabilities(graph)
"""

import os
import numpy as np

PYTORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    PYTORCH_AVAILABLE = False
    print("\nPyTorch is not installed.")


GNN_INPUT_FEATURES = 6   # back to the original 6 features (isolate the loss-function change specifically)
GNN_HIDDEN_DIM = 32
GNN_EPOCHS = 250
GNN_LEARNING_RATE = 0.01

GNN_RISK_THRESHOLD = 0.55
MODEL_DIR_NAME = "models"
GNN_MODEL_FILENAME = "gnn_regression_model_pytorch.pt"


def get_safe_model_dir():
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


def build_adjacency_matrix(graph):
    node_ids = list(graph.nodes())
    n = len(node_ids)
    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    A = np.zeros((n, n), dtype=np.float32)
    for node_a, node_b in graph.edges():
        i, j = id_to_idx[node_a], id_to_idx[node_b]
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + np.eye(n, dtype=np.float32)
    degrees = A.sum(axis=1)
    degrees[degrees == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return node_ids, A_norm.astype(np.float32)


def build_node_feature_matrix(graph, node_ids):
    X = np.zeros((len(node_ids), GNN_INPUT_FEATURES), dtype=np.float32)
    for i, node_id in enumerate(node_ids):
        data = graph.nodes[node_id]
        residual_energy = data.get("residual_energy", 100.0) / 100.0
        temperature_risk = data.get("temperature_risk", 0.0)
        packet_loss_rate = data.get("packet_loss_rate", 0.0)
        rssi = (data.get("rssi", -40) + 95) / 60.0
        delay = data.get("delay", 0.0) / 150.0
        forwarding_load = data.get("forwarding_load", 0.0) / 40.0
        X[i] = [residual_energy, temperature_risk, packet_loss_rate, rssi, delay, forwarding_load]
    return X


def calculate_continuous_risk_target(residual_energy_norm, temperature_risk,
                                      packet_loss_rate, rssi_norm,
                                      delay_norm, forwarding_load_norm):
    """The formula's CONTINUOUS output (0.0-1.0), used directly as the
    regression target -- no thresholding into a binary label."""
    energy_risk = 1 - residual_energy_norm
    risk = (
        0.25 * energy_risk + 0.25 * temperature_risk + 0.20 * packet_loss_rate +
        0.10 * rssi_norm + 0.10 * delay_norm + 0.10 * forwarding_load_norm
    )
    return min(max(risk, 0.0), 1.0)


def build_continuous_targets(X, node_ids, graph):
    """Continuous regression targets (not binary labels)."""
    y = np.zeros((len(node_ids), 1), dtype=np.float32)
    for i, node_id in enumerate(node_ids):
        if graph.nodes[node_id].get("node_type") == "sink":
            y[i] = 0.0
            continue
        residual_energy_norm, temperature_risk, packet_loss_rate, rssi_norm, delay_norm, forwarding_load_norm = X[i]
        y[i] = calculate_continuous_risk_target(
            residual_energy_norm, temperature_risk, packet_loss_rate,
            rssi_norm, delay_norm, forwarding_load_norm
        )
    return y


if PYTORCH_AVAILABLE:

    class GraphConvLayer(nn.Module):
        """Same GCN-style layer as the original module -- aggregation
        mechanism is unchanged; only the training objective differs."""
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, X, A_norm):
            aggregated = torch.matmul(A_norm, X)
            return self.linear(aggregated)


    class GNNRegressionPredictor(nn.Module):
        """
        Same architecture as the original GCN-based GNN, but with a
        LINEAR (not sigmoid-squashed-to-classify) output trained via MSE
        regression against the formula's continuous risk score.

        NOTE: sigmoid is still applied at output, since risk must stay in
        [0,1] -- the key change is the LOSS FUNCTION (MSE against a
        continuous target, not BCE against a binary label), not the
        output range itself.
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
            out = self.sigmoid(self.output_layer(h))  # kept in [0,1] range
            return out


def train_gnn_failure_prediction_model(graph):
    if not PYTORCH_AVAILABLE:
        print("\nCannot train GNN because PyTorch is not installed.")
        return None

    node_ids, A_norm = build_adjacency_matrix(graph)
    X = build_node_feature_matrix(graph, node_ids)
    y = build_continuous_targets(X, node_ids, graph)  # CONTINUOUS, not binary

    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = GNNRegressionPredictor()
    criterion = nn.MSELoss()  # REGRESSION loss, not BCE
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE)

    print("\n========== REGRESSION-GNN TRAINING STARTED ==========")
    print(f"Nodes: {len(node_ids)}, Epochs: {GNN_EPOCHS}, LR: {GNN_LEARNING_RATE}, Loss: MSE (regression)")
    print("=======================================================\n")

    model.train()
    for epoch in range(GNN_EPOCHS):
        optimizer.zero_grad()
        predictions = model(X_tensor, A_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                mae = torch.mean(torch.abs(predictions - y_tensor)).item()
            print(f"Epoch {epoch + 1}/{GNN_EPOCHS} | MSE Loss: {loss.item():.5f} | MAE: {mae:.5f}")

    print("\n========== REGRESSION-GNN TRAINING COMPLETE ==========\n")

    model_dir = get_safe_model_dir()
    model_path = os.path.join(model_dir, GNN_MODEL_FILENAME)
    torch.save(model.state_dict(), model_path)
    print(f"Regression-GNN model saved to: {model_path}")

    return model


def load_trained_gnn_model():
    if not PYTORCH_AVAILABLE:
        return None
    model_path = find_gnn_model_file()
    if model_path is None:
        print("\nCould not find trained regression-GNN model. Run train_gnn_failure_prediction_model() first.")
        return None
    model = GNNRegressionPredictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"\nRegression-GNN model loaded from: {model_path}")
    return model


def assign_gnn_failure_probabilities(graph, model=None):
    if model is None:
        model = load_trained_gnn_model()
    if model is None:
        print("\nRegression-GNN model could not be loaded. Cannot assign predictions.")
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
        graph.nodes[node_id]["prediction_source"] = "Regression-GNN"

        if gnn_probability >= GNN_RISK_THRESHOLD:
            graph.nodes[node_id]["risk_status"] = "High-risk"
        else:
            graph.nodes[node_id]["risk_status"] = "Normal"


def print_gnn_failure_risk_summary(graph):
    high_risk_nodes = [n for n, d in graph.nodes(data=True) if d.get("risk_status") == "High-risk"]
    normal_nodes = [n for n, d in graph.nodes(data=True) if d.get("risk_status") == "Normal"]
    print("\n========== REGRESSION-GNN FAILURE-RISK MODEL ==========")
    print(f"GNN risk threshold: P(failure) >= {GNN_RISK_THRESHOLD}")
    print(f"Normal nodes: {len(normal_nodes)}")
    print(f"High-risk nodes: {len(high_risk_nodes)}")
    print(f"High-risk node IDs: {high_risk_nodes}")
    print("=========================================================\n")


if __name__ == "__main__":
    print("This module is a drop-in replacement using regression instead of")
    print("classification. Import the same function names as the other GNN modules.")
