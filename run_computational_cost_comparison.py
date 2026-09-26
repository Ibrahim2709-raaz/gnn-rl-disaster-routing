"""
Computational Cost Comparison: LSTM vs GNN
=============================================

Measures, using your ACTUAL trained model classes (not placeholders):
  - Number of trainable parameters
  - Training time (wall-clock)
  - Inference latency per prediction (single node and full-network batch)

Both models are timed on the SAME machine, in the SAME session, so the
comparison is fair -- though the absolute numbers are specific to
whatever hardware this runs on (report your hardware in the paper).

HOW TO RUN:
    python run_computational_cost_comparison.py

Requires that both models have already been trained at least once (so
their saved weights exist in models/), OR this script will train fresh
copies of each for the timing measurement.
"""

import time
import csv
import torch
import numpy as np

from main_simulation_clean import (
    create_wsn_network,
    LSTMFailurePredictor,
    load_lstm_dataset,
    split_lstm_dataset,
    run_step_6_lstm_data_generation,
    find_lstm_dataset_file,
    LSTM_EPOCHS,
    LEARNING_RATE,
    LSTM_BATCH_SIZE,
    FEATURE_COLUMNS,
    TIME_STEPS,
    calculate_binary_classification_metrics,
)

from gnn_failure_prediction import (
    GNNFailurePredictor,
    build_adjacency_matrix,
    build_node_feature_matrix,
    build_pseudo_labels,
    GNN_EPOCHS,
    GNN_LEARNING_RATE,
)

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_lstm():
    print("\n========== BENCHMARKING LSTM ==========")

    if find_lstm_dataset_file() is None:
        print("LSTM dataset not found -- generating it now (this is normal on first run)...")
        graph = create_wsn_network()
        run_step_6_lstm_data_generation(graph)

    X, y = load_lstm_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = split_lstm_dataset(X, y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)

    model = LSTMFailurePredictor(input_size=X_train.shape[2]).to(device)
    n_params = count_parameters(model)
    print(f"LSTM trainable parameters: {n_params}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training time ---
    start = time.time()
    model.train()
    for epoch in range(LSTM_EPOCHS):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    print(f"LSTM training time ({LSTM_EPOCHS} epochs): {train_time:.2f} seconds")

    # --- Inference latency: single sequence ---
    model.eval()
    single_input = torch.rand(1, TIME_STEPS, len(FEATURE_COLUMNS)).to(device)
    with torch.no_grad():
        for _ in range(10):  # warmup
            _ = model(single_input)
        start = time.time()
        for _ in range(100):
            _ = model(single_input)
        single_latency_ms = ((time.time() - start) / 100) * 1000
    print(f"LSTM single-node inference latency: {single_latency_ms:.4f} ms")

    # --- Inference latency: batched (74 nodes at once) ---
    batch_input = torch.rand(74, TIME_STEPS, len(FEATURE_COLUMNS)).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch_input)
        start = time.time()
        for _ in range(100):
            _ = model(batch_input)
        batch_latency_ms = ((time.time() - start) / 100) * 1000
    print(f"LSTM batched (74-node) inference latency: {batch_latency_ms:.4f} ms")

    return {
        "model": "LSTM",
        "trainable_parameters": n_params,
        "training_time_sec": round(train_time, 2),
        "training_epochs": LSTM_EPOCHS,
        "single_node_inference_ms": round(single_latency_ms, 4),
        "batched_74_node_inference_ms": round(batch_latency_ms, 4),
    }


def benchmark_gnn():
    print("\n========== BENCHMARKING GNN ==========")

    graph = create_wsn_network()
    node_ids, A_norm = build_adjacency_matrix(graph)
    X = build_node_feature_matrix(graph, node_ids)
    y = build_pseudo_labels(X, node_ids, graph)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    A_tensor = torch.tensor(A_norm, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    model = GNNFailurePredictor().to(device)
    n_params = count_parameters(model)
    print(f"GNN trainable parameters: {n_params}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE)

    # --- Training time ---
    start = time.time()
    model.train()
    for epoch in range(GNN_EPOCHS):
        optimizer.zero_grad()
        predictions = model(X_tensor, A_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start
    print(f"GNN training time ({GNN_EPOCHS} epochs): {train_time:.2f} seconds")

    # --- Inference latency: full network (GNN always predicts for all nodes at once) ---
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # warmup
            _ = model(X_tensor, A_tensor)
        start = time.time()
        for _ in range(100):
            _ = model(X_tensor, A_tensor)
        full_network_latency_ms = ((time.time() - start) / 100) * 1000
    print(f"GNN full-network (74-node) inference latency: {full_network_latency_ms:.4f} ms")

    # per-node equivalent, for a fair single-node comparison against LSTM
    per_node_latency_ms = full_network_latency_ms / 74

    return {
        "model": "GNN",
        "trainable_parameters": n_params,
        "training_time_sec": round(train_time, 2),
        "training_epochs": GNN_EPOCHS,
        "single_node_inference_ms": round(per_node_latency_ms, 4),
        "batched_74_node_inference_ms": round(full_network_latency_ms, 4),
    }


def run_comparison():
    lstm_results = benchmark_lstm()
    gnn_results = benchmark_gnn()

    results = [lstm_results, gnn_results]

    with open("computational_cost_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lstm_results.keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n\n========== SUMMARY ==========")
    print(f"{'Model':<10}{'Params':<12}{'Train Time (s)':<18}{'Epochs':<10}"
          f"{'Single-node (ms)':<20}{'Batched-74 (ms)':<18}")
    for r in results:
        print(f"{r['model']:<10}{r['trainable_parameters']:<12}{r['training_time_sec']:<18}"
              f"{r['training_epochs']:<10}{r['single_node_inference_ms']:<20}{r['batched_74_node_inference_ms']:<18}")

    print(f"\nHardware used: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\nSaved: computational_cost_comparison.csv")

    return results


if __name__ == "__main__":
    run_comparison()
