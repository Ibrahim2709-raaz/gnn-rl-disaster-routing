# ============================================================
# 1. IMPORTS
# ============================================================


import random
import math
import os
import copy

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional PyTorch import
PYTORCH_AVAILABLE = True

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    PYTORCH_AVAILABLE = False
    print("\nPyTorch is not installed.")
    print("Install it using:")
    print("py -m pip install torch")


# ============================================================
# 2. GLOBAL CONFIGURATION
# ============================================================


AREA_SIZE = 100

NUM_PATIENT_NODES = 8
NUM_RELAY_NODES = 65

SINK_NODE = 0
SINK_POSITION = (8, 8)

COMMUNICATION_RANGE = 24

DISASTER_CENTER = (50, 50)
DISASTER_RADIUS = 28

PATIENT_ZONE_X = (72, 98)
PATIENT_ZONE_Y = (70, 98)

FAILURE_THRESHOLD = 0.55

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TIME_STEPS = 10
SAMPLES_PER_NODE = 40

FEATURE_COLUMNS = [
    "residual_energy",
    "temperature_risk",
    "packet_loss_rate",
    "rssi",
    "delay",
    "forwarding_load"
]

MODEL_DIR_NAME = "models"
PYTORCH_LSTM_MODEL_FILENAME = "lstm_failure_prediction_model_pytorch.pt"

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

LSTM_EPOCHS = 25
LSTM_BATCH_SIZE = 32
LEARNING_RATE = 0.001

LSTM_RISK_THRESHOLD = 0.55

RL_EPISODES = 1500
RL_MAX_STEPS = 18
RL_LEARNING_RATE = 0.12
RL_DISCOUNT_FACTOR = 0.85
RL_EPSILON = 0.35
RL_EPSILON_DECAY = 0.997
RL_MIN_EPSILON = 0.03

NORMAL_PACKET_PRIORITY = 0
CRITICAL_PACKET_PRIORITY = 1

DISASTER_SCENARIOS = {
    "Low": {"temperature_add": 0.05, "packet_loss_add": 0.03, "delay_add": 8,
            "energy_drain": 4, "load_add": 2, "rssi_drop": 2},
    "Medium": {"temperature_add": 0.15, "packet_loss_add": 0.08, "delay_add": 18,
               "energy_drain": 10, "load_add": 5, "rssi_drop": 5},
    "High": {"temperature_add": 0.28, "packet_loss_add": 0.15, "delay_add": 35,
             "energy_drain": 18, "load_add": 10, "rssi_drop": 10}
}


# ============================================================
# 3. SAFE DIRECTORY HELPERS
# ============================================================

def get_safe_results_dir():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    preferred_dir = os.path.join(script_dir, "results")
    try:
        os.makedirs(preferred_dir, exist_ok=True)
        return preferred_dir
    except PermissionError:
        home_dir = os.path.expanduser("~")
        fallback_dir = os.path.join(home_dir, "wsn_ai_routing_results")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir

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

def find_lstm_dataset_file():
    possible_paths = []
    try:
        possible_paths.append(os.path.join(RESULTS_DIR, "lstm_dataset.npz"))
    except NameError:
        pass
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    possible_paths.append(os.path.join(script_dir, "results", "lstm_dataset.npz"))
    possible_paths.append(os.path.join(os.getcwd(), "results", "lstm_dataset.npz"))
    possible_paths.append(os.path.join(os.path.expanduser("~"), "wsn_ai_routing_results", "lstm_dataset.npz"))
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def find_pytorch_lstm_model_file():
    possible_paths = []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    possible_paths.append(os.path.join(script_dir, MODEL_DIR_NAME, PYTORCH_LSTM_MODEL_FILENAME))
    possible_paths.append(os.path.join(os.getcwd(), MODEL_DIR_NAME, PYTORCH_LSTM_MODEL_FILENAME))
    possible_paths.append(os.path.join(os.path.expanduser("~"), "wsn_ai_routing_models", PYTORCH_LSTM_MODEL_FILENAME))
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

RESULTS_DIR = get_safe_results_dir()


# ============================================================
# 4. BASIC UTILITY FUNCTIONS
# ============================================================

def clip_value(value, min_value, max_value):
    return min(max(value, min_value), max_value)

def calculate_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def jitter_position(x, y, amount=5):
    return (
        min(max(x + random.uniform(-amount, amount), 0), AREA_SIZE),
        min(max(y + random.uniform(-amount, amount), 0), AREA_SIZE)
    )

def calculate_temperature_risk(position):
    distance = calculate_distance(position, DISASTER_CENTER)
    if distance >= DISASTER_RADIUS:
        return round(random.uniform(0.0, 0.25), 2)
    risk = 1 - (distance / DISASTER_RADIUS)
    risk = min(risk + random.uniform(0.10, 0.25), 1.0)
    return round(min(max(risk, 0), 1), 2)


# ============================================================
# 5. WSN NETWORK CREATION
# ============================================================

def create_wsn_network():
    graph = nx.Graph()

    graph.add_node(
        SINK_NODE, node_type="sink", position=SINK_POSITION,
        residual_energy=100.0, temperature_risk=0.0, packet_loss_rate=0.0,
        rssi=-40, delay=0.0, forwarding_load=0.0, status="active"
    )

    patient_node_ids = list(range(1, NUM_PATIENT_NODES + 1))

    for node_id in patient_node_ids:
        position = (
            random.uniform(PATIENT_ZONE_X[0], PATIENT_ZONE_X[1]),
            random.uniform(PATIENT_ZONE_Y[0], PATIENT_ZONE_Y[1])
        )
        temperature_risk = calculate_temperature_risk(position)
        residual_energy = round(random.uniform(60, 100), 2)
        packet_loss_rate = round(random.uniform(0.02, 0.12) + (temperature_risk * 0.30), 2)
        rssi = round(random.uniform(-82, -50), 2)
        delay = round(random.uniform(10, 45) + (temperature_risk * 45), 2)
        forwarding_load = round(random.uniform(1, 10), 2)

        graph.add_node(
            node_id, node_type="patient_sensor", position=position,
            residual_energy=residual_energy, temperature_risk=temperature_risk,
            packet_loss_rate=min(packet_loss_rate, 1.0), rssi=rssi, delay=delay,
            forwarding_load=forwarding_load, status="active"
        )

    planned_relay_positions = [
        (16, 16), (26, 25), (36, 35), (46, 45), (56, 55), (66, 65), (76, 75), (86, 84),
        (18, 40), (28, 55), (42, 68), (56, 78), (72, 86), (88, 90),
        (24, 14), (38, 22), (52, 30), (66, 42), (80, 58), (90, 72),
        (12, 22), (20, 8), (28, 12),
        (78, 92), (88, 80), (94, 92), (72, 72)
    ]

    relay_start_id = NUM_PATIENT_NODES + 1
    relay_id = relay_start_id

    for base_position in planned_relay_positions:
        if relay_id > NUM_PATIENT_NODES + NUM_RELAY_NODES:
            break
        position = jitter_position(base_position[0], base_position[1], amount=4)
        temperature_risk = calculate_temperature_risk(position)
        if temperature_risk >= 0.55:
            residual_energy = round(random.uniform(20, 65), 2)
            packet_loss_base = random.uniform(0.15, 0.35)
            delay_base = random.uniform(30, 75)
            load_base = random.uniform(15, 30)
            rssi = round(random.uniform(-90, -65), 2)
        else:
            residual_energy = round(random.uniform(50, 100), 2)
            packet_loss_base = random.uniform(0.02, 0.15)
            delay_base = random.uniform(8, 45)
            load_base = random.uniform(3, 20)
            rssi = round(random.uniform(-85, -45), 2)

        packet_loss_rate = round(packet_loss_base + (temperature_risk * 0.35), 2)
        delay = round(delay_base + (temperature_risk * 60), 2)
        forwarding_load = round(load_base + (temperature_risk * 10), 2)

        graph.add_node(
            relay_id, node_type="relay", position=position,
            residual_energy=residual_energy, temperature_risk=temperature_risk,
            packet_loss_rate=min(packet_loss_rate, 1.0), rssi=rssi,
            delay=min(delay, 120), forwarding_load=min(forwarding_load, 35), status="active"
        )
        relay_id += 1

    while relay_id <= NUM_PATIENT_NODES + NUM_RELAY_NODES:
        position = (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE))
        temperature_risk = calculate_temperature_risk(position)
        if temperature_risk >= 0.55:
            residual_energy = round(random.uniform(20, 65), 2)
            packet_loss_base = random.uniform(0.15, 0.35)
            delay_base = random.uniform(30, 75)
            load_base = random.uniform(15, 30)
            rssi = round(random.uniform(-90, -65), 2)
        else:
            residual_energy = round(random.uniform(50, 100), 2)
            packet_loss_base = random.uniform(0.02, 0.15)
            delay_base = random.uniform(8, 45)
            load_base = random.uniform(3, 20)
            rssi = round(random.uniform(-85, -45), 2)

        packet_loss_rate = round(packet_loss_base + (temperature_risk * 0.35), 2)
        delay = round(delay_base + (temperature_risk * 60), 2)
        forwarding_load = round(load_base + (temperature_risk * 10), 2)

        graph.add_node(
            relay_id, node_type="relay", position=position,
            residual_energy=residual_energy, temperature_risk=temperature_risk,
            packet_loss_rate=min(packet_loss_rate, 1.0), rssi=rssi,
            delay=min(delay, 120), forwarding_load=min(forwarding_load, 35), status="active"
        )
        relay_id += 1

    node_ids = list(graph.nodes())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            node_a = node_ids[i]
            node_b = node_ids[j]
            pos_a = graph.nodes[node_a]["position"]
            pos_b = graph.nodes[node_b]["position"]
            distance = calculate_distance(pos_a, pos_b)
            if distance <= COMMUNICATION_RANGE:
                if (
                    graph.nodes[node_a]["node_type"] == "sink"
                    and graph.nodes[node_b]["node_type"] == "patient_sensor"
                ) or (
                    graph.nodes[node_b]["node_type"] == "sink"
                    and graph.nodes[node_a]["node_type"] == "patient_sensor"
                ):
                    continue
                risk_a = graph.nodes[node_a]["temperature_risk"]
                risk_b = graph.nodes[node_b]["temperature_risk"]
                average_risk = (risk_a + risk_b) / 2
                distance_factor = distance / COMMUNICATION_RANGE
                link_quality = 1.0 - (0.45 * distance_factor) - (0.35 * average_risk)
                link_quality = round(min(max(link_quality, 0.1), 1.0), 2)
                graph.add_edge(node_a, node_b, distance=round(distance, 2), link_quality=link_quality)

    return graph


# ============================================================
# 6. NETWORK SUMMARY AND VALIDATION
# ============================================================

def print_network_summary(graph):
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    relay_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "relay"]
    high_disaster_risk_nodes = [n for n, d in graph.nodes(data=True) if d["temperature_risk"] >= 0.70]
    isolated_nodes = list(nx.isolates(graph))

    print("\n========== WSN NETWORK SUMMARY ==========")
    print(f"Total nodes: {total_nodes}")
    print(f"Total edges/links: {total_edges}")
    print(f"Sink/Base station node: {SINK_NODE}")
    print(f"Sink position: {graph.nodes[SINK_NODE]['position']}")
    print(f"Patient sensor nodes: {patient_nodes}")
    print(f"Number of relay nodes: {len(relay_nodes)}")
    print(f"Nodes inside high disaster risk zone: {high_disaster_risk_nodes}")
    print(f"Number of isolated nodes: {len(isolated_nodes)}")
    print(f"Isolated node IDs: {isolated_nodes}")
    print("=========================================\n")

def validate_network_connectivity(graph):
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    print("\n========== CONNECTIVITY CHECK ==========")
    reachable_count = 0
    for patient_node in patient_nodes:
        if nx.has_path(graph, patient_node, SINK_NODE):
            path = nx.shortest_path(graph, source=patient_node, target=SINK_NODE)
            print(f"Patient Node {patient_node} can reach sink. Shortest path length: {len(path) - 1} hops")
            reachable_count += 1
        else:
            print(f"Patient Node {patient_node} cannot reach sink.")
    print(f"\nReachable patient nodes: {reachable_count}/{len(patient_nodes)}")
    print("========================================\n")

def print_sample_node_data(graph, number_of_nodes=10):
    print("========== SAMPLE NODE DATA ==========")
    for node_id in list(graph.nodes())[:number_of_nodes]:
        data = graph.nodes[node_id]
        print(f"\nNode {node_id}")
        print(f"Type: {data['node_type']}")
        print(f"Position: {data['position']}")
        print(f"Residual Energy: {data['residual_energy']}%")
        print(f"Temperature/Disaster Risk: {data['temperature_risk']}")
        print(f"Packet Loss Rate: {data['packet_loss_rate']}")
        print(f"RSSI: {data['rssi']} dBm")
        print(f"Delay: {data['delay']} ms")
        print(f"Forwarding Load: {data['forwarding_load']} packets/sec")
        print(f"Status: {data['status']}")
    print("\n======================================\n")


# ============================================================
# 7. VISUALIZATION FUNCTIONS
# ============================================================

def get_node_groups(graph):
    sink_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "sink"]
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    relay_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "relay"]
    high_disaster_risk_nodes = [n for n, d in graph.nodes(data=True) if d["temperature_risk"] >= 0.70]
    predicted_high_risk_nodes = [n for n, d in graph.nodes(data=True) if d.get("risk_status") == "High-risk"]
    return sink_nodes, patient_nodes, relay_nodes, high_disaster_risk_nodes, predicted_high_risk_nodes

def visualize_network(graph, title="Realistic WSN Healthcare Monitoring Network Simulation"):
    positions = nx.get_node_attributes(graph, "position")
    sink_nodes, patient_nodes, relay_nodes, high_disaster_risk_nodes, _ = get_node_groups(graph)
    plt.figure(figsize=(11, 8))
    disaster_circle = plt.Circle(DISASTER_CENTER, DISASTER_RADIUS, fill=False, linestyle="--", linewidth=2, label="Disaster / Failure Zone")
    plt.gca().add_patch(disaster_circle)
    nx.draw_networkx_edges(graph, positions, alpha=0.25)
    nx.draw_networkx_nodes(graph, positions, nodelist=relay_nodes, node_size=120, label="Relay Nodes")
    nx.draw_networkx_nodes(graph, positions, nodelist=patient_nodes, node_size=190, node_shape="s", label="Patient Sensor Nodes")
    nx.draw_networkx_nodes(graph, positions, nodelist=sink_nodes, node_size=320, node_shape="*", label="Sink / Base Station")
    nx.draw_networkx_nodes(graph, positions, nodelist=high_disaster_risk_nodes, node_size=230, linewidths=2, edgecolors="black", label="High Disaster Risk Nodes")
    nx.draw_networkx_labels(graph, positions, font_size=8)
    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(0, AREA_SIZE)
    plt.ylim(0, AREA_SIZE)
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_path(graph, path, title, path_label):
    if path is None:
        print("No path available to visualize.")
        return
    positions = nx.get_node_attributes(graph, "position")
    sink_nodes, patient_nodes, relay_nodes, high_disaster_risk_nodes, predicted_high_risk_nodes = get_node_groups(graph)
    path_edges = list(zip(path, path[1:]))
    plt.figure(figsize=(11, 8))
    disaster_circle = plt.Circle(DISASTER_CENTER, DISASTER_RADIUS, fill=False, linestyle="--", linewidth=2, label="Disaster / Failure Zone")
    plt.gca().add_patch(disaster_circle)
    nx.draw_networkx_edges(graph, positions, alpha=0.18)
    nx.draw_networkx_edges(graph, positions, edgelist=path_edges, width=3, label=path_label)
    nx.draw_networkx_nodes(graph, positions, nodelist=relay_nodes, node_size=120, label="Relay Nodes")
    nx.draw_networkx_nodes(graph, positions, nodelist=patient_nodes, node_size=190, node_shape="s", label="Patient Sensor Nodes")
    nx.draw_networkx_nodes(graph, positions, nodelist=sink_nodes, node_size=320, node_shape="*", label="Sink / Base Station")
    nx.draw_networkx_nodes(graph, positions, nodelist=high_disaster_risk_nodes, node_size=230, linewidths=2, edgecolors="black", label="High Disaster Risk Nodes")
    if predicted_high_risk_nodes:
        nx.draw_networkx_nodes(graph, positions, nodelist=predicted_high_risk_nodes, node_size=250, linewidths=2, edgecolors="black", label="Predicted High-Risk Nodes")
    nx.draw_networkx_nodes(graph, positions, nodelist=path, node_size=330, linewidths=2, edgecolors="black", label="Nodes in Selected Path")
    nx.draw_networkx_labels(graph, positions, font_size=8)
    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(0, AREA_SIZE)
    plt.ylim(0, AREA_SIZE)
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 8. ROUTE METRIC FUNCTIONS
# ============================================================

def calculate_path_latency(graph, path):
    if path is None:
        return None
    total_latency = 0
    for node in path:
        total_latency += graph.nodes[node]["delay"]
    return round(total_latency, 2)

def calculate_path_risk(graph, path):
    if path is None:
        return None
    total_risk = 0
    for node in path:
        total_risk += graph.nodes[node]["temperature_risk"]
    return round(total_risk / len(path), 2)

def count_high_risk_nodes_in_path(graph, path):
    if path is None:
        return None
    count = 0
    for node in path:
        if graph.nodes[node]["temperature_risk"] >= 0.70:
            count += 1
    return count

def calculate_path_average_energy(graph, path):
    if path is None:
        return None
    total_energy = 0
    for node in path:
        total_energy += graph.nodes[node]["residual_energy"]
    return round(total_energy / len(path), 2)

def calculate_path_energy_cost(graph, path):
    if path is None:
        return None
    total_cost = 0
    for node_a, node_b in zip(path, path[1:]):
        distance = graph.edges[node_a, node_b]["distance"]
        link_quality = graph.edges[node_a, node_b]["link_quality"]
        transmission_cost = 0.05 * distance
        poor_link_penalty = (1 - link_quality) * 1.5
        processing_cost = 0.10
        total_cost += transmission_cost + poor_link_penalty + processing_cost
    return round(total_cost, 2)

def calculate_path_failure_risk(graph, path):
    if path is None:
        return None
    total_failure_risk = 0
    for node in path:
        total_failure_risk += graph.nodes[node].get("failure_probability", 0)
    return round(total_failure_risk / len(path), 2)

def count_predicted_high_risk_nodes_in_path(graph, path):
    if path is None:
        return None
    count = 0
    for node in path:
        if graph.nodes[node].get("risk_status") == "High-risk":
            count += 1
    return count


# ============================================================
# 9. SHORTEST PATH ROUTING BASELINE
# ============================================================

def find_shortest_path(graph, source_node, sink_node=SINK_NODE):
    try:
        return nx.shortest_path(graph, source=source_node, target=sink_node)
    except nx.NetworkXNoPath:
        return None

def test_shortest_path_routing(graph):
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    print("\n========== SHORTEST PATH ROUTING BASELINE ==========")
    routing_results = []
    for patient_node in patient_nodes:
        path = find_shortest_path(graph, patient_node, SINK_NODE)
        if path is None:
            print(f"\nPatient Node {patient_node} -> Sink {SINK_NODE}")
            print("No available path.")
            routing_results.append({"model": "Shortest Path", "patient_node": patient_node, "path": None, "delivered": False})
        else:
            hop_count = len(path) - 1
            latency = calculate_path_latency(graph, path)
            average_risk = calculate_path_risk(graph, path)
            high_risk_nodes = count_high_risk_nodes_in_path(graph, path)
            average_energy = calculate_path_average_energy(graph, path)
            energy_cost = calculate_path_energy_cost(graph, path)
            print(f"\nPatient Node {patient_node} -> Sink {SINK_NODE}")
            print(f"Path: {path}")
            print(f"Hop Count: {hop_count}")
            print(f"Total Latency: {latency} ms")
            print(f"Average Path Disaster Risk: {average_risk}")
            print(f"High-Risk Nodes in Path: {high_risk_nodes}")
            print(f"Average Path Energy: {average_energy}%")
            print(f"Estimated Energy Cost: {energy_cost}")
            routing_results.append({"model": "Shortest Path", "patient_node": patient_node, "path": path, "delivered": True,
                "hop_count": hop_count, "latency": latency, "average_risk": average_risk,
                "high_risk_nodes": high_risk_nodes, "average_energy": average_energy, "energy_cost": energy_cost})
    print("\n====================================================\n")
    return routing_results


# ============================================================
# 10. ENERGY-AWARE ROUTING BASELINE
# ============================================================

def calculate_energy_aware_edge_weight(graph, node_a, node_b):
    energy_a = graph.nodes[node_a]["residual_energy"]
    energy_b = graph.nodes[node_b]["residual_energy"]
    average_energy = (energy_a + energy_b) / 2
    energy_cost = 100 - average_energy
    distance = graph.edges[node_a, node_b]["distance"]
    total_weight = energy_cost + (0.35 * distance)
    return round(total_weight, 2)

def assign_energy_aware_weights(graph):
    for node_a, node_b in graph.edges():
        graph.edges[node_a, node_b]["energy_aware_weight"] = calculate_energy_aware_edge_weight(graph, node_a, node_b)

def find_energy_aware_path(graph, source_node, sink_node=SINK_NODE):
    try:
        return nx.shortest_path(graph, source=source_node, target=sink_node, weight="energy_aware_weight")
    except nx.NetworkXNoPath:
        return None

def test_energy_aware_routing(graph):
    assign_energy_aware_weights(graph)
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    print("\n========== ENERGY-AWARE ROUTING BASELINE ==========")
    routing_results = []
    for patient_node in patient_nodes:
        path = find_energy_aware_path(graph, patient_node, SINK_NODE)
        if path is None:
            print(f"\nPatient Node {patient_node} -> Sink {SINK_NODE}")
            print("No available path.")
            routing_results.append({"model": "Energy-Aware", "patient_node": patient_node, "path": None, "delivered": False})
        else:
            hop_count = len(path) - 1
            latency = calculate_path_latency(graph, path)
            average_energy = calculate_path_average_energy(graph, path)
            energy_cost = calculate_path_energy_cost(graph, path)
            average_risk = calculate_path_risk(graph, path)
            high_risk_nodes = count_high_risk_nodes_in_path(graph, path)
            print(f"\nPatient Node {patient_node} -> Sink {SINK_NODE}")
            print(f"Energy-Aware Path: {path}")
            print(f"Hop Count: {hop_count}")
            print(f"Total Latency: {latency} ms")
            print(f"Average Path Energy: {average_energy}%")
            print(f"Estimated Energy Cost: {energy_cost}")
            print(f"Average Path Disaster Risk: {average_risk}")
            print(f"High-Risk Nodes in Path: {high_risk_nodes}")
            routing_results.append({"model": "Energy-Aware", "patient_node": patient_node, "path": path, "delivered": True,
                "hop_count": hop_count, "latency": latency, "average_energy": average_energy,
                "energy_cost": energy_cost, "average_risk": average_risk, "high_risk_nodes": high_risk_nodes})
    print("\n===================================================\n")
    return routing_results


# ============================================================
# 11. WEIGHTED FAILURE-RISK MODEL
# ============================================================

def normalize_rssi_risk(rssi):
    rssi_risk = (abs(rssi) - 40) / 50
    return min(max(rssi_risk, 0), 1)

def normalize_delay_risk(delay):
    delay_risk = delay / 120
    return min(max(delay_risk, 0), 1)

def normalize_load_risk(forwarding_load):
    load_risk = forwarding_load / 35
    return min(max(load_risk, 0), 1)

def calculate_weighted_failure_probability(graph, node_id):
    node = graph.nodes[node_id]
    residual_energy = node["residual_energy"]
    temperature_risk = node["temperature_risk"]
    packet_loss_rate = node["packet_loss_rate"]
    rssi = node["rssi"]
    delay = node["delay"]
    forwarding_load = node["forwarding_load"]
    energy_risk = 1 - (residual_energy / 100)
    rssi_risk = normalize_rssi_risk(rssi)
    delay_risk = normalize_delay_risk(delay)
    load_risk = normalize_load_risk(forwarding_load)
    failure_probability = (0.25 * energy_risk + 0.25 * temperature_risk + 0.20 * packet_loss_rate +
        0.10 * rssi_risk + 0.10 * delay_risk + 0.10 * load_risk)
    return round(min(max(failure_probability, 0), 1), 2)

def assign_failure_probabilities(graph):
    for node_id in graph.nodes():
        if graph.nodes[node_id]["node_type"] == "sink":
            graph.nodes[node_id]["failure_probability"] = 0.0
            graph.nodes[node_id]["risk_status"] = "Sink"
            continue
        failure_probability = calculate_weighted_failure_probability(graph, node_id)
        graph.nodes[node_id]["failure_probability"] = failure_probability
        if failure_probability >= FAILURE_THRESHOLD:
            graph.nodes[node_id]["risk_status"] = "High-risk"
        else:
            graph.nodes[node_id]["risk_status"] = "Normal"

def print_failure_risk_summary(graph):
    high_risk_nodes = [n for n, d in graph.nodes(data=True) if d.get("risk_status") == "High-risk"]
    normal_nodes = [n for n, d in graph.nodes(data=True) if d.get("risk_status") == "Normal"]
    print("\n========== WEIGHTED FAILURE-RISK MODEL ==========")
    print(f"Failure threshold: P(failure) >= {FAILURE_THRESHOLD}")
    print(f"Normal nodes: {len(normal_nodes)}")
    print(f"High-risk nodes: {len(high_risk_nodes)}")
    print(f"High-risk node IDs: {high_risk_nodes}")
    print("=================================================\n")

def print_sample_failure_probabilities(graph, number_of_nodes=10):
    print("========== SAMPLE FAILURE PROBABILITIES ==========")
    count = 0
    for node_id, data in graph.nodes(data=True):
        if data["node_type"] == "sink":
            continue
        print(f"\nNode {node_id}")
        print(f"Type: {data['node_type']}")
        print(f"Residual Energy: {data['residual_energy']}%")
        print(f"Temperature Risk: {data['temperature_risk']}")
        print(f"Packet Loss Rate: {data['packet_loss_rate']}")
        print(f"RSSI: {data['rssi']} dBm")
        print(f"Delay: {data['delay']} ms")
        print(f"Forwarding Load: {data['forwarding_load']}")
        print(f"P(failure): {data['failure_probability']}")
        print(f"Risk Status: {data['risk_status']}")
        count += 1
        if count >= number_of_nodes:
            break
    print("\n==================================================\n")


# ============================================================
# 12. FAILURE-RISK-AWARE ROUTING
# ============================================================

def calculate_failure_aware_edge_weight(graph, node_a, node_b):
    failure_a = graph.nodes[node_a].get("failure_probability", 0)
    failure_b = graph.nodes[node_b].get("failure_probability", 0)
    average_failure_risk = (failure_a + failure_b) / 2
    distance = graph.edges[node_a, node_b]["distance"]
    link_quality = graph.edges[node_a, node_b]["link_quality"]
    risk_cost = average_failure_risk * 120
    distance_cost = 0.30 * distance
    poor_link_cost = (1 - link_quality) * 20
    total_weight = risk_cost + distance_cost + poor_link_cost
    return round(total_weight, 2)

def assign_failure_aware_weights(graph):
    for node_a, node_b in graph.edges():
        graph.edges[node_a, node_b]["failure_aware_weight"] = calculate_failure_aware_edge_weight(graph, node_a, node_b)

def find_failure_aware_path(graph, source_node, sink_node=SINK_NODE):
    try:
        return nx.shortest_path(graph, source=source_node, target=sink_node, weight="failure_aware_weight")
    except nx.NetworkXNoPath:
        return None

def test_failure_aware_routing(graph):
    assign_failure_probabilities(graph)
    assign_failure_aware_weights(graph)
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    print("\n========== FAILURE-RISK-AWARE ROUTING ==========")
    routing_results = []
    for patient_node in patient_nodes:
        path = find_failure_aware_path(graph, patient_node, SINK_NODE)
        if path is None:
            print(f"\nPatient Node {patient_node} -> Sink {SINK_NODE}")
            print("No available path.")
            routing_results.append({"model": "Failure-Aware", "patient_node": patient_node, "path": None, "delivered": False})
        else:
            hop_count = len(path) - 1
            latency = calculate_path_latency(graph, path)
            average_failure_risk = calculate_path_failure_risk(graph, path)
            predicted_high_risk_nodes = count_predicted_high_risk_nodes_in_path(graph, path)
            average_energy = calculate_path_average_energy(graph, path)
            energy_cost = calculate_path_energy_cost(graph, path)
            average_risk = calculate_path_risk(graph, path)
            high_risk_nodes = count_high_risk_nodes_in_path(graph, path)
            print(f"\nPatient Node {patient_node} -> Sink {SINK_NODE}")
            print(f"Failure-Aware Path: {path}")
            print(f"Hop Count: {hop_count}")
            print(f"Total Latency: {latency} ms")
            print(f"Average Failure Risk: {average_failure_risk}")
            print(f"Predicted High-Risk Nodes in Path: {predicted_high_risk_nodes}")
            print(f"Average Path Energy: {average_energy}%")
            print(f"Estimated Energy Cost: {energy_cost}")
            print(f"Average Disaster Risk: {average_risk}")
            print(f"High Disaster-Risk Nodes in Path: {high_risk_nodes}")
            routing_results.append({"model": "Failure-Aware", "patient_node": patient_node, "path": path, "delivered": True,
                "hop_count": hop_count, "latency": latency, "average_failure_risk": average_failure_risk,
                "predicted_high_risk_nodes": predicted_high_risk_nodes, "average_energy": average_energy,
                "energy_cost": energy_cost, "average_risk": average_risk, "high_risk_nodes": high_risk_nodes})
    print("\n================================================\n")
    return routing_results


# ============================================================
# 13. RESULT SUMMARY
# ============================================================

def summarize_routing_results(results, model_name):
    delivered_results = [r for r in results if r.get("delivered")]
    if not delivered_results:
        print(f"\n{model_name}: No packets delivered.")
        return
    pdr = len(delivered_results) / len(results)
    avg_latency = sum(r["latency"] for r in delivered_results) / len(delivered_results)
    avg_hops = sum(r["hop_count"] for r in delivered_results) / len(delivered_results)
    avg_energy = sum(r["average_energy"] for r in delivered_results) / len(delivered_results)
    avg_energy_cost = sum(r["energy_cost"] for r in delivered_results) / len(delivered_results)
    avg_disaster_risk = sum(r["average_risk"] for r in delivered_results) / len(delivered_results)
    avg_high_risk_nodes = sum(r["high_risk_nodes"] for r in delivered_results) / len(delivered_results)
    print(f"\n========== {model_name} SUMMARY ==========")
    print(f"Packet Delivery Ratio: {pdr:.2f}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Average Hop Count: {avg_hops:.2f}")
    print(f"Average Path Energy: {avg_energy:.2f}%")
    print(f"Average Energy Cost: {avg_energy_cost:.2f}")
    print(f"Average Disaster Risk: {avg_disaster_risk:.2f}")
    print(f"Average High-Risk Nodes in Path: {avg_high_risk_nodes:.2f}")
    print("==========================================\n")


# ============================================================
# 14. STEP 6: SYNTHETIC LSTM DATA GENERATION
# ============================================================

def calculate_failure_probability_from_features(residual_energy, temperature_risk, packet_loss_rate, rssi, delay, forwarding_load):
    energy_risk = 1 - (residual_energy / 100)
    rssi_risk = normalize_rssi_risk(rssi)
    delay_risk = normalize_delay_risk(delay)
    load_risk = normalize_load_risk(forwarding_load)
    failure_probability = (0.25 * energy_risk + 0.25 * temperature_risk + 0.20 * packet_loss_rate +
        0.10 * rssi_risk + 0.10 * delay_risk + 0.10 * load_risk)
    return round(min(max(failure_probability, 0), 1), 3)

def generate_node_time_series_sample(node_data, scenario_type):
    sequence = []
    residual_energy = node_data["residual_energy"]
    temperature_risk = node_data["temperature_risk"]
    packet_loss_rate = node_data["packet_loss_rate"]
    rssi = node_data["rssi"]
    delay = node_data["delay"]
    forwarding_load = node_data["forwarding_load"]

    for t in range(TIME_STEPS):
        if scenario_type == "stable":
            residual_energy -= random.uniform(0.05, 0.40)
            temperature_risk += random.uniform(-0.01, 0.02)
            packet_loss_rate += random.uniform(-0.01, 0.01)
            rssi += random.uniform(-1.0, 1.0)
            delay += random.uniform(-2.0, 2.0)
            forwarding_load += random.uniform(-0.5, 0.5)
        elif scenario_type == "degrading":
            residual_energy -= random.uniform(0.50, 1.80)
            temperature_risk += random.uniform(0.00, 0.04)
            packet_loss_rate += random.uniform(0.00, 0.03)
            rssi -= random.uniform(0.50, 2.00)
            delay += random.uniform(1.00, 4.00)
            forwarding_load += random.uniform(0.20, 1.50)
        elif scenario_type == "disaster_hit":
            residual_energy -= random.uniform(1.00, 3.00)
            temperature_risk += random.uniform(0.03, 0.08)
            packet_loss_rate += random.uniform(0.02, 0.06)
            rssi -= random.uniform(1.00, 3.00)
            delay += random.uniform(3.00, 8.00)
            forwarding_load += random.uniform(1.00, 3.00)

        residual_energy = clip_value(residual_energy, 0, 100)
        temperature_risk = clip_value(temperature_risk, 0, 1)
        packet_loss_rate = clip_value(packet_loss_rate, 0, 1)
        rssi = clip_value(rssi, -95, -35)
        delay = clip_value(delay, 0, 150)
        forwarding_load = clip_value(forwarding_load, 0, 40)

        sequence.append([residual_energy, temperature_risk, packet_loss_rate, rssi, delay, forwarding_load])

    final_features = sequence[-1]
    final_failure_probability = calculate_failure_probability_from_features(
        residual_energy=final_features[0], temperature_risk=final_features[1],
        packet_loss_rate=final_features[2], rssi=final_features[3],
        delay=final_features[4], forwarding_load=final_features[5])
    label = 1 if final_failure_probability >= FAILURE_THRESHOLD else 0
    return sequence, label, final_failure_probability

def generate_lstm_training_data(graph):
    X = []
    y = []
    metadata_rows = []
    node_ids = [node_id for node_id, data in graph.nodes(data=True) if data["node_type"] != "sink"]
    scenario_options = ["stable", "degrading", "disaster_hit"]

    for node_id in node_ids:
        node_data = graph.nodes[node_id]
        for sample_id in range(SAMPLES_PER_NODE):
            scenario_type = random.choice(scenario_options)
            sequence, label, final_failure_probability = generate_node_time_series_sample(node_data, scenario_type)
            X.append(sequence)
            y.append(label)
            for time_step, feature_values in enumerate(sequence):
                metadata_rows.append({
                    "node_id": node_id, "sample_id": sample_id, "time_step": time_step,
                    "scenario_type": scenario_type,
                    "residual_energy": round(feature_values[0], 3), "temperature_risk": round(feature_values[1], 3),
                    "packet_loss_rate": round(feature_values[2], 3), "rssi": round(feature_values[3], 3),
                    "delay": round(feature_values[4], 3), "forwarding_load": round(feature_values[5], 3),
                    "final_failure_probability": final_failure_probability, "failure_label": label
                })

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    metadata_df = pd.DataFrame(metadata_rows)
    return X, y, metadata_df

def normalize_lstm_features(X):
    X_normalized = X.copy()
    X_normalized[:, :, 0] = X_normalized[:, :, 0] / 100.0
    X_normalized[:, :, 1] = X_normalized[:, :, 1]
    X_normalized[:, :, 2] = X_normalized[:, :, 2]
    X_normalized[:, :, 3] = (X_normalized[:, :, 3] + 95) / 60.0
    X_normalized[:, :, 4] = X_normalized[:, :, 4] / 150.0
    X_normalized[:, :, 5] = X_normalized[:, :, 5] / 40.0
    return X_normalized

def save_lstm_dataset(X, y, metadata_df):
    X_normalized = normalize_lstm_features(X)
    csv_path = os.path.join(RESULTS_DIR, "lstm_synthetic_timeseries_data.csv")
    npz_path = os.path.join(RESULTS_DIR, "lstm_dataset.npz")
    try:
        metadata_df.to_csv(csv_path, index=False)
        np.savez(npz_path, X_raw=X, X_normalized=X_normalized, y=y, feature_columns=np.array(FEATURE_COLUMNS))
    except PermissionError:
        fallback_dir = os.path.join(os.path.expanduser("~"), "wsn_ai_routing_results")
        os.makedirs(fallback_dir, exist_ok=True)
        csv_path = os.path.join(fallback_dir, "lstm_synthetic_timeseries_data.csv")
        npz_path = os.path.join(fallback_dir, "lstm_dataset.npz")
        metadata_df.to_csv(csv_path, index=False)
        np.savez(npz_path, X_raw=X, X_normalized=X_normalized, y=y, feature_columns=np.array(FEATURE_COLUMNS))

    print("\n========== LSTM DATASET GENERATED ==========")
    print(f"Raw X shape: {X.shape}")
    print(f"Normalized X shape: {X_normalized.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature columns: {FEATURE_COLUMNS}")
    print(f"CSV saved to: {csv_path}")
    print(f"NPZ saved to: {npz_path}")
    total_samples = len(y)
    failure_samples = int(np.sum(y))
    normal_samples = total_samples - failure_samples
    print(f"Total samples: {total_samples}")
    print(f"Normal samples: {normal_samples}")
    print(f"Failure-risk samples: {failure_samples}")
    if total_samples > 0:
        print(f"Failure-risk percentage: {(failure_samples / total_samples) * 100:.2f}%")
    print("===========================================\n")

def run_step_6_lstm_data_generation(graph):
    X, y, metadata_df = generate_lstm_training_data(graph)
    save_lstm_dataset(X, y, metadata_df)
    print("Sample rows from generated LSTM dataset:")
    print(metadata_df.head(15))


# ============================================================
# 15. STEP 7: PYTORCH LSTM TRAINING
# ============================================================

def load_lstm_dataset():
    dataset_path = find_lstm_dataset_file()
    if dataset_path is None:
        raise FileNotFoundError("Could not find lstm_dataset.npz. Run Step 6 first to generate the dataset.")
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X_normalized"].astype(np.float32)
    y = data["y"].astype(np.float32)
    print("\n========== LSTM DATASET LOADED ==========")
    print(f"Dataset path: {dataset_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Normal samples: {np.sum(y == 0)}")
    print(f"Failure-risk samples: {np.sum(y == 1)}")
    print("=========================================\n")
    return X, y

def split_lstm_dataset(X, y):
    total_samples = len(X)
    indices = np.arange(total_samples)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    train_end = int(total_samples * TRAIN_SPLIT)
    validation_end = int(total_samples * (TRAIN_SPLIT + VALIDATION_SPLIT))
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_validation = X[train_end:validation_end]
    y_validation = y[train_end:validation_end]
    X_test = X[validation_end:]
    y_test = y[validation_end:]
    print("\n========== DATA SPLIT ==========")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Testing samples: {len(X_test)}")
    print("===============================\n")
    return X_train, y_train, X_validation, y_validation, X_test, y_test

_LSTM_BASE_CLASS = nn.Module if PYTORCH_AVAILABLE else object


class LSTMFailurePredictor(_LSTM_BASE_CLASS):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout_rate=0.30):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to create an LSTMFailurePredictor.")
        super(LSTMFailurePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        last_time_step_output = lstm_output[:, -1, :]
        x = self.dropout(last_time_step_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def calculate_binary_classification_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (true_positive + true_negative) / len(y_true)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score,
        "true_positive": int(true_positive), "true_negative": int(true_negative),
        "false_positive": int(false_positive), "false_negative": int(false_negative)}

def train_lstm_failure_prediction_model():
    if not PYTORCH_AVAILABLE:
        print("\nCannot train LSTM because PyTorch is not installed.")
        return None
    X, y = load_lstm_dataset()
    X_train, y_train, X_validation, y_validation, X_test, y_test = split_lstm_dataset(X, y)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    input_size = X_train.shape[2]
    model = LSTMFailurePredictor(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("\n========== PYTORCH LSTM TRAINING STARTED ==========")
    print(model)
    print("===================================================\n")
    best_validation_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_model_state = None

    for epoch in range(LSTM_EPOCHS):
        model.train()
        training_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        average_training_loss = training_loss / len(train_loader)
        model.eval()
        with torch.no_grad():
            validation_outputs = model(X_validation_tensor)
            validation_loss = criterion(validation_outputs, y_validation_tensor).item()
            validation_predictions = (validation_outputs.numpy().flatten() >= 0.50).astype(int)
            validation_metrics = calculate_binary_classification_metrics(y_validation, validation_predictions)
        print(f"Epoch {epoch + 1}/{LSTM_EPOCHS} | Train Loss: {average_training_loss:.4f} | "
              f"Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_metrics['accuracy']:.4f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_probabilities = test_outputs.numpy().flatten()
        test_predictions = (test_probabilities >= 0.50).astype(int)
    metrics = calculate_binary_classification_metrics(y_test, test_predictions)
    print("\n========== PYTORCH LSTM TEST RESULTS ==========")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("==============================================\n")
    model_dir = get_safe_model_dir()
    model_path = os.path.join(model_dir, PYTORCH_LSTM_MODEL_FILENAME)
    try:
        torch.save(model.state_dict(), model_path)
    except PermissionError:
        fallback_dir = os.path.join(os.path.expanduser("~"), "wsn_ai_routing_models")
        os.makedirs(fallback_dir, exist_ok=True)
        model_path = os.path.join(fallback_dir, PYTORCH_LSTM_MODEL_FILENAME)
        torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}\n")
    return model


# ============================================================
# 16. STEP 8: LSTM FAILURE PREDICTION AND ROUTING
# ============================================================

def load_trained_lstm_model():
    if not PYTORCH_AVAILABLE:
        return None
    model_path = find_pytorch_lstm_model_file()
    if model_path is None:
        print("\nCould not find trained LSTM model. Run Step 7 first.")
        return None
    model = LSTMFailurePredictor(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"\nModel loaded from: {model_path}\n")
    return model

def create_lstm_sequence_from_current_node(node_data):
    current_energy = node_data["residual_energy"]
    current_temperature = node_data["temperature_risk"]
    current_packet_loss = node_data["packet_loss_rate"]
    current_rssi = node_data["rssi"]
    current_delay = node_data["delay"]
    current_load = node_data["forwarding_load"]
    sequence = []
    for t in range(TIME_STEPS):
        progress = t / (TIME_STEPS - 1)
        energy = current_energy + ((1 - progress) * random.uniform(2.0, 8.0))
        temperature = current_temperature - ((1 - progress) * random.uniform(0.01, 0.08))
        packet_loss = current_packet_loss - ((1 - progress) * random.uniform(0.01, 0.06))
        rssi = current_rssi + ((1 - progress) * random.uniform(1.0, 6.0))
        delay = current_delay - ((1 - progress) * random.uniform(2.0, 12.0))
        load = current_load - ((1 - progress) * random.uniform(0.5, 4.0))
        energy = clip_value(energy, 0, 100)
        temperature = clip_value(temperature, 0, 1)
        packet_loss = clip_value(packet_loss, 0, 1)
        rssi = clip_value(rssi, -95, -35)
        delay = clip_value(delay, 0, 150)
        load = clip_value(load, 0, 40)
        sequence.append([energy, temperature, packet_loss, rssi, delay, load])
    sequence = np.array(sequence, dtype=np.float32)
    return sequence

def normalize_single_lstm_sequence(sequence):
    sequence = sequence.copy()
    sequence[:, 0] = sequence[:, 0] / 100.0
    sequence[:, 3] = (sequence[:, 3] + 95) / 60.0
    sequence[:, 4] = sequence[:, 4] / 150.0
    sequence[:, 5] = sequence[:, 5] / 40.0
    sequence = np.expand_dims(sequence, axis=0)
    return sequence.astype(np.float32)

def predict_node_failure_with_lstm(model, node_data):
    sequence = create_lstm_sequence_from_current_node(node_data)
    normalized_sequence = normalize_single_lstm_sequence(sequence)
    input_tensor = torch.tensor(normalized_sequence, dtype=torch.float32)
    with torch.no_grad():
        probability = model(input_tensor).item()
    return float(probability)

def assign_lstm_failure_probabilities(graph):
    model = load_trained_lstm_model()
    if model is None:
        print("\nLSTM model could not be loaded. Falling back to weighted failure probabilities.")
        assign_failure_probabilities(graph)
        return
    for node_id in graph.nodes():
        if graph.nodes[node_id]["node_type"] == "sink":
            graph.nodes[node_id]["failure_probability"] = 0.0
            graph.nodes[node_id]["risk_status"] = "Sink"
            graph.nodes[node_id]["prediction_source"] = "sink"
            continue
        node_data = graph.nodes[node_id]
        lstm_probability = predict_node_failure_with_lstm(model, node_data)
        graph.nodes[node_id]["failure_probability"] = lstm_probability
        graph.nodes[node_id]["prediction_source"] = "LSTM"
        if lstm_probability >= LSTM_RISK_THRESHOLD:
            graph.nodes[node_id]["risk_status"] = "High-risk"
        else:
            graph.nodes[node_id]["risk_status"] = "Normal"

def print_lstm_failure_risk_summary(graph):
    high_risk_nodes = [n for n, d in graph.nodes(data=True) if d.get("risk_status") == "High-risk"]
    normal_nodes = [n for n, d in graph.nodes(data=True) if d.get("risk_status") == "Normal"]
    print("\n========== LSTM-BASED FAILURE-RISK MODEL ==========")
    print(f"LSTM risk threshold: P(failure) >= {LSTM_RISK_THRESHOLD}")
    print(f"Normal nodes: {len(normal_nodes)}")
    print(f"High-risk nodes: {len(high_risk_nodes)}")
    print(f"High-risk node IDs: {high_risk_nodes}")
    print("===================================================\n")

def print_sample_lstm_predictions(graph, number_of_nodes=10):
    print("========== SAMPLE LSTM FAILURE PREDICTIONS ==========")
    count = 0
    for node_id, data in graph.nodes(data=True):
        if data["node_type"] == "sink":
            continue
        print(f"\nNode {node_id}")
        print(f"LSTM P(failure): {data['failure_probability']:.4f}")
        print(f"Risk Status: {data['risk_status']}")
        count += 1
        if count >= number_of_nodes:
            break
    print("\n=====================================================\n")

def print_lstm_high_risk_predictions(graph):
    print("\n========== LSTM HIGH-RISK NODE PREDICTIONS ==========")
    high_risk_nodes = [(n, d) for n, d in graph.nodes(data=True) if d.get("risk_status") == "High-risk"]
    for node_id, data in high_risk_nodes:
        print(f"\nNode {node_id} | LSTM P(failure): {data['failure_probability']:.4f}")
    print("\n=====================================================\n")

def test_lstm_based_failure_aware_routing(graph):
    assign_failure_aware_weights(graph)
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    print("\n========== LSTM-BASED FAILURE-AWARE ROUTING ==========")
    routing_results = []
    for patient_node in patient_nodes:
        path = find_failure_aware_path(graph, patient_node, SINK_NODE)
        if path is None:
            routing_results.append({"model": "LSTM-Based Failure-Aware", "patient_node": patient_node, "path": None, "delivered": False})
        else:
            hop_count = len(path) - 1
            latency = calculate_path_latency(graph, path)
            average_failure_risk = calculate_path_failure_risk(graph, path)
            predicted_high_risk_nodes = count_predicted_high_risk_nodes_in_path(graph, path)
            average_energy = calculate_path_average_energy(graph, path)
            energy_cost = calculate_path_energy_cost(graph, path)
            average_risk = calculate_path_risk(graph, path)
            high_risk_nodes = count_high_risk_nodes_in_path(graph, path)
            routing_results.append({"model": "LSTM-Based Failure-Aware", "patient_node": patient_node, "path": path, "delivered": True,
                "hop_count": hop_count, "latency": latency, "average_failure_risk": average_failure_risk,
                "predicted_high_risk_nodes": predicted_high_risk_nodes, "average_energy": average_energy,
                "energy_cost": energy_cost, "average_risk": average_risk, "high_risk_nodes": high_risk_nodes})
    print("\n======================================================\n")
    return routing_results


# ============================================================
# 17. STEP 9: Q-LEARNING RL ROUTING AGENT
# ============================================================

def initialize_q_table(graph):
    q_table = {}
    for node in graph.nodes():
        q_table[node] = {}
        for neighbor in graph.neighbors(node):
            q_table[node][neighbor] = 0.0
    return q_table

def get_valid_neighbors(graph, current_node):
    return [n for n in graph.neighbors(current_node) if graph.nodes[n].get("status") == "active"]

def calculate_rl_step_reward(graph, current_node, next_node, sink_node, packet_priority):
    current_position = graph.nodes[current_node]["position"]
    next_position = graph.nodes[next_node]["position"]
    sink_position = graph.nodes[sink_node]["position"]
    current_distance_to_sink = calculate_distance(current_position, sink_position)
    next_distance_to_sink = calculate_distance(next_position, sink_position)
    progress = current_distance_to_sink - next_distance_to_sink
    next_node_delay = graph.nodes[next_node]["delay"]
    next_node_failure_risk = graph.nodes[next_node].get("failure_probability", 0)
    next_node_energy = graph.nodes[next_node]["residual_energy"]
    edge_data = graph.edges[current_node, next_node]
    link_quality = edge_data["link_quality"]
    distance = edge_data["distance"]
    energy_cost = 0.05 * distance + ((1 - link_quality) * 1.5)
    reward = 0.0
    reward += 8.0 * progress
    if progress <= 0:
        reward -= 120.0
    reward -= 25.0
    reward -= 0.70 * next_node_delay
    reward -= 150.0 * next_node_failure_risk
    reward -= 4.0 * energy_cost
    reward += 25.0 * link_quality
    reward += 0.10 * next_node_energy
    if packet_priority == CRITICAL_PACKET_PRIORITY:
        reward -= 0.50 * next_node_delay
        reward -= 100.0 * next_node_failure_risk
        reward += 15.0 * link_quality
    if next_node == sink_node:
        reward += 600.0
    if graph.nodes[next_node].get("risk_status") == "High-risk":
        reward -= 500.0
    return reward

def choose_rl_action(graph, q_table, current_node, epsilon):
    valid_neighbors = get_valid_neighbors(graph, current_node)
    if not valid_neighbors:
        return None
    if random.random() < epsilon:
        return random.choice(valid_neighbors)
    best_neighbor = None
    best_q_value = float("-inf")
    for neighbor in valid_neighbors:
        q_value = q_table[current_node].get(neighbor, 0.0)
        if q_value > best_q_value:
            best_q_value = q_value
            best_neighbor = neighbor
    return best_neighbor

def train_q_learning_routing_agent(graph, sink_node=SINK_NODE, packet_priority=NORMAL_PACKET_PRIORITY):
    q_table = initialize_q_table(graph)
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    epsilon = RL_EPSILON
    print("\n========== RL Q-LEARNING TRAINING STARTED ==========")
    print(f"Episodes: {RL_EPISODES}")
    print("====================================================\n")
    for episode in range(RL_EPISODES):
        source_node = random.choice(patient_nodes)
        current_node = source_node
        visited_nodes = set()
        for step in range(RL_MAX_STEPS):
            if current_node == sink_node:
                break
            visited_nodes.add(current_node)
            action_node = choose_rl_action(graph, q_table, current_node, epsilon)
            if action_node is None:
                break
            reward = calculate_rl_step_reward(graph, current_node, action_node, sink_node, packet_priority)
            if action_node in visited_nodes:
                reward -= 80.0
            future_neighbors = get_valid_neighbors(graph, action_node)
            if future_neighbors:
                max_future_q = max(q_table[action_node].get(neighbor, 0.0) for neighbor in future_neighbors)
            else:
                max_future_q = 0.0
            old_q = q_table[current_node].get(action_node, 0.0)
            new_q = old_q + RL_LEARNING_RATE * (reward + RL_DISCOUNT_FACTOR * max_future_q - old_q)
            q_table[current_node][action_node] = new_q
            current_node = action_node
        epsilon = max(RL_MIN_EPSILON, epsilon * RL_EPSILON_DECAY)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{RL_EPISODES} completed | Epsilon: {epsilon:.4f}")
    print("\n========== RL Q-LEARNING TRAINING COMPLETE ==========\n")
    return q_table

def find_rl_route(graph, q_table, source_node, sink_node=SINK_NODE):
    current_node = source_node
    path = [current_node]
    visited_nodes = set()
    sink_position = graph.nodes[sink_node]["position"]
    for step in range(RL_MAX_STEPS):
        if current_node == sink_node:
            return path
        visited_nodes.add(current_node)
        valid_neighbors = get_valid_neighbors(graph, current_node)
        if not valid_neighbors:
            return None
        best_neighbor = None
        best_score = float("-inf")
        current_distance_to_sink = calculate_distance(graph.nodes[current_node]["position"], sink_position)
        for neighbor in valid_neighbors:
            if neighbor in visited_nodes:
                continue
            neighbor_distance_to_sink = calculate_distance(graph.nodes[neighbor]["position"], sink_position)
            progress = current_distance_to_sink - neighbor_distance_to_sink
            q_value = q_table[current_node].get(neighbor, 0.0)
            failure_risk = graph.nodes[neighbor].get("failure_probability", 0)
            delay = graph.nodes[neighbor]["delay"]
            link_quality = graph.edges[current_node, neighbor]["link_quality"]
            score = (q_value + (20.0 * progress) - (200.0 * failure_risk) - (0.40 * delay) + (20.0 * link_quality))
            if graph.nodes[neighbor].get("risk_status") == "High-risk":
                score -= 500.0
            if score > best_score:
                best_score = score
                best_neighbor = neighbor
        if best_neighbor is None:
            return None
        path.append(best_neighbor)
        current_node = best_neighbor
    if path[-1] == sink_node:
        return path
    return None

def test_lstm_rl_self_healing_routing(graph, packet_priority=NORMAL_PACKET_PRIORITY):
    q_table = train_q_learning_routing_agent(graph, SINK_NODE, packet_priority=packet_priority)
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    print("\n========== LSTM + RL SELF-HEALING ROUTING ==========")
    routing_results = []
    for patient_node in patient_nodes:
        path = find_rl_route(graph, q_table, patient_node, SINK_NODE)
        if path is None:
            routing_results.append({"model": "LSTM + RL Self-Healing", "patient_node": patient_node, "path": None, "delivered": False})
        else:
            hop_count = len(path) - 1
            latency = calculate_path_latency(graph, path)
            average_failure_risk = calculate_path_failure_risk(graph, path)
            predicted_high_risk_nodes = count_predicted_high_risk_nodes_in_path(graph, path)
            average_energy = calculate_path_average_energy(graph, path)
            energy_cost = calculate_path_energy_cost(graph, path)
            average_risk = calculate_path_risk(graph, path)
            high_risk_nodes = count_high_risk_nodes_in_path(graph, path)
            routing_results.append({"model": "LSTM + RL Self-Healing", "patient_node": patient_node, "path": path, "delivered": True,
                "hop_count": hop_count, "latency": latency, "average_failure_risk": average_failure_risk,
                "predicted_high_risk_nodes": predicted_high_risk_nodes, "average_energy": average_energy,
                "energy_cost": energy_cost, "average_risk": average_risk, "high_risk_nodes": high_risk_nodes})
    print("\n====================================================\n")
    return routing_results


# ============================================================
# 18. STEP 10: FINAL SCENARIO-BASED EXPERIMENTATION
# ============================================================

def apply_disaster_scenario(graph, scenario_name):
    scenario = DISASTER_SCENARIOS[scenario_name]
    for node_id, data in graph.nodes(data=True):
        if data["node_type"] == "sink":
            continue
        position = data["position"]
        distance_from_disaster = calculate_distance(position, DISASTER_CENTER)
        affected_distance = DISASTER_RADIUS + 12
        if distance_from_disaster <= affected_distance:
            closeness = 1 - (distance_from_disaster / affected_distance)
            closeness = clip_value(closeness, 0, 1)
            data["temperature_risk"] = clip_value(data["temperature_risk"] + scenario["temperature_add"] * closeness, 0, 1)
            data["packet_loss_rate"] = clip_value(data["packet_loss_rate"] + scenario["packet_loss_add"] * closeness, 0, 1)
            data["delay"] = clip_value(data["delay"] + scenario["delay_add"] * closeness, 0, 180)
            data["residual_energy"] = clip_value(data["residual_energy"] - scenario["energy_drain"] * closeness, 0, 100)
            data["forwarding_load"] = clip_value(data["forwarding_load"] + scenario["load_add"] * closeness, 0, 45)
            data["rssi"] = clip_value(data["rssi"] - scenario["rssi_drop"] * closeness, -98, -35)
    return graph

def build_route_result(graph, model_name, patient_node, path):
    if path is None:
        return {"model": model_name, "patient_node": patient_node, "path": None, "delivered": False}
    return {"model": model_name, "patient_node": patient_node, "path": path, "delivered": True,
        "hop_count": len(path) - 1, "latency": calculate_path_latency(graph, path),
        "average_energy": calculate_path_average_energy(graph, path),
        "energy_cost": calculate_path_energy_cost(graph, path),
        "average_risk": calculate_path_risk(graph, path),
        "high_risk_nodes": count_high_risk_nodes_in_path(graph, path),
        "average_failure_risk": calculate_path_failure_risk(graph, path),
        "predicted_high_risk_nodes": count_predicted_high_risk_nodes_in_path(graph, path)}

def estimate_path_success_probability(graph, path):
    """Estimate end-to-end delivery probability across nodes in a path."""
    if path is None:
        return 0.0
    success_probability = 1.0
    for node in path:
        node_loss = graph.nodes[node].get("packet_loss_rate", 0)
        node_failure_risk = graph.nodes[node].get("failure_probability", 0)
        node_success = (1.0 - node_loss) * (1.0 - node_failure_risk)
        success_probability *= max(0.0, min(1.0, node_success))
    return success_probability

if __name__ == "__main__":
    wsn_graph = create_wsn_network()
    print_network_summary(wsn_graph)
    validate_network_connectivity(wsn_graph)
    print_sample_node_data(wsn_graph, number_of_nodes=10)

    shortest_path_results = test_shortest_path_routing(wsn_graph)
    summarize_routing_results(shortest_path_results, "Shortest Path Routing")

    energy_aware_results = test_energy_aware_routing(wsn_graph)
    summarize_routing_results(energy_aware_results, "Energy-Aware Routing")

    assign_failure_probabilities(wsn_graph)
    print_failure_risk_summary(wsn_graph)
    failure_aware_results = test_failure_aware_routing(wsn_graph)
    summarize_routing_results(failure_aware_results, "Weighted Failure-Risk-Aware Routing")

    run_step_6_lstm_data_generation(wsn_graph)
    train_lstm_failure_prediction_model()

    assign_lstm_failure_probabilities(wsn_graph)
    print_lstm_failure_risk_summary(wsn_graph)
    lstm_failure_aware_results = test_lstm_based_failure_aware_routing(wsn_graph)
    summarize_routing_results(lstm_failure_aware_results, "LSTM-Based Failure-Aware Routing")

    lstm_rl_results = test_lstm_rl_self_healing_routing(wsn_graph, packet_priority=CRITICAL_PACKET_PRIORITY)
    summarize_routing_results(lstm_rl_results, "LSTM + RL Self-Healing Routing")


def get_failed_nodes_for_recovery_test(graph):
    return [n for n, d in graph.nodes(data=True) if d.get("risk_status") == "High-risk"]


def estimate_recovery_time(graph, model_name, patient_node, original_path, failed_nodes):
    if original_path is None:
        return 999.0
    affected = any(node in failed_nodes for node in original_path)
    if not affected:
        return 0.0
    recovery_graph = graph.copy()
    for failed_node in failed_nodes:
        if failed_node in recovery_graph and failed_node not in [patient_node, SINK_NODE]:
            recovery_graph.remove_node(failed_node)
    try:
        if model_name == "Shortest Path":
            new_path = nx.shortest_path(recovery_graph, source=patient_node, target=SINK_NODE)
        elif model_name == "Energy-Aware":
            assign_energy_aware_weights(recovery_graph)
            new_path = nx.shortest_path(recovery_graph, source=patient_node, target=SINK_NODE, weight="energy_aware_weight")
        elif model_name == "Weighted Failure-Risk-Aware":
            assign_failure_probabilities(recovery_graph)
            assign_failure_aware_weights(recovery_graph)
            new_path = nx.shortest_path(recovery_graph, source=patient_node, target=SINK_NODE, weight="failure_aware_weight")
        elif model_name == "LSTM-Based Failure-Aware":
            assign_lstm_failure_probabilities(recovery_graph)
            assign_failure_aware_weights(recovery_graph)
            new_path = nx.shortest_path(recovery_graph, source=patient_node, target=SINK_NODE, weight="failure_aware_weight")
        else:
            new_path = nx.shortest_path(recovery_graph, source=patient_node, target=SINK_NODE)
        recovery_time = 25 + 8 * len(original_path) + 5 * len(new_path)
        return round(recovery_time, 2)
    except Exception:
        return 999.0


def summarize_experiment_results(results, scenario_name, model_name, failed_nodes):
    delivered_results = [r for r in results if r.get("delivered")]
    if not results:
        return None
    if not delivered_results:
        return {"scenario": scenario_name, "model": model_name, "route_delivery_ratio": 0.0,
            "estimated_packet_delivery_ratio": 0.0, "average_latency": None, "average_hop_count": None,
            "average_energy_cost": None, "average_disaster_risk": None, "average_high_risk_nodes": None,
            "average_recovery_time": 999.0}
    success_probabilities = [estimate_path_success_probability(graph, r["path"]) for r in delivered_results]
    recovery_times = [estimate_recovery_time(graph, model_name, r["patient_node"], r["path"], failed_nodes) for r in delivered_results]
    summary = {"scenario": scenario_name, "model": model_name,
        "route_delivery_ratio": round(len(delivered_results) / len(results), 4),
        "estimated_packet_delivery_ratio": round(sum(success_probabilities) / len(success_probabilities), 4),
        "average_latency": round(sum(r["latency"] for r in delivered_results) / len(delivered_results), 2),
        "average_hop_count": round(sum(r["hop_count"] for r in delivered_results) / len(delivered_results), 2),
        "average_path_energy": round(sum(r["average_energy"] for r in delivered_results) / len(delivered_results), 2),
        "average_energy_cost": round(sum(r["energy_cost"] for r in delivered_results) / len(delivered_results), 2),
        "average_disaster_risk": round(sum(r["average_risk"] for r in delivered_results) / len(delivered_results), 2),
        "average_high_risk_nodes": round(sum(r["high_risk_nodes"] for r in delivered_results) / len(delivered_results), 2),
        "average_recovery_time": round(sum(recovery_times) / len(recovery_times), 2)}
    return summary


def get_shortest_path_results_silent(graph):
    results = []
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    for patient_node in patient_nodes:
        path = find_shortest_path(graph, patient_node, SINK_NODE)
        results.append(build_route_result(graph, "Shortest Path", patient_node, path))
    return results


def get_energy_aware_results_silent(graph):
    results = []
    assign_energy_aware_weights(graph)
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    for patient_node in patient_nodes:
        path = find_energy_aware_path(graph, patient_node, SINK_NODE)
        results.append(build_route_result(graph, "Energy-Aware", patient_node, path))
    return results


def get_weighted_failure_aware_results_silent(graph):
    results = []
    assign_failure_probabilities(graph)
    assign_failure_aware_weights(graph)
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    for patient_node in patient_nodes:
        path = find_failure_aware_path(graph, patient_node, SINK_NODE)
        results.append(build_route_result(graph, "Weighted Failure-Risk-Aware", patient_node, path))
    return results


def get_lstm_failure_aware_results_silent(graph):
    results = []
    assign_lstm_failure_probabilities(graph)
    assign_failure_aware_weights(graph)
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    for patient_node in patient_nodes:
        path = find_failure_aware_path(graph, patient_node, SINK_NODE)
        results.append(build_route_result(graph, "LSTM-Based Failure-Aware", patient_node, path))
    return results


def get_lstm_rl_results_silent(graph):
    assign_lstm_failure_probabilities(graph)
    q_table = train_q_learning_routing_agent(graph, SINK_NODE, packet_priority=CRITICAL_PACKET_PRIORITY)
    results = []
    patient_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "patient_sensor"]
    for patient_node in patient_nodes:
        path = find_rl_route(graph, q_table, patient_node, SINK_NODE)
        results.append(build_route_result(graph, "LSTM + RL Self-Healing", patient_node, path))
    return results
