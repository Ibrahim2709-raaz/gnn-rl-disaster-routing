import unittest

import networkx as nx

from main_simulation_clean import create_wsn_network, estimate_path_success_probability


class SimulationSmokeTests(unittest.TestCase):
    def test_default_network_has_expected_nodes(self):
        graph = create_wsn_network()
        node_types = nx.get_node_attributes(graph, "node_type")

        self.assertEqual(graph.number_of_nodes(), 74)
        self.assertEqual(sum(t == "patient_sensor" for t in node_types.values()), 8)
        self.assertEqual(sum(t == "sink" for t in node_types.values()), 1)

    def test_path_success_probability_is_bounded(self):
        graph = nx.Graph()
        graph.add_node("a", packet_loss_rate=0.10, failure_probability=0.20)
        graph.add_node("b", packet_loss_rate=0.05, failure_probability=0.10)

        probability = estimate_path_success_probability(graph, ["a", "b"])

        self.assertAlmostEqual(probability, 0.9 * 0.8 * 0.95 * 0.9)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
