# Test/simulate.py

import sys
import os
import networkx as nx
import time
from collections import Counter

# Add the project's root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from logic.robot_controller import RobotController
from logic.graph_manager import GraphManager

# --- Mock Classes ---
class MockRobotCommunicator:
    """A mock of the robot communicator that just prints commands."""
    def set_command(self, command):
        print(f"    [ROBOT CMD] ==> {command}")

class MockCSharpCommunicator:
    """A mock of the C# UI communicator that does nothing."""
    def register_command_callback(self, callback): pass
    def send_update(self, packet): pass

# --- Ground Truth Map ---
# def create_ground_truth_map():
#     """Defines the map where some edges meet at a central node."""
#     G = nx.Graph()
#     edges = [
#         ('N0', 'N1'), 
#         ('N0', 'N7'), 
#         ('N1', 'N8'),
#         ('N1', 'N2'), 
#         ('N2', 'N3'), 
#         ('N3', 'N8'), 
#         ('N3', 'N4'), 
#         ('N4', 'N5'), 
#         ('N5', 'N8'), 
#         ('N5', 'N6'), 
#         ('N6', 'N7'), 
#         ('N7', 'N8'), 
#         ('N7', 'N9'), 
#     ]
#     G.add_edges_from(edges)
#     return G
# def create_ground_truth_map():
#     """Defines the map based on the uploaded ArUco tag image."""
    # G = nx.Graph()
    # edges = [
    #     ('N0', 'N1'), ('N0', 'N6'), ('N1', 'N2'), ('N1', 'N7'), 
    #     ('N2', 'N3'), ('N2', 'N8'), ('N3', 'N4'), ('N3', 'N9'),
    #     ('N4', 'N5'), ('N4', 'N10'), ('N5', 'N11'), ('N6', 'N7'),
    #     ('N6', 'N12'), ('N7', 'N8'), ('N7', 'N13'), ('N8', 'N9'),
    #     ('N8', 'N14'), ('N9', 'N10'), ('N9', 'N15'), ('N10', 'N11'),
    #     ('N10', 'N16'), ('N11', 'N17'), ('N12', 'N13'), ('N12', 'N18'),
    #     ('N13', 'N14'), ('N13', 'N19'), ('N14', 'N15'), ('N14', 'N20'),
    #     ('N15', 'N16'), ('N15', 'N21'), ('N16', 'N17'), ('N16', 'N22'),
    #     ('N17', 'N23'), ('N18', 'N19'), ('N19', 'N20'), ('N20', 'N21'),
    #     ('N21', 'N22'), ('N22', 'N23'),
    # ]
    # edges = [
    #     # Horizontal connections
    #     ('N0', 'N1'), ('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'),
    #     ('N5', 'N6'), ('N6', 'N7'),
    #     ('N8', 'N9'),
    #     ('N10', 'N11'), ('N11', 'N12'),
    #     ('N13', 'N14'),
    #     ('N15', 'N16'), ('N16', 'N17'),

    #     # Vertical connections
    #     ('N0', 'N5'),
    #     ('N2', 'N7'),
    #     ('N3', 'N8'),
    #     ('N4', 'N9'),
    #     ('N5', 'N10'),
    #     ('N6', 'N11'),
    #     ('N7', 'N12'),
    #     ('N9', 'N14'),
    #     ('N10', 'N15'),
    #     ('N11', 'N16'),
    #     ('N12', 'N17'),
    #     ('N13', 'N18'),
    #     ('N18', 'N19'),
    # ]
    # G.add_edges_from(edges)
    # return G

# In Test/simulate.py

def create_ground_truth_map():
    """Defines the new, challenging 20-node map."""
    G = nx.Graph()
    edges = [
    ('N0', 'N1'), ('N0', 'N9'), ('N1', 'N4'), ('N2', 'N3'), 
    ('N2', 'N5'), ('N3', 'N8'), ('N4', 'N5'), ('N5', 'N6'), 
    ('N5', 'N10'), ('N6', 'N11'), ('N7', 'N8'), ('N7', 'N14'), 
    ('N8', 'N15'), ('N9', 'N10'), ('N10', 'N11'), ('N10', 'N13'), 
    ('N12', 'N13'), ('N12', 'N16'), ('N13', 'N14'), ('N14', 'N15'), 
    ('N14', 'N18'), ('N15', 'N19'), ('N17', 'N18'), ('N18', 'N19'), 
    ]
    G.add_edges_from(edges)
    return G


def get_mock_node_type(ground_truth_map, node_id):
    """Simulates node type based on the number of neighbors."""
    degree = ground_truth_map.degree(node_id)
    if degree >= 4: return '4-way'
    if degree == 3: return '3-way'
    if degree == 2: return 'Turn'
    if degree <= 1: return 'dead-end'
    return 'unknown'

import random 

def run_simulation():
    print("--- 🤖 STARTING AGENT-BASED SIMULATION (RANDOMIZED) 🤖 ---")

    # --- 1. INITIALIZATION ---
    ground_truth_map = create_ground_truth_map()
    robot_map_manager = GraphManager()
    controller = RobotController(
        graph_manager=robot_map_manager,
        image_processor=None,
        robot_comm=MockRobotCommunicator(),
        csharp_comm=MockCSharpCommunicator(),
        calibrator=None,
        transformer=None
    )

    # Statistics trackers
    edge_visit_counter = Counter()
    visited_node_sequence = []

    # Simulation state variables
    last_node = None
    current_node = 'N0' # Start node
    
    step_count = 0
    max_steps = 200

    # --- 2. MAIN AGENT-BASED SIMULATION LOOP ---
    while step_count < max_steps:
        step_count += 1
        
        controller.last_node_id = last_node
        controller.current_node_id = current_node
        
        print(f"\n--- Step {step_count}: Robot at '{current_node}' (came from '{last_node}') ---")

        if last_node:
            edge = tuple(sorted((last_node, current_node)))
            edge_visit_counter[edge] += 1
            controller.visited_edges.add(edge)

        visited_node_sequence.append(current_node)
        controller.visited_nodes.add(current_node)
        controller.revisit_count[current_node] += 1
        robot_map_manager.add_node(current_node)

        for neighbor in ground_truth_map.neighbors(current_node):
            robot_map_manager.add_node(neighbor)
            robot_map_manager.add_edge(current_node, neighbor, cost=10)

        node_type = get_mock_node_type(ground_truth_map, current_node)
        
        unexplored_connections = [
            n for n in ground_truth_map.neighbors(current_node) 
            if tuple(sorted((current_node, n))) not in controller.visited_edges
        ]
        random.shuffle(unexplored_connections)
        
        controller.unexplored_exits[current_node] = unexplored_connections
        print(f"  Analysis: Type='{node_type}', Unexplored Connections (Shuffled)={unexplored_connections}")

        potential_local_choice = unexplored_connections[0] if unexplored_connections else None
        decision_log = controller._decide_next_exploration_target()
        print(f"  Decision: {decision_log}")

        next_node = None
        if controller.state == "MAPPING_PATH" or controller.state == "TURNING":
            next_node = potential_local_choice
            if next_node:
                print(f"  Action: Moving to '{next_node}' via untraversed edge.")
        
        elif controller.state == "PATHFINDING":
            if len(controller.navigation_path) > 1:
                next_node = controller.navigation_path[1]
                print(f"  Action: Start backtracking. Next hop: '{next_node}'.")
            else:
                controller.state = "IDLE"
                print("  Action: Pathfinding path too short. Halting.")

        elif controller.state == "IDLE":
            print("  Action: Exploration complete. Halting.")
            break
        
        if not next_node:
            print("  Action: No next node decided. Halting.")
            break

        last_node = current_node
        current_node = next_node
            
        time.sleep(0.02)

    # --- 3. FINAL REPORT (Corrected single block) ---
    print("\n" + "="*40)
    print("--- 🤖 SIMULATION COMPLETE 🤖 ---")
    print("="*40)
    
    print("\n--- MAP COVERAGE ---")
    print(f"Ground truth: {len(ground_truth_map.nodes())} nodes, {len(ground_truth_map.edges())} edges.")
    print(f"Robot discovered: {len(robot_map_manager.graph.nodes())} nodes, {len(robot_map_manager.graph.edges)} edges.")
    
    print("\n--- 📊 STATISTICS ---")
    print(f"\n👟 Total Steps Taken: {step_count - 1}")
    print("\n🧭 Node Visit Sequence:")
    print(" -> ".join(visited_node_sequence))

    print("\n📌 Node Visit Counts:")
    for node, count in sorted(controller.revisit_count.items()):
        print(f"  {node}: {count} times")

    print("\n🚦 Edge Traversal Counts:")
    for edge, count in sorted(edge_visit_counter.items()):
        print(f"  {edge}: {count} times {'(Backtrack)' if count > 1 else ''}")

if __name__ == "__main__":
    run_simulation()