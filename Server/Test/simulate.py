import sys
import os
import networkx as nx
import time
from collections import Counter
import math

# Add the project's root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from logic.robot_controller import RobotController
from logic.graph_manager import GraphManager

# --- Mock Classes ---
class MockRobotCommunicator:
    """A mock of the robot communicator that just prints commands."""
    def set_drive_command(self, command, payload):
        if payload:
            print(f"    [ROBOT CMD] ==> {command} {payload}")
        else:
            print(f"    [ROBOT CMD] ==> {command}")

class MockCSharpCommunicator:
    """A mock of the C# UI communicator that does nothing."""
    def register_command_callback(self, callback): pass
    def send_update(self, packet): pass

# --- Ground Truth Map ---
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
    # We assign a (x, y) coordinate to each node to calculate directions and distances.
    positions = {
        'N0': (0, 1), 'N1': (1, 1), 'N2': (2, 1), 'N3': (3, 1), 'N4': (1, 0),
        'N5': (2, 0), 'N6': (3, 0), 'N7': (4, 2), 'N8': (4, 1), 'N9': (0, 2),
        'N10': (1, 2), 'N11': (2, 2), 'N12': (1, 3), 'N13': (2, 3), 'N14': (3, 2),
        'N15': (4, 3), 'N16': (1, 4), 'N17': (4, 5), 'N18': (4, 4), 'N19': (3, 4)
    }
    G.add_edges_from(edges)
    nx.set_node_attributes(G, positions, 'pos')
    return G

def get_direction_from_nodes(graph, from_node, to_node):
    """Calculates the cardinal direction from one node to another based on their positions."""
    pos = nx.get_node_attributes(graph, 'pos')
    pos_from = pos.get(from_node)
    pos_to = pos.get(to_node)

    # Failsafe if a node somehow doesn't have a position
    if not pos_from or not pos_to:
        return 'up'

    delta_x = pos_to[0] - pos_from[0]
    delta_y = pos_to[1] - pos_from[1]
    
    # Use -delta_y because in typical screen coordinates, the y-axis is inverted (0 is at the top).
    angle = math.degrees(math.atan2(-delta_y, delta_x)) 
    
    if -45 <= angle < 45: return "right"
    if 45 <= angle < 135: return "up"
    if 135 <= angle <= 180 or -180 <= angle < -135: return "left"
    return "down" # -135 <= angle < -45


def get_mock_node_type(ground_truth_map, node_id):
    """Simulates node type based on the number of neighbors."""
    degree = ground_truth_map.degree(node_id)
    if degree >= 4: return '4-way'
    if degree == 3: return '3-way'
    if degree == 2: return 'Turn'
    if degree <= 1: return 'dead-end'
    return 'unknown'

def run_simulation():
    print("--- 🤖 STARTING AGENT-BASED SIMULATION (v2 - ACCURATE PREDICTION) 🤖 ---")

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

    # <<< شروع تغییرات: کالیبره کردن پارامترهای شبیه‌ساز با کنترلر >>>
    # این پارامتر باید با مقیاس مختصات مجازی ما هماهنگ باشد
    controller.pixels_per_frame = 1.0
    # <<< پایان تغییرات >>>

    edge_visit_counter = Counter()
    visited_node_sequence = []
    last_node = None
    current_node = 'N0'
    step_count = 0
    max_steps = 200

    # --- 2. MAIN AGENT-BASED SIMULATION LOOP ---
    while step_count < max_steps:
        step_count += 1
        
        controller.last_node_id = last_node
        controller.current_node_id = current_node
        
        current_pos_tuple = nx.get_node_attributes(ground_truth_map, 'pos').get(current_node)
        if current_pos_tuple:
            controller.robot_position = current_pos_tuple

        if last_node:
            direction_moved = get_direction_from_nodes(ground_truth_map, last_node, current_node)
            direction_angles = {'up': -90.0, 'right': 0.0, 'down': 90.0, 'left': 180.0}
            controller.robot_heading = direction_angles.get(direction_moved, -90.0)

        print(f"\n--- Step {step_count}: Robot at '{current_node}' (from '{last_node}') ---")
        print(f"  Internal Heading: {controller.robot_heading} deg")

        if last_node:
            edge = tuple(sorted((last_node, current_node)))
            edge_visit_counter[edge] += 1
            controller.visited_edges.add(edge)

        visited_node_sequence.append(current_node)
        controller.visited_nodes.add(current_node)
        controller.revisit_count[current_node] += 1
        
        current_pos_list = list(current_pos_tuple) if current_pos_tuple else [0,0]
        robot_map_manager.add_node(current_node, position=current_pos_list)

        if last_node:
            # <<< شروع تغییرات: محاسبه هزینه واقعی بر اساس فاصله >>>
            last_pos_tuple = nx.get_node_attributes(ground_truth_map, 'pos').get(last_node)
            if last_pos_tuple:
                geometric_distance = math.dist(current_pos_tuple, last_pos_tuple)
                # تبدیل فاصله به "فریم" برای کنترلر
                cost_in_frames = int(geometric_distance / controller.pixels_per_frame)
                robot_map_manager.add_edge(last_node, current_node, cost=cost_in_frames)
                print(f"  Edge cost ({last_node}-{current_node}) based on distance: {cost_in_frames} frames")
            # <<< پایان تغییرات >>>

        unexplored_neighbors = [
            n for n in ground_truth_map.neighbors(current_node) 
            if tuple(sorted((current_node, n))) not in controller.visited_edges
        ]
        
        # <<< شروع تغییرات: حذف جهت‌های تکراری >>>
        # مختصات مجازی ما کامل نیست و ممکن است دو مسیر مختلف یک جهت را نشان دهند
        # با استفاده از set، موارد تکراری را حذف می‌کنیم
        unexplored_directions = list(set([
            get_direction_from_nodes(ground_truth_map, current_node, neighbor)
            for neighbor in unexplored_neighbors
        ]))
        # <<< پایان تغییرات >>>
        
        controller.unexplored_exits[current_node] = unexplored_directions
        print(f"  Analysis: Unexplored Directions = {unexplored_directions}")

        controller._set_state("AT_NODE", f"Simulated arrival at {current_node}")
        
        decision_log = controller._decide_next_exploration_target()
        print(f"  Decision: {decision_log}")

        next_node = None
        if controller.state in ["DEPARTING_NODE", "CENTERING_AT_NODE", "TURNING"]:
            chosen_direction = None
            if decision_log:
                # تلاش برای استخراج جهت از پیام تصمیم‌گیری
                parts = decision_log.split("'")
                if len(parts) > 1:
                    potential_dir = parts[1]
                    if potential_dir in ['up', 'down', 'left', 'right']:
                        chosen_direction = potential_dir
            
            # اگر جهت استخراج نشد، از اولین گزینه موجود استفاده کن
            if not chosen_direction and unexplored_directions:
                sorted_dirs = sorted(unexplored_directions, key=lambda d: {'up':0, 'left':1, 'right':2}.get(d, 99))
                chosen_direction = sorted_dirs[0]

            for neighbor in unexplored_neighbors:
                if get_direction_from_nodes(ground_truth_map, current_node, neighbor) == chosen_direction:
                    next_node = neighbor
                    break
            
            if next_node:
                print(f"  Action: Moving towards '{chosen_direction}' -> '{next_node}'.")

        elif controller.state == "PATHFINDING":
            if controller.navigation_path and len(controller.navigation_path) > 1:
                next_node = controller.navigation_path[1]
                print(f"  Action: Backtracking. Next hop: '{next_node}'.")

        elif controller.state == "IDLE":
            print("  Action: Exploration complete. Halting.")
            break
        
        if not next_node:
            if controller.state == "PATHFINDING" and controller.navigation_path:
                 # In pathfinding, the next node is the first element of the remaining path
                 next_node = controller.navigation_path[0]
                 print(f"  Action: No local exits. Backtracking towards '{next_node}'.")
            else:
                # This can happen if all exits were pruned and no global path was found
                print("  Action: No next node decided. Halting.")
                break

        last_node = current_node
        current_node = next_node
            
        time.sleep(0.01)

    # --- 3. FINAL REPORT ---
    print("\n" + "="*40)
    print("--- 🤖 SIMULATION COMPLETE 🤖 ---")
    print("="*40)
    print(f"\n--- MAP COVERAGE ---")
    print(f"Ground truth: {len(ground_truth_map.nodes())} nodes, {len(ground_truth_map.edges())} edges.")
    discovered_nodes = len(robot_map_manager.graph.nodes())
    discovered_edges = len(robot_map_manager.graph.edges)
    print(f"Robot discovered: {discovered_nodes} nodes, {discovered_edges} edges.")
    node_coverage = (discovered_nodes / len(ground_truth_map.nodes())) * 100
    edge_coverage = (discovered_edges / len(ground_truth_map.edges())) * 100
    print(f"Coverage: {node_coverage:.1f}% of nodes, {edge_coverage:.1f}% of edges.")
    
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