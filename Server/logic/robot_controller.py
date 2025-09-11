import base64
import math
import cv2
import numpy as np
from logic.graph_manager import GraphManager
from vision.image_processor import ImageProcessor
from communication.robot_communicator import RobotCommunicator
from communication.csharp_communicator import CSharpCommunicator
from vision.camera_calibrator import CameraCalibrator
from vision.perspective_transformer import PerspectiveTransformer
from collections import defaultdict, Counter

class RobotController:
    """The central brain with the state machine and decision-making logic."""

    def __init__(self, graph_manager: GraphManager, image_processor: ImageProcessor, 
                 robot_comm: RobotCommunicator, csharp_comm: CSharpCommunicator, 
                 calibrator: CameraCalibrator, transformer: PerspectiveTransformer):
            
        # <<< ADDED: Variables for motion calibration >>>
        self.motion_calib_state = {'sub_state': 'idle'}
        # Default high values, will be overwritten by calibration
        self.calibrated_turn_frames = {"left_360": 120, "right_360": 120}
        self._load_motion_calibration_data() # Load saved values on startup
        
        self.graph_manager = graph_manager
        self.image_processor = image_processor
        self.robot_comm = robot_comm
        self.csharp_comm = csharp_comm
        self.calibrator = calibrator
        self.transformer = transformer

        # --- State Management ---
        self.state = "IDLE"
        self.current_node_id = None
        self.last_node_id = None
        self.navigation_path = []
        self.processed_frame = None 
        self.last_calib_check_success = False

        # --- WFD and Exploration Algorithm Variables ---
        self.visited_nodes = set()
        self.visited_edges = set()
        self.revisit_count = defaultdict(int)
        self.unexplored_exits = {} 
        self.turn_state = {} 
        self.node_approach_data = {}
        self.initial_search_frame_counter = 0 
        self.frame_counter = 0
        self.no_line_frame_count = 0
        self.wfd_weights = {"w1": 1.0, "w2": 2.5, "w3": 2.0, "w4": 0.3}
        self.frontier_age = {}
        self.age_counter = 0
        self.turn_escape_frame_goal = 0
        self.no_line_centering_count = 0 
        self.last_positional_error = 0
        self.last_known_line_direction = "LEFT"
        self.await_frame_counter = 0

        self.departing_frame_counter = 0
        self.next_state_after_await = "IDLE"

        self.robot_position = (150, 250)
        self.robot_heading = -90.0
        self.pixels_per_frame = 0.5

        self.turn_logic_state = {}
        self.is_node_analyzed = False
        
        self.node_analysis_data = {}

        self.dead_end_exits = defaultdict(set)
        
        # --- Register Callback ---
        self.csharp_comm.register_command_callback(self.on_command_received)

    # <<< ADDED: Helper function for state transition logging >>>
    def _set_state(self, new_state: str, reason: str):
        """
        Changes the robot's state and prints a formatted log message.
        This is the ONLY function that should modify self.state.
        """
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            print("\n" + "="*60)
            print(f"STATE TRANSITION:  {old_state} -> {self.state}")
            print(f"       REASON:  {reason}")
            print("="*60 + "\n")

    def on_command_received(self, command_info: dict):
        """Callback to handle all incoming commands from the C# UI."""
        command_type = command_info.get("type")
        payload = command_info.get("payload")
        
        # print(f"\n--- STATE: {self.state} --- ALL_VARS: {self.__dict__}") # <<< This can be uncommented for extreme debugging

        if command_type == "start_exploration": self._handle_start_exploration()
        elif command_type == "emergency_stop": self._handle_emergency_stop()
        elif command_type == "reset_map": self._handle_reset_map()
        elif command_type == "start_calibration": self._handle_start_calibration()
        elif command_type == "capture_calib_image": self._handle_capture_calib_image()
        elif command_type == "finish_calib": self._handle_finish_calibration()
        elif command_type == "start_perspective_setup": self._handle_start_perspective_setup()
        elif command_type == "perspective_point_clicked": self._handle_perspective_point_clicked(payload)
        elif command_type == "finish_perspective": self._handle_finish_perspective()
        elif command_type == "cancel_setup": self._handle_cancel_setup()
        elif command_type == "toggle_led": self._handle_toggle_led(payload)
        elif command_type == "set_target": self._handle_set_target(payload)
        elif command_type == "start_motion_calibration":
            self._set_state("MOTION_CALIBRATION", "Start Motion Calibration command received.")
            self.motion_calib_state = {
                'sub_state': 'start',
                'target_marker_id': None,
                'calib_frame_counter': 0,
                'confirmation_counter': 0
            }

        else: print(f"⚠️ Unknown command type received: {command_type}")

    # --- Command Handler Implementations ---

    def _handle_start_exploration(self):
        print("▶️ Command: Start Exploration received.")
        
        # --- THIS IS THE FIX ---
        # Reset all state and memory variables for a fresh start
        self.frame_counter = 0
        self.current_node_id = None
        self.last_node_id = None
        self.navigation_path = []
        self.visited_nodes = set()
        self.visited_edges = set()
        self.revisit_count = defaultdict(int)
        self.unexplored_exits = {}
        self.dead_end_exits = defaultdict(set)
        # -----------------------

        self._set_state("INITIAL_SEARCH", "Start Exploration command received from UI.")

    def _handle_emergency_stop(self):
        print("⏹️ Command: Emergency Stop")
        # <<< CHANGED >>>
        self._set_state("IDLE", "Emergency Stop command received from UI.")
        self.robot_comm.set_drive_command("STOP", None)

    def _handle_reset_map(self):
        print("🔄 Command: Reset Map")
        self.graph_manager.clear()
        self.current_node_id = None
        self.last_node_id = None
        self.navigation_path = []
        # <<< CHANGED >>>
        self._set_state("IDLE", "Reset Map command received from UI.")

    def _handle_start_calibration(self):
        print("📷 Command: Start Calibration")
        # <<< CHANGED >>>
        self._set_state("CALIBRATING", "Start Calibration command received.")
        self.calibrator.reset()

    def _handle_capture_calib_image(self):
        if self.state == "CALIBRATING" and self.last_calib_check_success:
            print(f"📷 Image captured for calibration.")
            self.calibrator.capture_current_frame()

    def _handle_finish_calibration(self):
        if self.state == "CALIBRATING":
            print("💾 Command: Finish & Save Calibration")
            if self.calibrator.calibrate():
                self.image_processor.load_calibration_data()
            # <<< CHANGED >>>
            self._set_state("IDLE", "Calibration finished.")

    def _handle_start_perspective_setup(self):
        print("📐 Command: Start Perspective Setup")
        # <<< CHANGED >>>
        self._set_state("PERSPECTIVE_SETUP", "Start Perspective Setup command received.")
        self.transformer.reset_points()

    def _handle_perspective_point_clicked(self, payload: dict):
        if self.state == "PERSPECTIVE_SETUP" and payload:
            point = (payload['x'], payload['y'])
            self.transformer.add_point(point)
            print(f"📐 Perspective point added: {point}")

    def _handle_finish_perspective(self):
        if self.state == "PERSPECTIVE_SETUP":
            print("💾 Command: Finish & Save Perspective")
            if self.transformer.set_transform_from_points():
                self.transformer.save_perspective_data()
                self.image_processor.load_perspective_data()
            # <<< CHANGED >>>
            self._set_state("IDLE", "Perspective setup finished.")

    def _handle_cancel_setup(self):
        print("❌ Command: Cancel Setup")
        # <<< CHANGED >>>
        self._set_state("IDLE", "Setup cancelled by user.")

    def _handle_toggle_led(self, payload: dict):
        if payload:
            led_state = payload.get("state", False)
            self.robot_comm.set_led_state("LED_ON" if led_state else "LED_OFF")

    def _handle_set_target(self, payload: dict):
        if not self.current_node_id:
            print("⚠️ Cannot set target, current position is unknown.")
            return
        if payload and "target_node" in payload:
            target_id = payload["target_node"]
            print(f"🎯 New target received: {target_id}")
            path = self.graph_manager.get_path_astar(self.current_node_id, target_id)
            if path:
                self.navigation_path = path
                # <<< CHANGED >>>
                self._set_state("PATHFINDING", f"Path calculated by user to target: {target_id}.")
                print(f"🗺️ Path calculated: {self.navigation_path}")
            else:
                print(f"❌ Could not calculate path to {target_id}.")

    def process_frame(self, raw_frame: np.ndarray):
        log_message = f"State: {self.state}"
        
        # --- 1. Handle Special Setup Modes First ---
        if self.state == "CALIBRATING":
            found, calib_frame = self.calibrator.check_chessboard(raw_frame.copy())
            self.last_calib_check_success = found
            self.processed_frame = calib_frame
            log_message = "Point camera at chessboard. Ready to capture." if found else "Chessboard not found."
            self._send_csharp_update(log_message)
            return

        elif self.state == "PERSPECTIVE_SETUP":
            try:
                undistorted_frame = self.image_processor.undistort_frame(raw_frame)
                self.processed_frame = undistorted_frame
            except Exception:
                self.processed_frame = raw_frame # Failsafe
            log_message = f"Click points on C# UI. {len(self.transformer.points)}/4 points selected."
            self._send_csharp_update(log_message)
            return

        # --- 2. Main Operational Logic ---
        corrected_frame = self.image_processor.process_and_correct_frame(raw_frame)
        
        if corrected_frame is not None:
            # --- 3. Initialize variables for this frame ---
            line_info = None
            intersection_result = None
            node_info = self.image_processor.detect_node_in_roi(corrected_frame)
            (detected_node_id, _) = node_info
            node_id_str = f"N{detected_node_id}" if detected_node_id is not None else None

            # <<< --- START OF THE CRITICAL FIX --- >>>
            # --- 4. Priority Check: Handle Node Arrival FIRST ---
            # This block checks if we've arrived at a new node and decides what to do
            # based on the CURRENT state (Pathfinding vs. Exploring).
            # MIN_FRAMES_BEFORE_NODE_DETECTION = 20
            MIN_FRAMES_BEFORE_NODE_DETECTION = 15
            if node_id_str and node_id_str != self.last_node_id and self.frame_counter > MIN_FRAMES_BEFORE_NODE_DETECTION:
                
                # If we are PATHFINDING, we are just passing through.
                # We update our current location and let the pathfinding logic handle the next turn.
                if self.state == "PATHFINDING":
                    self.current_node_id = node_id_str
                    # The _handle_pathfinding_state will now take over
                
                # If we are EXPLORING, this is a new discovery. We must stop to analyze.
                elif self.state in ["EXPLORING_PATH", "MAPPING_PATH"]:
                    self.robot_comm.set_drive_command("STOP", None)
                    self.node_approach_data = {"node_id": node_id_str}
                    self._set_state("AT_NODE", f"Node {node_id_str} sighted after {self.frame_counter} frames. Stopped for analysis.")
            # <<< ---  END OF THE CRITICAL FIX  --- >>>

            # --- 5. Execute State-Specific Logic (The Main State Machine) ---
            # The logic here is now cleaner because node arrival is handled above.
            
            if self.state == "MOTION_CALIBRATION":
                log_message = self._handle_motion_calibration_state(detected_node_id)
            
            elif self.state in ["TURNING", "TURNING_AROUND"]:
                turn_direction = self.turn_logic_state.get("direction", "LEFT").upper()
                line_info = self.image_processor.calculate_line_command(corrected_frame, focus_area=turn_direction)
                log_message = self._handle_turning_state(line_info)
            
            else: # All other operational states
                # For most states, we need the line information
                if self.state not in ["IDLE", "AT_NODE", "CENTERING_AT_NODE", "DEPARTING_NODE"]:
                    line_info = self.image_processor.calculate_line_command(corrected_frame)

                if self.state == "IDLE":
                    log_message = self._handle_idle_state()
                elif self.state == "INITIAL_SEARCH":
                    log_message = self._handle_initial_search_state(line_info, detected_node_id)
                elif self.state == "EXPLORING_PATH" or self.state == "MAPPING_PATH":
                    log_message = self._handle_exploring_path_state(line_info, None) # Node ID is now handled by the priority check
                elif self.state == "PATHFINDING":
                    log_message = self._handle_pathfinding_state(line_info, None) # Node ID is now handled by the priority check
                elif self.state == "AWAITING_DRIVE":
                    log_message = self._handle_awaiting_drive_state()
                elif self.state == "APPROACHING_NODE":
                    log_message = self._handle_approaching_node()
                elif self.state == "AT_NODE":
                    intersection_result = self.image_processor.analyze_intersection(corrected_frame)
                    log_message = self._handle_at_node_state(intersection_result)
                elif self.state == "CENTERING_AT_NODE":
                    log_message = self._handle_centering_at_node_state()
                elif self.state == "DEPARTING_NODE":
                    log_message = self._handle_departing_node_state()

            self.processed_frame = self.image_processor.draw_diagnostics(
                frame=corrected_frame,
                line_info=line_info,
                node_info=node_info,
                intersection_analysis=intersection_result
            )
        else:
            self.processed_frame = raw_frame
            log_message = "Graceful fallback: Using raw frame due to correction failure."

        # --- 6. Send Final Update to C# ---
        self._send_csharp_update(log_message)

    def get_processed_frame(self) -> np.ndarray | None:
        """Returns the last frame that was processed with diagnostics."""
        return self.processed_frame

    def _send_csharp_update(self, log_message: str):
        """Prepares and sends the full data packet to the C# client."""
        if self.processed_frame is None:
            self.processed_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        _, buffer = cv2.imencode('.jpg', self.processed_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        graph_data = self.graph_manager.get_graph_data_for_client()
        
        heading_str = "none"
        heading_deg = self.robot_heading
        if -135 <= heading_deg < -45: heading_str = "up"
        elif -45 <= heading_deg < 45: heading_str = "right"
        elif 45 <= heading_deg < 135: heading_str = "down"
        elif 135 <= heading_deg <= 180 or -180 <= heading_deg < -135: heading_str = "left"


        packet = {
            "image": jpg_as_text,
            "log": log_message,
            "is_connected": self.csharp_comm.data_client_socket is not None,
            "current_node": self.current_node_id,
            "nodes": graph_data.get("nodes", []),
            "edges": graph_data.get("edges", []),
            "navigation_path": self.navigation_path,
            "robot_heading": heading_str 
        }

        self.csharp_comm.send_update(packet)

    # --- State Handler Methods ---
    def _handle_idle_state(self):
        """
        Handles the IDLE state. Ensures the robot is stopped and waits for a new command.
        """
        self.robot_comm.set_drive_command("STOP", None) 
        return "Idle. Waiting for command."

    def _handle_turning_around_state(self, line_info: dict | None):
        """
        Handles the 180-degree turn at a dead-end by using the main turning logic.
        """
        return self._handle_turning_state(line_info)

    def _is_more_complex_type(self, new_type: str, old_type: str | None) -> bool:
        """
        Determines if a newly detected node type is more complex than the old one.
        """
        if old_type is None:
            return True
        hierarchy = {
            'unknown': 0, 'dead-end': 1, 'straight': 2, 
            'Turn-left': 2, 'Turn-right': 2, '3-way': 3, '4-way': 4
        }
        return hierarchy.get(new_type, 0) > hierarchy.get(old_type, 0)

    def _handle_mapping_path_state(self, line_info, detected_node_id):
        """
        Handles moving along a newly discovered path. The logic is identical
        to the normal exploring state, so we reuse it.
        """
        return self._handle_exploring_path_state(line_info, detected_node_id)
                
    def _compute_utility(self, path: list, frontier_edge: tuple) -> float:
        """
        Calculates the utility score for a potential path to a frontier.
        """
        (frontier_start, frontier_end) = frontier_edge
        distance = len(path) - 1 if len(path) > 1 else 1
        info_gain = sum(1 for n in self.graph_manager.get_neighbors(frontier_end) if n not in self.visited_nodes)
        repeat_penalty = sum(1 for i in range(len(path) - 1) if tuple(sorted((path[i], path[i+1]))) in self.visited_edges)
        revisit_penalty = sum(self.revisit_count[n] for n in path)
        age = self.age_counter - self.frontier_age.get(frontier_edge, 0)

        utility = (
            self.wfd_weights["w1"] * (info_gain / distance if distance > 0 else info_gain) -
            self.wfd_weights["w2"] * repeat_penalty -
            self.wfd_weights["w3"] * revisit_penalty -
            self.wfd_weights["w4"] * age
        )
        return utility
                           
    def _handle_approaching_node(self):
        """
        Handles moving the robot forward a fixed amount to center it on the node.
        """
        self.frame_counter += 1
        node_id = self.node_approach_data["node_id"]

        # This state is deprecated in the final logic, but we keep it for safety.
        # The logic now transitions directly from EXPLORING to AT_NODE.
        # We will set a default frame_goal if not present.
        frame_goal = self.node_approach_data.get("frame_goal", 10)

        if self.frame_counter >= frame_goal:
            self.robot_comm.set_drive_command("STOP", None)
            # <<< CHANGED >>>
            self._set_state("AT_NODE", f"Finished approaching node {node_id}. Ready for analysis.")
            return f"Arrived at {node_id}. Preparing for analysis..."
        else:
            self.robot_comm.set_drive_command("FORWARD", None)
            return f"Approaching {node_id}..."

    def _get_all_global_frontiers(self):
        """
        Finds all unexplored exits from any visited node.
        """
        frontiers = []
        for node_id, exits in self.unexplored_exits.items():
            if exits:
                frontiers.append(node_id)
        return frontiers
    # In logic/robot_controller.py
# <<< REPLACE the existing _handle_initial_search_state function with this new version >>>

    def _handle_initial_search_state(self, line_info: dict | None, detected_node_id: int | None):
        """
        Handles the logic for finding the starting point on the map.
        This version is modified to use a gentle swing turn for searching,
        reserving the pivot turn only for dead-ends.
        """
        if detected_node_id is not None:
            self.robot_comm.set_drive_command("STOP", None)
            node_id = f"N{detected_node_id}"
            self.current_node_id = node_id
            self.graph_manager.add_node(node_id, position=list(self.robot_position))
            self.visited_nodes.add(node_id)
            self.revisit_count[node_id] += 1
            self.frame_counter = 0
            self.is_node_analyzed = False 
            self._set_state("AT_NODE", f"Priority 1 (Initial Search): Found starting node {node_id}.")
            return f"✅ Starting at Node {node_id}. Preparing for analysis..."

        elif line_info is not None:
            self.next_state_after_await = "EXPLORING_PATH"
            self._set_state("AWAITING_DRIVE", "Priority 2 (Initial Search): Found initial line.")
            return "Found initial line. Starting exploration..."

        else:
            self.initial_search_frame_counter += 1
            SEARCH_PATIENCE_FRAMES = 45
            if self.initial_search_frame_counter < SEARCH_PATIENCE_FRAMES:
                self.robot_comm.set_drive_command("STOP", None)
                return f"⚠️ No line or node detected. Scanning... ({self.initial_search_frame_counter}/{SEARCH_PATIENCE_FRAMES})"
            else:
                # --- THIS IS THE FIX ---
                # Use a gentle swing turn for searching instead of a pivot turn.
                SEARCH_TURN_SPEED = 190
                search_payload = {"left": 0, "right": SEARCH_TURN_SPEED}
                self.robot_comm.set_drive_command("DRIVE", search_payload)
                # --- END OF THE FIX ---
                return "⚠️ Still no line or node. Turning gently to search."
                    
    def _handle_awaiting_drive_state(self):
        """
        A simple state that waits for a few frames before transitioning.
        """
        AWAIT_FRAMES = 5
        self.robot_comm.set_drive_command("STOP", None)

        if self.await_frame_counter < AWAIT_FRAMES:
            self.await_frame_counter += 1
            return f"Pausing... ({self.await_frame_counter}/{AWAIT_FRAMES})"
        else:
            self.await_frame_counter = 0
            # <<< CHANGED >>>
            self.is_node_analyzed = False # <<< ADD THIS LINE
            self._set_state("EXPLORING_PATH", "Finished awaiting drive. Starting to explore.")
            self.frame_counter = 0
            return "Starting to explore new path."
            
    def _decide_next_exploration_target(self):
        """
        Decides the next best target. THIS IS THE FINAL VERSION that correctly calculates
        the required turn based on the robot's current heading before exploring a new local exit.
        """
        import math

        # --- Initial Filtering (same as your provided code) ---
        all_possible_exits = self.unexplored_exits.get(self.current_node_id, [])
        known_dead_ends = self.dead_end_exits.get(self.current_node_id, set())
        exits_to_check = [exit_dir for exit_dir in all_possible_exits if exit_dir not in known_dead_ends]
        if self.graph_manager.graph.has_node(self.current_node_id):
            exits_to_check = self._prune_exits_with_prediction(self.current_node_id, exits_to_check)
        print(f"DECISION: Final candidates after prediction={exits_to_check}")

        # --- Local Priority: If there are any valid local exits left, choose one ---
        if exits_to_check:
            # 1. Choose the highest priority exit (e.g., 'up' > 'left' > 'right')
            priority_map = {'up': 0, 'left': 1, 'right': 2}
            sorted_exits = sorted(exits_to_check, key=lambda exit_dir: priority_map.get(exit_dir, 99))
            chosen_exit_direction = sorted_exits[0]
            
            # 2. Convert the chosen GLOBAL direction name to a target heading angle
            direction_to_heading = {'up': -90.0, 'right': 0.0, 'down': 90.0, 'left': 180.0}
            target_heading = direction_to_heading.get(chosen_exit_direction)

            # Handle 180 vs -180 ambiguity for shortest turn calculation
            if target_heading == 180.0 and self.robot_heading == 0.0: target_heading = -180.0

            # 3. Calculate the required turn to get from current heading to target heading
            turn_angle = target_heading - self.robot_heading
            
            # Normalize the turn angle to the shortest rotation
            if turn_angle > 180: turn_angle -= 360
            if turn_angle <= -180: turn_angle += 360

            # 4. Determine the physical turn command based on the calculated angle
            turn_decision = "none"
            if abs(turn_angle) > 135:
                turn_decision = "around"
            elif turn_angle > 45:
                turn_decision = "right" # Physical right turn
            elif turn_angle < -45:
                turn_decision = "left"  # Physical left turn

            # 5. Execute the turn or depart
            self.last_node_id = self.current_node_id
            
            if turn_decision != "none":
                # If a turn is needed, store all necessary info for the centering/turning states
                self.centering_state_data = {
                    'frame_counter': 0, 
                    'turn_decision': turn_decision,      # The physical turn command (e.g., 'left')
                    'target_heading': target_heading     # The precise global heading to aim for
                }
                self._set_state("CENTERING_AT_NODE", f"Node reached. Creeping forward to execute '{turn_decision}' turn to face '{chosen_exit_direction}'.")
                return f"Centering robot before turning {turn_decision} to face {chosen_exit_direction}."
            else:
                # If already facing the correct direction, depart immediately.
                self.departing_frame_counter = 0
                self._set_state("DEPARTING_NODE", f"Local exit '{chosen_exit_direction}' chosen. Already facing correct direction. Departing.")
                return f"Choosing prioritized local exit: {chosen_exit_direction}. Moving forward."

        # --- Global Priority: If no valid local exits, find the best global frontier (same as your provided code) ---
        else:
            print("🌐 No valid local exits. Searching for the best global frontier...")
            
            all_frontiers = []
            for node_id, exits in self.unexplored_exits.items():
                known_dead_ends_for_node = self.dead_end_exits.get(node_id, set())
                valid_exits = [e for e in exits if e not in known_dead_ends_for_node]
                
                if self.graph_manager.graph.has_node(node_id):
                    valid_exits = self._prune_exits_with_prediction(node_id, valid_exits)
                
                if valid_exits:
                    all_frontiers.append(node_id)

            frontier_nodes_to_visit = [node for node in all_frontiers if node != self.current_node_id]

            if not frontier_nodes_to_visit:
                self._set_state("IDLE", "Exploration complete. No valid frontiers remain.")
                return "🎉 Exploration Complete! No frontiers left."

            best_path, min_cost = None, float('inf')
            for node_id in frontier_nodes_to_visit:
                path = self.graph_manager.get_path_astar(self.current_node_id, node_id)
                if path:
                    cost = len(path)
                    if cost < min_cost:
                        min_cost = cost
                        best_path = path
            
            if best_path:
                self.navigation_path = best_path
                self._set_state("PATHFINDING", f"No local exits. Backtracking to nearest frontier at {best_path[-1]}.")
                self.last_node_id = self.current_node_id
                print(f"🗺️ Nearest frontier is at node {best_path[-1]}. Path chosen: {best_path}")
                return f"Backtracking to nearest frontier at {best_path[-1]}."
            else:
                self._set_state("IDLE", "Exploration complete. No path to remaining frontiers.")
                return "❌ No path found to any remaining frontiers. Exploration finished."
          # In logic/robot_controller.py

    # <<< REPLACE the existing _handle_departing_node_state function with this new version >>>
    def _handle_departing_node_state(self):
        """
        Moves the robot forward for a fixed number of frames to ensure it has
        physically left the current node's area. This version is now aware of
        whether the robot is exploring or pathfinding.
        """
        DEPART_FRAMES = 50  # Move forward for 50 frames (approx 1.5-2 seconds)

        if self.departing_frame_counter < DEPART_FRAMES:
            self.robot_comm.set_drive_command("FORWARD", None)
            self.departing_frame_counter += 1
            return f"Departing node... ({self.departing_frame_counter}/{DEPART_FRAMES})"
        else:
            # Finished departing, reset counters and decide the next state based on the mission.
            self.frame_counter = 0
            self.is_node_analyzed = False
            
            # --- THIS IS THE FIX ---
            # Check if a navigation path exists. If so, we are pathfinding.
            if self.navigation_path:
                self._set_state("PATHFINDING", "Finished departure, continuing on the calculated path.")
                return "Continuing on calculated path..."
            # Otherwise, we are exploring a new frontier.
            else:
                self._set_state("EXPLORING_PATH", "Finished departure, exploring new frontier.")
                return "Exploring new path..."
            
    # <<< ADD these two new functions to the RobotController class >>>
    def _load_motion_calibration_data(self, filename="motion_calibration.yml"):
        try:
            with open(filename, 'r') as f:
                import yaml
                data = yaml.safe_load(f)
                self.calibrated_turn_frames = data
                print(f"✅ Motion calibration data loaded successfully: {self.calibrated_turn_frames}")
        except FileNotFoundError:
            print("⚠️ Motion calibration file not found. Using default values.")
        except Exception as e:
            print(f"❌ Error loading motion calibration data: {e}")

    def _save_motion_calibration_data(self, filename="motion_calibration.yml"):
        try:
            with open(filename, 'w') as f:
                import yaml
                yaml.dump(self.calibrated_turn_frames, f)
                print(f"💾 Motion calibration data saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving motion calibration data: {e}")
        
    # <<< REPLACE in logic/robot_controller.py >>>
    def _handle_motion_calibration_state(self, detected_node_id: int | None):
        sub_state = self.motion_calib_state.get('sub_state', 'idle')
        CONFIRMATION_GOAL = 5 # Must see marker for 5 frames to confirm re-acquisition
        LOSS_CONFIRMATION_GOAL = 10 # <<< ADDED: Must lose marker for 10 frames to confirm loss

        # --- STATE: START ---
        if sub_state == 'start':
            self.robot_comm.set_drive_command("STOP", None)
            if detected_node_id is not None:
                self.motion_calib_state['target_marker_id'] = detected_node_id
                self.motion_calib_state['sub_state'] = 'turn_left_start'
                # <<< ADDED: Initialize loss counter >>>
                self.motion_calib_state['loss_confirmation_counter'] = 0
                return f"Marker {detected_node_id} found. Starting LEFT turn calibration."
            return "Place robot in front of any ArUco marker to start calibration."

        # --- LEFT TURN CALIBRATION ---
        elif sub_state == 'turn_left_start':
            self.robot_comm.set_drive_command("LEFT", None)
            self.motion_calib_state['sub_state'] = 'turn_left_wait_for_loss'
            return "Calibrating LEFT turn: Waiting for marker to disappear."
        
        # <<< CHANGED: Logic for wait_for_loss is now more robust >>>
        elif sub_state == 'turn_left_wait_for_loss':
            self.robot_comm.set_drive_command("LEFT", None)
            target_marker = self.motion_calib_state.get('target_marker_id')
            
            if detected_node_id != target_marker:
                # If marker is not seen, increment loss counter
                self.motion_calib_state['loss_confirmation_counter'] += 1
            else:
                # If marker is seen again, reset counter
                self.motion_calib_state['loss_confirmation_counter'] = 0
                
            # Only proceed if marker has been lost for enough consecutive frames
            if self.motion_calib_state['loss_confirmation_counter'] >= LOSS_CONFIRMATION_GOAL:
                self.motion_calib_state['calib_frame_counter'] = 0
                self.motion_calib_state['sub_state'] = 'turn_left_wait_for_reacquire'
                
            return "Calibrating LEFT turn: Rotating..."

        elif sub_state == 'turn_left_wait_for_reacquire':
            self.robot_comm.set_drive_command("LEFT", None)
            self.motion_calib_state['calib_frame_counter'] += 1
            if detected_node_id == self.motion_calib_state.get('target_marker_id'):
                self.motion_calib_state['confirmation_counter'] += 1
                if self.motion_calib_state['confirmation_counter'] >= CONFIRMATION_GOAL:
                    self.robot_comm.set_drive_command("STOP", None)
                    self.calibrated_turn_frames['left_360'] = self.motion_calib_state['calib_frame_counter']
                    print(f"LEFT TURN 360: {self.calibrated_turn_frames['left_360']} frames.")
                    
                    self.motion_calib_state['sub_state'] = 'turn_right_start'
                    self.motion_calib_state['confirmation_counter'] = 0
                    self.motion_calib_state['loss_confirmation_counter'] = 0 # <<< ADDED: Reset for next phase
                    return f"LEFT turn calibrated. Starting RIGHT turn."
            else:
                self.motion_calib_state['confirmation_counter'] = 0
            return f"Calibrating LEFT turn: Searching for marker... [{self.motion_calib_state['calib_frame_counter']}]"

        # --- RIGHT TURN CALIBRATION ---
        elif sub_state == 'turn_right_start':
            self.robot_comm.set_drive_command("RIGHT", None)
            self.motion_calib_state['sub_state'] = 'turn_right_wait_for_loss'
            return "Calibrating RIGHT turn: Waiting for marker to disappear."

        # <<< CHANGED: Logic for wait_for_loss is now more robust >>>
        elif sub_state == 'turn_right_wait_for_loss':
            self.robot_comm.set_drive_command("RIGHT", None)
            target_marker = self.motion_calib_state.get('target_marker_id')

            if detected_node_id != target_marker:
                self.motion_calib_state['loss_confirmation_counter'] += 1
            else:
                self.motion_calib_state['loss_confirmation_counter'] = 0

            if self.motion_calib_state['loss_confirmation_counter'] >= LOSS_CONFIRMATION_GOAL:
                self.motion_calib_state['calib_frame_counter'] = 0
                self.motion_calib_state['sub_state'] = 'turn_right_wait_for_reacquire'
            return "Calibrating RIGHT turn: Rotating..."

        elif sub_state == 'turn_right_wait_for_reacquire':
            self.robot_comm.set_drive_command("RIGHT", None)
            self.motion_calib_state['calib_frame_counter'] += 1
            if detected_node_id == self.motion_calib_state.get('target_marker_id'):
                self.motion_calib_state['confirmation_counter'] += 1
                if self.motion_calib_state['confirmation_counter'] >= CONFIRMATION_GOAL:
                    self.robot_comm.set_drive_command("STOP", None)
                    self.calibrated_turn_frames['right_360'] = self.motion_calib_state['calib_frame_counter']
                    print(f"RIGHT TURN 360: {self.calibrated_turn_frames['right_360']} frames.")
                    self._save_motion_calibration_data()
                    self._set_state("IDLE", "Motion calibration complete.")
                    return f"RIGHT turn calibrated. Calibration complete and values saved."
            else:
                self.motion_calib_state['confirmation_counter'] = 0
            return f"Calibrating RIGHT turn: Searching for marker... [{self.motion_calib_state['calib_frame_counter']}]"
        
        return "Motion calibration in unknown state."
        
    # <<< ADD this new helper function to the RobotController class >>>
    def _get_average_edge_length(self):
        """Calculates the average length of all explored edges in the graph."""
        if not self.visited_edges:
            return 150 * self.pixels_per_frame # Return a default guess if no edges exist yet

        total_cost = 0
        edge_count = 0
        for u, v, data in self.graph_manager.graph.edges(data=True):
            cost = data.get('weight', 0)
            if cost > 0:
                total_cost += cost
                edge_count += 1
        
        if edge_count == 0:
            return 150 * self.pixels_per_frame
            
        # The cost is in frames, convert it to pixels
        return (total_cost / edge_count) * self.pixels_per_frame
    
    # <<< ADD this new prediction function to the RobotController class >>>
    def _prune_exits_with_prediction(self, current_node_id, exits_to_check):
        """
        Uses geometry to predict where exits lead. If an exit likely leads to an
        already visited node, it is "pruned" from the list of candidates to explore.
        """
        from math import cos, sin, radians

        # If there's nothing to check, exit early
        if not exits_to_check:
            return []

        pruned_exits = list(exits_to_check)
        current_pos = self.graph_manager.get_node_attribute(current_node_id, 'position')
        if not current_pos:
            return pruned_exits # Cannot predict without a starting position

        avg_length = self._get_average_edge_length()
        MATCH_THRESHOLD = avg_length * 0.5 # Match if predicted pos is within 50% of avg length

        # Map exit directions to heading angles
        direction_to_heading = {'up': -90, 'right': 0, 'down': 90, 'left': 180}

        for exit_dir in exits_to_check:
            heading = direction_to_heading.get(exit_dir)
            if heading is None: continue

            # Calculate the predicted coordinates of the destination
            predicted_x = current_pos[0] + avg_length * cos(radians(heading))
            predicted_y = current_pos[1] + avg_length * sin(radians(heading))

            # Check if these coordinates match any OTHER visited node
            for visited_node_id in self.visited_nodes:
                if visited_node_id == current_node_id: continue
                
                visited_node_pos = self.graph_manager.get_node_attribute(visited_node_id, 'position')
                if not visited_node_pos: continue

                # Calculate distance between predicted and actual positions
                dist = ((predicted_x - visited_node_pos[0])**2 + (predicted_y - visited_node_pos[1])**2)**0.5

                if dist < MATCH_THRESHOLD:
                    print(f"🧠 PREDICTION: Exit '{exit_dir}' from {current_node_id} likely leads to known node {visited_node_id}. Pruning this path.")
                    
                    # Prune the exit from the list
                    if exit_dir in pruned_exits:
                        pruned_exits.remove(exit_dir)

                    # Add the predicted edge to the graph so the map is complete
                    if not self.graph_manager.graph.has_edge(current_node_id, visited_node_id):
                        self.graph_manager.add_edge(current_node_id, visited_node_id, cost=int(avg_length / self.pixels_per_frame), 
                                                    exit_from_A=exit_dir, entry_to_B='predicted')
                    break # Move to the next exit to check
                    
        return pruned_exits               

    # In logic/robot_controller.py
    # <<< ADD THIS ENTIRE NEW HELPER FUNCTION to the RobotController class >>>
    def _calculate_edge_directions_from_geometry(self, pos_a: list, pos_b: list) -> dict:
        """
        Calculates the exit/entry directions between two nodes based on their
        geometric positions, avoiding reliance on the error-prone robot_heading variable.
        """
        import math
        
        if pos_a is None or pos_b is None:
            return {"exit_from_A": "unknown", "entry_to_B": "unknown"}

        delta_x = pos_b[0] - pos_a[0]
        delta_y = pos_b[1] - pos_a[1]
        
        # Get the angle in degrees
        angle = math.degrees(math.atan2(delta_y, delta_x))

        # Determine the primary direction based on the angle
        if -45 <= angle < 45:
            direction = "right"
            opposite = "left"
        elif 45 <= angle < 135:
            direction = "down"
            opposite = "up"
        elif 135 <= angle <= 180 or -180 <= angle < -135:
            direction = "left"
            opposite = "right"
        else:  # -135 <= angle < -45
            direction = "up"
            opposite = "down"
            
        return {"exit_from_A": direction, "entry_to_B": opposite}
    
# In logic/robot_controller.py
# <<< REPLACE the existing _handle_pathfinding_state function with this new version >>>
    def _handle_pathfinding_state(self, line_info, detected_node_id):
        """
        Handles robustly following a pre-calculated path. It first orients the robot
        correctly towards the next node, then follows the line until it arrives.
        """
        import math

        # --- Scenario 1: Robot is at a node and needs to decide the next turn ---
        if self.current_node_id == self.navigation_path[0]:
            # If path is too short or invalid, we've arrived or there's an error.
            if len(self.navigation_path) < 2:
                self.navigation_path = []
                self._set_state("AT_NODE", f"Arrived at pathfinding destination: {self.current_node_id}")
                return f"Arrived at destination {self.current_node_id}."

            next_node = self.navigation_path[1]
            self.last_node_id = self.current_node_id
            
            # Calculate the required turn using GEOMETRY for maximum accuracy
            current_pos = self.graph_manager.get_node_attribute(self.current_node_id, 'position')
            next_pos = self.graph_manager.get_node_attribute(next_node, 'position')

            if not current_pos or not next_pos:
                self._set_state("IDLE", "FATAL ERROR: Cannot pathfind without node positions.")
                return "Error: Missing position data for nodes."

            delta_x = next_pos[0] - current_pos[0]
            delta_y = next_pos[1] - current_pos[1]
            target_heading = math.degrees(math.atan2(delta_y, delta_x))

            turn_angle = target_heading - self.robot_heading
            
            # Normalize the turn angle to the shortest rotation
            if turn_angle > 180: turn_angle -= 360
            if turn_angle < -180: turn_angle += 360

            # --- CORRECTED DECISION LOGIC ---
            # The 180-degree check (around) now correctly happens FIRST.
            turn_decision = "none"
            if abs(turn_angle) > 135:
                turn_decision = "around"
            elif turn_angle > 45:
                turn_decision = "right"
            elif turn_angle < -45:
                turn_decision = "left"
            
            # Execute the turn by calling the improved _initiate_turn
            if turn_decision != "none":
                # We pass the PRECISE target_heading to the turn function for accuracy
                self._initiate_turn(turn_decision, target_heading=target_heading)
                self.turn_state['next_state_after_turn'] = "DEPARTING_NODE"
                self._set_state("TURNING", f"Pathfinding: Need to turn '{turn_decision}' to face {next_node}.")
            else: # Already facing the correct direction
                self.departing_frame_counter = 0
                self._set_state("DEPARTING_NODE", "Pathfinding: Already facing correct direction. Departing.")
            
            # We've processed the current node, so we remove it from the path
            self.navigation_path.pop(0)
            return f"Orienting for path to {next_node}..."

        # --- Scenario 2: Robot is between nodes, so it just follows the line ---
        else:
            # Re-use the standard line-following logic from the exploration state
            return self._handle_exploring_path_state(line_info, detected_node_id)
        
# ADD THIS NEW HELPER FUNCTION
    def _log_decision_details_at_node(self, node_id, raw_analysis_results, stable_exits, explored_exits, final_unexplored_exits):
        """
        Logs a detailed summary of the decision-making process at a node.
        """
        from collections import Counter

        print("\n" + "#" * 25 + f" DECISION LOG AT NODE {node_id} " + "#" * 25)
        
        # 1. Log raw vision data
        if raw_analysis_results:
            hashable_results = [tuple(sorted(res.get('exits', []))) for res in raw_analysis_results]
            detection_counts = Counter(hashable_results)
            print("  [Vision] Raw Detections (30 frames):")
            for exits, count in detection_counts.items():
                print(f"    - Exits {list(exits)} detected {count} times.")
        else:
            print("  [Vision] No raw intersection data was collected.")

        # 2. Log the stable outcome
        print(f"\n  [Analysis] Stable Physical Exits Identified: {list(stable_exits)}")

        # 3. Log what the robot already knows
        print(f"  [Memory] Previously Explored Exits from this Node: {list(explored_exits)}")

        # 4. Log the final "To-Do List"
        print(f"  [Decision] Final Unexplored Exits (To-Do List): {final_unexplored_exits}")
        
        if not final_unexplored_exits:
            print("    -> No local exits remain. Will search for a global frontier.")
        
        print("#" * 80 + "\n")

# ADD THIS NEW, CRUCIAL HELPER FUNCTION
    def _transform_local_exits_to_global(self, local_exits: set) -> set:
        """
        Transforms perceived local exits (up, left, right) into global map directions
        based on the robot's current absolute heading.
        """
        # Mapping of local direction names to their relative angle
        local_to_angle_offset = {'up': 0, 'left': -90, 'right': 90, 'down': 180}

        # Mapping of absolute global angles to global direction names
        # NOTE: -180 and 180 are both 'left' on the map
        global_angle_to_name = {
            -90: 'up',
            0: 'right',
            90: 'down',
            180: 'left',
            -180: 'left'
        }

        global_exits = set()
        robot_heading = self.robot_heading # Current global heading (e.g., -90 for up, 0 for right)

        for local_dir in local_exits:
            angle_offset = local_to_angle_offset.get(local_dir, 0)
            
            # Calculate the global angle of the exit
            global_angle = robot_heading + angle_offset

            # Normalize the angle to be within the range [-180, 180]
            if global_angle > 180:
                global_angle -= 360
            if global_angle <= -180: # Use <= to handle -180 correctly
                global_angle += 360

            # Find the closest global direction name for the calculated angle
            # This handles small inaccuracies in heading
            closest_angle = min(global_angle_to_name.keys(), key=lambda angle: abs(angle - global_angle))
            global_dir_name = global_angle_to_name[closest_angle]
            
            global_exits.add(global_dir_name)
            
        return global_exits
    # In logic/robot_controller.py
# In logic/robot_controller.py

    def _handle_at_node_state(self, intersection_info: dict | None):
        """
        Handles all logic at a node. This final version uses a DYNAMIC approach
        to determine unexplored exits and GEOMETRIC calculations for edge directions,
        and correctly transforms LOCAL vision data to GLOBAL map coordinates.
        """
        from math import cos, sin, radians
        from collections import Counter
        self.robot_comm.set_drive_command("STOP", None)

        # --- Phase 1: Update graph with arrival info (runs only once per new connection) ---
        if not self.is_node_analyzed:
            newly_arrived_node_id = self.node_approach_data.get("node_id")
            if newly_arrived_node_id:
                # Add node to graph if it's brand new
                if not self.graph_manager.graph.has_node(newly_arrived_node_id):
                    new_position = None
                    if self.last_node_id is None:
                        new_position = list(self.robot_position)
                    else:
                        last_node_pos = self.graph_manager.get_node_attribute(self.last_node_id, 'position')
                        if last_node_pos:
                            distance = self.frame_counter * self.pixels_per_frame
                            heading_rad = radians(self.robot_heading)
                            new_x = last_node_pos[0] + distance * cos(heading_rad)
                            new_y = last_node_pos[1] + distance * sin(heading_rad)
                            self.robot_position = (new_x, new_y)
                            new_position = [new_x, new_y]
                    self.graph_manager.add_node(newly_arrived_node_id, position=new_position or [150, 150])

                # Add edge to graph if it's a new connection using GEOMETRY
                if self.last_node_id and not self.graph_manager.graph.has_edge(self.last_node_id, newly_arrived_node_id):
                    cost = self.frame_counter
                    last_node_pos = self.graph_manager.get_node_attribute(self.last_node_id, 'position')
                    current_node_pos = self.graph_manager.get_node_attribute(newly_arrived_node_id, 'position')
                    directions = self._calculate_edge_directions_from_geometry(last_node_pos, current_node_pos)
                    exit_direction = directions['exit_from_A']
                    entry_direction = directions['entry_to_B']
                    self.graph_manager.add_edge(self.last_node_id, newly_arrived_node_id, cost=cost, exit_from_A=exit_direction, entry_to_B=entry_direction)
                    print(f"🔗 GEOMETRIC Edge added {self.last_node_id} -> {newly_arrived_node_id} (Exit from {exit_direction}, Entry via {entry_direction})")
                    self._remotely_update_source_node_status()

                # Update robot's current state variables
                self.current_node_id = newly_arrived_node_id
                if newly_arrived_node_id not in self.visited_nodes:
                    self.visited_nodes.add(self.current_node_id)
                self.revisit_count[self.current_node_id] += 1
                self.frame_counter = 0
                self.node_approach_data = {}
                self.is_node_analyzed = True
                self.node_analysis_data = {'frame_count': 0, 'results': []}

        # --- Phase 2: Analyze intersection over several frames ---
        if 'frame_count' not in self.node_analysis_data: self.node_analysis_data = {'frame_count': 0, 'results': []}
        self.node_analysis_data['frame_count'] += 1
        if intersection_info: self.node_analysis_data['results'].append(intersection_info)
        
        analysis_frame_goal = 30
        if self.node_analysis_data['frame_count'] < analysis_frame_goal:
            return f"Analyzing node {self.current_node_id}... ({self.node_analysis_data['frame_count']}/{analysis_frame_goal})"

        # --- Phase 3: DYNAMICALLY determine unexplored exits, Print Debug Info, Decide, and Exit ---
        else:
            # 1. Get the stable, LOCAL physical exits from vision analysis
            stable_local_physical_exits = set()
            if self.node_analysis_data['results']:
                hashable_results = [tuple(sorted(res['exits'])) for res in self.node_analysis_data['results']]
                if hashable_results:
                    stable_local_physical_exits = set(Counter(hashable_results).most_common(1)[0][0])
            
            # 2. Transform local exits to GLOBAL exits using the robot's heading
            stable_global_physical_exits = self._transform_local_exits_to_global(stable_local_physical_exits)

            # 3. Get already explored GLOBAL exits from memory
            explored_exits = set()
            if self.graph_manager.graph.has_node(self.current_node_id):
                current_node_pos = self.graph_manager.get_node_attribute(self.current_node_id, 'position')
                for neighbor in self.graph_manager.get_neighbors(self.current_node_id):
                    neighbor_pos = self.graph_manager.get_node_attribute(neighbor, 'position')
                    directions = self._calculate_edge_directions_from_geometry(current_node_pos, neighbor_pos)
                    exit_to_neighbor = directions.get('exit_from_A')
                    if exit_to_neighbor:
                        explored_exits.add(exit_to_neighbor)
            
            # 4. The true "To-Do List" is the difference of two GLOBAL sets
            current_unexplored_exits = list(stable_global_physical_exits - explored_exits)
            
            # 5. Update the central memory with this fresh list
            self.unexplored_exits[self.current_node_id] = current_unexplored_exits

            # Call the logging function (with updated variable names for clarity)
            self._log_decision_details_at_node(
                node_id=self.current_node_id,
                raw_analysis_results=self.node_analysis_data.get('results', []),
                stable_exits=stable_global_physical_exits, # Log the transformed global exits
                explored_exits=explored_exits,
                final_unexplored_exits=current_unexplored_exits
            )
            
            # 6. Make a decision and EXIT
            self.node_analysis_data = {}
            decision_log = self._decide_next_exploration_target()
            self.is_node_analyzed = False
            
            return f"Node analysis complete for {self.current_node_id}. {decision_log}"
                
    # In vision/image_processor.py
    # <<< REPLACE the existing analyze_intersection function with this new version >>>
    def analyze_intersection(self, bird_eye_frame: np.ndarray) -> dict | None:
        """
        This version has UPDATED classification logic to better distinguish
        between T-junctions and simple corners, resolving the user's issue.
        """
        corners, ids, _ = self.aruco_detector.detectMarkers(bird_eye_frame)
        if ids is None:
            return None

        # --- 1. Get Marker's Center ---
        marker_corners = corners[0][0]
        marker_center_x = int(np.mean(marker_corners[:, 0]))
        marker_center_y = int(np.mean(marker_corners[:, 1]))

        # --- 2. Create Binary Image for Probing ---
        processed_frame = bird_eye_frame.copy()
        cv2.fillPoly(processed_frame, [np.int32(marker_corners)], (255, 255, 255))
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)

        # --- 3. Define the Probe ROIs ---
        PROBE_DISTANCE_FROM_CENTER = 35 
        probe_w, probe_h = 25, 25
        p_dist = PROBE_DISTANCE_FROM_CENTER
        
        y1_up, y2_up = marker_center_y - p_dist - probe_h, marker_center_y - p_dist
        x1_up, x2_up = marker_center_x - probe_w // 2, marker_center_x + probe_w // 2
        roi_up = binary[y1_up:y2_up, x1_up:x2_up]

        y1_left, y2_left = marker_center_y - probe_h // 2, marker_center_y + probe_h // 2
        x1_left, x2_left = marker_center_x - p_dist - probe_w, marker_center_x - p_dist
        roi_left = binary[y1_left:y2_left, x1_left:x2_left]

        y1_right, y2_right = marker_center_y - probe_h // 2, marker_center_y + probe_h // 2
        x1_right, x2_right = marker_center_x + p_dist, marker_center_x + p_dist + probe_w
        roi_right = binary[y1_right:y2_right, x1_right:x2_right]

        # --- 4. Probe Each ROI ---
        INTENSITY_THRESHOLD = 40 
        exits = []
        if roi_up.size > 0 and np.mean(roi_up) > INTENSITY_THRESHOLD:
            exits.append("up")
        if roi_left.size > 0 and np.mean(roi_left) > INTENSITY_THRESHOLD:
            exits.append("left")
        if roi_right.size > 0 and np.mean(roi_right) > INTENSITY_THRESHOLD:
            exits.append("right")

        # <<< --- START OF THE LOGIC FIX --- >>>
        # --- 5. UPDATED Intersection Classification Logic ---
        num_exits = len(exits)
        node_type = "unknown"
        exits_set = set(exits)

        if num_exits == 3:
            node_type = '4-way'  # All 3 probes detected a line -> crossroad
        
        elif num_exits == 2:
            # A T-junction allows going straight OR turning, or forces a left/right turn.
            # A corner is the combination of the entry path and one turn.
            if 'up' in exits_set or ('left' in exits_set and 'right' in exits_set):
                node_type = '3-way' # This is a classic T-junction
            else:
                # e.g., {'down', 'left'} or {'down', 'right'} from the robot's local view
                node_type = 'Corner' # This is a simple 90-degree turn
                
        elif num_exits == 1:
            if 'up' in exits:
                node_type = 'straight'
            elif 'left' in exits:
                node_type = 'Turn-left'
            elif 'right' in exits:
                node_type = 'Turn-right'
                
        elif num_exits == 0:
            node_type = 'dead-end'

        # <<< ---  END OF THE LOGIC FIX  --- >>>

        return {"type": node_type, "exits": exits}
  
# In logic/robot_controller.py
    # =================================================================================
    # START OF THE FIX: REPLACE the existing _remotely_update_source_node_status function
    # =================================================================================
    def _remotely_update_source_node_status(self):
        """
        Remotely updates the 'unexplored_exits' list of the previous node
        after successfully traversing an edge. This prevents unnecessary backtracking.
        This version is more robust and uses a try-except block to handle potential
        mismatches between geometric and vision-based exit names.
        """
        # Ensure we have a valid last node and current node
        if not self.last_node_id or not self.current_node_id:
            return

        # Get the data for the edge we just traversed
        edge_data = self.graph_manager.graph.get_edge_data(self.last_node_id, self.current_node_id)
        if not edge_data:
            return

        # Find out which exit we took from the last node to get here
        exit_from_last_node = edge_data.get('exit_from_A')

        # --- THIS IS THE FIX ---
        # Instead of checking if the exit exists before removing, we directly try to remove it.
        # This is more robust against small mismatches (e.g., from vision vs. geometry).
        if exit_from_last_node and self.last_node_id in self.unexplored_exits:
            try:
                self.unexplored_exits[self.last_node_id].remove(exit_from_last_node)
                print(f"🧠 REMOTE UPDATE: Removed exit '{exit_from_last_node}' from node {self.last_node_id}'s to-do list.")
            except ValueError:
                # This block catches the error if .remove() is called on a list for an item that doesn't exist.
                # This can happen if the geometrically calculated exit name doesn't perfectly match
                # the name from the vision analysis phase. This acts as a failsafe.
                print(f"⚠️ REMOTE UPDATE WARNING: Tried to remove exit '{exit_from_last_node}' from {self.last_node_id}, but it wasn't in the to-do list: {self.unexplored_exits[self.last_node_id]}")
    # =================================================================================
    # END OF THE FIX
    # =================================================================================

    # In logic/robot_controller.py
    # <<< ADD THIS ENTIRE NEW FUNCTION to the RobotController class >>>
    def _handle_centering_at_node_state(self):
        """
        Handles the "creep forward" action. This UPDATED version uses the
        precise turning information calculated by the decision function.
        """
        CENTERING_FRAMES = 0

        if self.centering_state_data['frame_counter'] < CENTERING_FRAMES:
            self.robot_comm.set_drive_command("FORWARD", None)
            self.centering_state_data['frame_counter'] += 1
            return f"Centering at node... ({self.centering_state_data['frame_counter']}/{CENTERING_FRAMES})"
        else:
            # Centering is complete. Now, initiate the PRECISE turn.
            turn_cmd = self.centering_state_data.get('turn_decision')
            target_hdg = self.centering_state_data.get('target_heading')
            
            if turn_cmd:
                # We pass BOTH the physical turn command and the precise target heading
                self._initiate_turn(turn_cmd, target_heading=target_hdg)
                self._set_state("TURNING", f"Centering complete. Initiating '{turn_cmd}' turn.")
                return f"Initiating '{turn_cmd}' turn."
            else:
                # Failsafe
                self._set_state("IDLE", "Error: Lost turn direction after centering.")
                return "Error: Could not initiate turn."

    def _handle_exploring_path_state(self, line_info: dict | None, detected_node_id: int | None):
        MIN_FRAMES_BEFORE_NODE_DETECTION = 15


        node_id_str = f"N{detected_node_id}" if detected_node_id is not None else None
        
        # --- Priority 1: A new node is detected ---
        if node_id_str and node_id_str != self.last_node_id and self.frame_counter > MIN_FRAMES_BEFORE_NODE_DETECTION:
            self.robot_comm.set_drive_command("STOP", None)
            self.node_approach_data = {"node_id": node_id_str}
            self._set_state("AT_NODE", f"Node {node_id_str} sighted after {self.frame_counter} frames. Stopped for analysis.")
            return f"Arrived at {node_id_str}. Preparing for analysis."

        # --- Priority 2: A line is visible, so follow it ---
        if line_info:
            self.no_line_frame_count = 0
            positional_error = line_info.get("positional_error", 0)
            angle_error = line_info.get("angle_error", 0)

            steering_correction = 0.0
            CENTER_THRESHOLD = 45
            OUTER_THRESHOLD = 75
            
            if abs(positional_error) > OUTER_THRESHOLD:
                Kp_strong = 1.8
                steering_correction = Kp_strong * positional_error
            elif abs(positional_error) > CENTER_THRESHOLD:
                Kp_normal = 1.0
                steering_correction = Kp_normal * positional_error
            else:
                Ka = 2.8
                steering_correction = Ka * angle_error

            MAX_SPEED = 190
            left_speed = MAX_SPEED
            right_speed = MAX_SPEED
            
            if steering_correction > 0: 
                right_speed -= steering_correction
            elif steering_correction < 0:
                left_speed += steering_correction

            left_speed = max(0, min(MAX_SPEED, int(left_speed)))
            right_speed = max(0, min(MAX_SPEED, int(right_speed)))
        
            command_payload = {"left": left_speed, "right": right_speed}
            self.robot_comm.set_drive_command("DRIVE", command_payload)
            
            if positional_error < -15: self.last_known_line_direction = "LEFT"
            elif positional_error > 15: self.last_known_line_direction = "RIGHT"
            
            self.frame_counter += 1
            return f"Exploring... PosErr:{positional_error}, AngErr:{angle_error:.1f} -> L:{left_speed}, R:{right_speed}"
        
        # --- Priority 3: The line has been lost ---
        else:
            self.no_line_frame_count += 1
            if self.no_line_frame_count > 30:
                # <<< CHANGED SECTION: Logic to remember the dead end >>>
                
                # 1. Identify which exit from the last node led to this dead end
                exit_direction = self.turn_state.get('direction', 'unknown')
                
                # 2. If the exit is known, add it to the dead-end memory for that node
                if self.last_node_id and exit_direction != 'unknown':
                    self.dead_end_exits[self.last_node_id].add(exit_direction)
                    print(f"🧠 MEMORY: Exit '{exit_direction}' from node {self.last_node_id} is now marked as a dead end.")
                
                # 3. Always perform a 180-degree turn at a dead end
                self._initiate_turn('around')
                self._set_state("TURNING_AROUND", f"Line lost for {self.no_line_frame_count} frames (Dead-end).")
                self.no_line_frame_count = 0
                return f"⚠️ Dead-end! Turning around."
                # <<< END CHANGED SECTION >>>
            else:
                self.robot_comm.set_drive_command("STOP", None)
                return f"Line lost, stopping... ({self.no_line_frame_count}/30)"


# In logic/robot_controller.py
# <<< REPLACE the existing _initiate_turn function with this new version >>>

    # def _initiate_turn(self, direction: str, target_heading: float = None):
    #     """
    #     Initiates a turn. This version is modified so that ALL turns, including
    #     the 180-degree 'around' turn, use a more stable swing turn mechanism
    #     instead of a pivot turn, preventing the use of the "initial search" turn style at nodes.
    #     """
    #     # 1. Update robot's logical heading (No changes here)
    #     if target_heading is not None:
    #         self.robot_heading = target_heading
    #     else:
    #         if direction == "left": self.robot_heading -= 90
    #         elif direction == "right": self.robot_heading += 90
    #         elif direction == "around": self.robot_heading += 180

    #     if self.robot_heading > 180: self.robot_heading -= 360
    #     if self.robot_heading <= -180: self.robot_heading += 360
    #     print(f"🔄 Robot heading updated to: {self.robot_heading:.1f} degrees.")

    #     # --- START OF THE MODIFIED LOGIC ---

    #     # 2. Determine physical turn command, payload, and duration for ALL turns
        
    #     turn_cmd = "DRIVE"  # All turns will now use the DRIVE command for consistency
    #     turn_payload = None
    #     turn_goal_frames = 40 # Default value
    #     TURN_SPEED = 220     # The speed of the moving wheel during a swing turn
        
    #     # --- Logic for 90-degree turns ---
    #     if direction == "left":
    #         print(f"Executing SWING turn for direction: {direction}")
    #         turn_payload = {"left": 0, "right": TURN_SPEED}
    #         turn_goal_frames = self.calibrated_turn_frames.get('left_360', 100) // 4

    #     elif direction == "right":
    #         print(f"Executing SWING turn for direction: {direction}")
    #         turn_payload = {"left": TURN_SPEED, "right": 0}
    #         turn_goal_frames = self.calibrated_turn_frames.get('right_360', 100) // 4

    #     # --- MODIFIED Logic for 180-degree turns ---
    #     elif direction == "around":
    #         print("Executing 180-degree SWING turn instead of a pivot turn.")
    #         # We'll perform a 180-degree swing turn to the left by default.
    #         # This is more stable than a pivot turn.
    #         turn_payload = {"left": 0, "right": TURN_SPEED}
    #         # The duration is double that of a 90-degree turn.
    #         turn_goal_frames = self.calibrated_turn_frames.get('left_360', 100) // 2
        
    #     # 3. Initialize the state machine for the turn
    #     self.turn_logic_state = {
    #         "phase": "BLIND_TURN",
    #         "frame_counter": 0,
    #         "align_frame_counter": 0,
    #         "confirmation_counter": 0,
    #         "turn_goal_frames": turn_goal_frames,
    #         "direction": turn_cmd, 
    #         "payload": turn_payload 
    #     }
        
    #     print(f"Initiating calibrated swing turn: {turn_cmd} with payload {turn_payload} for {turn_goal_frames} frames.")

# In logic/robot_controller.py
# <<< REPLACE the existing _initiate_turn function with this new version >>>

    def _initiate_turn(self, direction: str, target_heading: float = None):
        """
        Initiates a turn based on the required type.
        - For 'left'/'right': Uses a gentle swing turn (one wheel stops, one moves).
        - For 'around': Uses a fast pivot turn (wheels move in opposite directions).
        This is compatible with the limited ESP32 firmware.
        """
        # 1. Update robot's logical heading (No changes here)
        if target_heading is not None:
            self.robot_heading = target_heading
        else:
            if direction == "left": self.robot_heading -= 90
            elif direction == "right": self.robot_heading += 90
            elif direction == "around": self.robot_heading += 180

        if self.robot_heading > 180: self.robot_heading -= 360
        if self.robot_heading <= -180: self.robot_heading += 360
        print(f"🔄 Robot heading updated to: {self.robot_heading:.1f} degrees.")

        # --- START OF THE NEW LOGIC ---

        # 2. Determine physical turn command and duration
        # This section is now split based on the turn type.
        
        turn_cmd = "STOP" # Default command
        turn_payload = None
        
        # --- Gentle Swing Turns for Intersections ('left' or 'right') ---
        if direction in ["left", "right"]:
            print(f"Executing SWING turn for direction: {direction}")
            turn_cmd = "DRIVE" # We will send specific wheel speeds
            TURN_SPEED = 220 # The speed of the moving wheel
            
            if direction == "left":
                # To turn left, stop the left wheel and move the right wheel forward.
                turn_payload = {"left": 0, "right": TURN_SPEED}
            else: # direction == "right"
                # To turn right, stop the right wheel and move the left wheel forward.
                turn_payload = {"left": TURN_SPEED, "right": 0}
                
            turn_duration_map = {
                "left": self.calibrated_turn_frames.get('left_360', 120) // 4,
                "right": self.calibrated_turn_frames.get('right_360', 120) // 4
            }
            turn_goal_frames = turn_duration_map.get(direction, 25)

        # --- Fast Pivot Turn for Dead-Ends ('around') ---
        elif direction == "around":
            print("Executing PIVOT turn for a dead-end.")
            # For a 180-degree turn, we use the simple LEFT/RIGHT commands.
            # The robot_communicator will translate these to pivot turns (e.g., left=200, right=-200).
            # This is the ONLY place we use this method now.
            if self.last_known_line_direction == "RIGHT":
                turn_cmd = "RIGHT"
                turn_goal_frames = self.calibrated_turn_frames.get('right_360', 120) // 2
            else:
                turn_cmd = "LEFT"
                turn_goal_frames = self.calibrated_turn_frames.get('left_360', 120) // 2
        
        # 3. Initialize the state machine for the turn
        self.turn_logic_state = {
            "phase": "BLIND_TURN",
            "frame_counter": 0,
            "align_frame_counter": 0,
            "confirmation_counter": 0,
            "turn_goal_frames": turn_goal_frames,
            "direction": turn_cmd, # This can be 'DRIVE', 'LEFT', or 'RIGHT'
            "payload": turn_payload # This will be None for 'around' turns
        }
        
        print(f"Initiating calibrated turn: {turn_cmd} with payload {turn_payload} for {turn_goal_frames} frames.")

# In logic/robot_controller.py
# REPLACE the existing _handle_turning_state function with this new version

    def _handle_turning_state(self, line_info: dict | None):
        """
        Handles the multi-phase turning process.
        This version has an increased timeout to prevent failures in dead-ends.
        """
        turn_logic = self.turn_logic_state
        turn_cmd = turn_logic.get("direction", "LEFT")
        turn_payload = turn_logic.get("payload", None)
        
        CONFIRMATION_GOAL = 10
        # <<< --- THIS IS THE FIX --- >>>
        # Increased the timeout from half a turn to a full 3/4 turn duration
        # to give more time for alignment, especially after 180-degree turns.
        ALIGN_TIMEOUT_FRAMES = self.calibrated_turn_frames.get('right_360', 100) * 3 // 4
        # <<< --- END OF THE FIX --- >>>

        # --- Phase 1: BLIND_TURN ---
        if turn_logic.get("phase") == "BLIND_TURN":
            self.robot_comm.set_drive_command(turn_cmd, turn_payload)
            turn_logic["frame_counter"] += 1
            
            if turn_logic["frame_counter"] >= turn_logic["turn_goal_frames"]:
                turn_logic["phase"] = "ALIGN"
                turn_logic['align_frame_counter'] = 0
                turn_logic['has_attempted_recovery'] = False 
                print("DEBUG (Turn): Calibrated blind turn complete. Now aligning with vision.")
            
            return f"Turning (Blind Phase)... {turn_logic['frame_counter']}/{turn_logic['turn_goal_frames']}"

        # --- Phase 2: ALIGN ---
        elif turn_logic.get("phase") == "ALIGN":
            turn_logic['align_frame_counter'] += 1

            if turn_logic['align_frame_counter'] > ALIGN_TIMEOUT_FRAMES:
                if turn_logic.get('has_attempted_recovery', False):
                    print(f"❌ RECOVERY FAILED: Timeout reached again after {ALIGN_TIMEOUT_FRAMES} frames.")
                    self.robot_comm.set_drive_command("STOP", None)
                    self._set_state("IDLE", "Turn recovery failed due to timeout.")
                    return "ERROR: Turn recovery timed out."
                else:
                    print(f"⚠️ ALIGN TIMEOUT: Attempting recovery turn.")
                    turn_logic['phase'] = 'RECOVERY'
                    turn_logic['recovery_frame_counter'] = 0
                    turn_logic['has_attempted_recovery'] = True
                    return "Align timed out. Starting recovery turn."

            if line_info:
                turn_logic["confirmation_counter"] += 1
            else:
                turn_logic["confirmation_counter"] = 0
            
            if turn_logic["confirmation_counter"] >= CONFIRMATION_GOAL:
                print("✅ Turn successful: Line re-acquired and confirmed.")
                self.robot_comm.set_drive_command("STOP", None)
                
                next_state = self.turn_state.get('next_state_after_turn', "DEPARTING_NODE")
                if next_state == "DEPARTING_NODE":
                    self.departing_frame_counter = 0

                self._set_state(next_state, "Turn complete with confirmation.")
                self.turn_logic_state = {}
                return "Turn complete. Now departing."
            
            else:
                self.robot_comm.set_drive_command(turn_cmd, turn_payload)
                return f"Turning (Align Phase)... Searching [{turn_logic['confirmation_counter']}/{CONFIRMATION_GOAL}]"

        # --- Phase 3: RECOVERY ---
        elif turn_logic.get("phase") == "RECOVERY":
            turn_logic['recovery_frame_counter'] += 1

            if turn_logic['recovery_frame_counter'] > ALIGN_TIMEOUT_FRAMES:
                turn_logic['phase'] = 'ALIGN'
                turn_logic['align_frame_counter'] = 0
                turn_logic['confirmation_counter'] = 0
                print("DEBUG (Turn): Recovery turn complete. Re-attempting alignment.")
                return "Recovery complete. Re-aligning..."
            
            else:
                # For recovery, we always use a simple pivot turn
                opposite_turn = "RIGHT" if turn_cmd == "LEFT" else "LEFT"
                self.robot_comm.set_drive_command(opposite_turn, None)
                return f"Recovery Turn ({opposite_turn})... {turn_logic['recovery_frame_counter']}/{ALIGN_TIMEOUT_FRAMES}"

        # Failsafe
        self._set_state("IDLE", "ERROR: Unknown turn phase.")
        return "ERROR: Unknown turn phase. Stopping."