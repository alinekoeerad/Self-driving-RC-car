# vision/image_processor.py

from vision.perspective_transformer import PerspectiveTransformer
import cv2
import numpy as np
import yaml
import math
from config import *

class ImageProcessor:
    def __init__(self):
        """Initializes all vision components."""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.perspective_transformer = PerspectiveTransformer()
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # This line improves the accuracy of the detected corner locations.
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Load any existing data on startup
        self.load_calibration_data()
        self.perspective_transformer.load_perspective_data()

# vision/image_processor.py
# In vision/image_processor.py

    def draw_diagnostics(self, frame: np.ndarray, line_info: dict | None, node_info: tuple, intersection_analysis: dict | None) -> np.ndarray:
        """
        Draws all diagnostic information onto the frame, including all FOUR 
        intersection probe ROIs.
        """
        diag_frame = frame.copy()
        node_roi_coords, line_roi_coords = self._get_rois(diag_frame.shape)
        (nx1, ny1, nx2, ny2), (lx1, ly1, lx2, ly2) = node_roi_coords, line_roi_coords

        # --- Draw Node and Line ROIs ---
        cv2.rectangle(diag_frame, (nx1, ny1), (nx2, ny2), (255, 100, 100), 2)
        cv2.rectangle(diag_frame, (lx1, ly1), (lx2, ly2), (100, 255, 100), 2)
        
        # --- Draw Node Info (Aruco ID) ---
        node_id, node_corners = node_info
        node_text = f"Node Detection: ID {node_id}" if node_id is not None else "Node Detection: ---"
        cv2.putText(diag_frame, node_text, (nx1 + 10, ny1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        if node_corners is not None:
            cv2.polylines(diag_frame, [np.int32(node_corners)], True, (0, 255, 255), 2)

        # --- Draw Line Following Info ---
        if line_info:
            pos_err = line_info.get("positional_error", 0)
            ang_err = line_info.get("angle_error", 0.0)
            line_text = f"PosErr:{pos_err}, AngErr:{ang_err:.1f}"
            cv2.putText(diag_frame, line_text, (lx1 + 10, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # --- Draw Intersection Analysis Probes ---
        if intersection_analysis and node_id is not None and node_corners is not None:
            # Get the same parameters used in analyze_intersection
            PROBE_DISTANCE_FROM_CENTER = 35 
            probe_w, probe_h = 25, 25
            p_dist = PROBE_DISTANCE_FROM_CENTER
            
            # Get the marker's center coordinates from node_info
            marker_center_x = int(np.mean(node_corners[0][:, 0]))
            marker_center_y = int(np.mean(node_corners[0][:, 1]))
            
            probe_color = (255, 200, 0) # Light Blue

            # Top probe
            y1_up, y2_up = marker_center_y - p_dist - probe_h, marker_center_y - p_dist
            x1_up, x2_up = marker_center_x - probe_w // 2, marker_center_x + probe_w // 2
            cv2.rectangle(diag_frame, (x1_up, y1_up), (x2_up, y2_up), probe_color, 2)

            # Left probe
            y1_left, y2_left = marker_center_y - probe_h // 2, marker_center_y + probe_h // 2
            x1_left, x2_left = marker_center_x - p_dist - probe_w, marker_center_x - p_dist
            cv2.rectangle(diag_frame, (x1_left, y1_left), (x2_left, y2_left), probe_color, 2)

            # Right probe
            y1_right, y2_right = marker_center_y - probe_h // 2, marker_center_y + probe_h // 2
            x1_right, x2_right = marker_center_x + p_dist, marker_center_x + p_dist + probe_w
            cv2.rectangle(diag_frame, (x1_right, y1_right), (x2_right, y2_right), probe_color, 2)

            # <<< ADDED: Define and draw the coordinates for the DOWN probe rectangle >>>
            y1_down, y2_down = marker_center_y + p_dist, marker_center_y + p_dist + probe_h
            x1_down, x2_down = marker_center_x - probe_w // 2, marker_center_x + probe_w // 2
            cv2.rectangle(diag_frame, (x1_down, y1_down), (x2_down, y2_down), probe_color, 2)
            
            # Display the analysis result text
            analysis_text = f"Type: {intersection_analysis.get('type', '...')}, Exits: {intersection_analysis.get('exits', [])}"
            cv2.putText(diag_frame, analysis_text, (nx1 + 10, ny1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        return diag_frame
        
    def load_calibration_data(self, filename: str = "calibration_data.yml"):
        try:
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
                self.camera_matrix = np.array(data['camera_matrix'])
                self.dist_coeffs = np.array(data['dist_coeffs'])
                print("✅ Camera calibration data loaded successfully.")
        except FileNotFoundError:
            print("⚠️ Calibration file not found. Running with uncalibrated images.")
        except Exception as e:
            print(f"❌ Error loading calibration data: {e}")

    def process_and_correct_frame(self, frame: np.ndarray) -> np.ndarray:
        processed_image = frame
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = processed_image.shape[:2]
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
            processed_image = cv2.undistort(processed_image, self.camera_matrix, self.dist_coeffs, None, new_mtx)
            x, y, w, h = roi
            processed_image = processed_image[y:y+h, x:x+w]
        processed_image = self.perspective_transformer.warp(processed_image)
        return processed_image


    def _get_rois(self, frame_shape: tuple):
        """
        Defines the regions of interest for node and line detection.
        The node ROI is made larger to better detect intersection geometry.
        """
        h, w = frame_shape[:2]
        
        # NEW: Extend the bottom of the ROI from 50% down to 70% of the frame height.
        node_roi_coords = (0, int(h * 0.10), w, int(h * 0.70))
        
        # Adjust the line ROI to start slightly lower to reduce overlap.
        line_roi_coords = (0, int(h * 0.70), w, int(h * 0.95))
        
        return node_roi_coords, line_roi_coords

    def detect_node_in_roi(self, frame: np.ndarray) -> tuple:
        """
        Detects an ArUco marker in the node ROI and translates its corner
        coordinates to be relative to the full frame.
        """
        node_roi_coords, _ = self._get_rois(frame.shape)
        (x1, y1, x2, y2) = node_roi_coords
        node_roi_frame = frame[y1:y2, x1:x2]
        
        corners, ids, _ = self.aruco_detector.detectMarkers(node_roi_frame)
        
        if ids is not None:
            # FIX: Translate corner coordinates from ROI-relative to frame-relative.
            # Add the ROI's top-left offset (x1, y1) to each corner point.
            corners[0][0][:, 0] += x1
            corners[0][0][:, 1] += y1
            
            return (ids[0][0], corners[0])
            
        return (None, None)
    

# In vision/image_processor.py

    def analyze_intersection(self, bird_eye_frame: np.ndarray) -> dict | None:
        """
        Analyzes an intersection by probing in FOUR directions and provides a detailed 
        classification of the node type, including specific turns.
        """
        # --- 1. Find the ArUco marker and its center ---
        corners, ids, _ = self.aruco_detector.detectMarkers(bird_eye_frame)
        if ids is None:
            return None

        marker_corners = corners[0][0]
        marker_center_x = int(np.mean(marker_corners[:, 0]))
        marker_center_y = int(np.mean(marker_corners[:, 1]))

        # --- 2. Create a clean binary image for analysis ---
        processed_frame = bird_eye_frame.copy()
        cv2.fillPoly(processed_frame, [np.int32(marker_corners)], (255, 255, 255))
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)

        # --- 3. Define and execute probes for all FOUR directions ---
        PROBE_DISTANCE_FROM_CENTER = 35
        probe_w, probe_h = 25, 25
        p_dist = PROBE_DISTANCE_FROM_CENTER

        # Define ROIs for all four directions
        y1_up, y2_up = marker_center_y - p_dist - probe_h, marker_center_y - p_dist
        x1_up, x2_up = marker_center_x - probe_w // 2, marker_center_x + probe_w // 2
        roi_up = binary[y1_up:y2_up, x1_up:x2_up]

        y1_left, y2_left = marker_center_y - probe_h // 2, marker_center_y + probe_h // 2
        x1_left, x2_left = marker_center_x - p_dist - probe_w, marker_center_x - p_dist
        roi_left = binary[y1_left:y2_left, x1_left:x2_left]

        y1_right, y2_right = marker_center_y - probe_h // 2, marker_center_y + probe_h // 2
        x1_right, x2_right = marker_center_x + p_dist, marker_center_x + p_dist + probe_w
        roi_right = binary[y1_right:y2_right, x1_right:x2_right]

        y1_down = marker_center_y + p_dist
        y2_down = marker_center_y + p_dist + probe_h
        x1_down, x2_down = marker_center_x - probe_w // 2, marker_center_x + probe_w // 2
        roi_down = binary[y1_down:y2_down, x1_down:x2_down]
        
        # --- 4. Collect all detected exits ---
        INTENSITY_THRESHOLD = 40
        exits = []
        if roi_up.size > 0 and np.mean(roi_up) > INTENSITY_THRESHOLD: exits.append("up")
        if roi_left.size > 0 and np.mean(roi_left) > INTENSITY_THRESHOLD: exits.append("left")
        if roi_right.size > 0 and np.mean(roi_right) > INTENSITY_THRESHOLD: exits.append("right")
        if roi_down.size > 0 and np.mean(roi_down) > INTENSITY_THRESHOLD: exits.append("down")

        # --- 5. Classify the intersection with detailed turn logic ---
        num_exits = len(exits)
        node_type = "unknown"
        exits_set = set(exits)

        if num_exits == 4:
            node_type = '4-way'
        
        elif num_exits == 3:
            node_type = '3-way'
            
        elif num_exits == 2:
            # Check for straight paths first
            if ('up' in exits_set and 'down' in exits_set) or \
            ('left' in exits_set and 'right' in exits_set):
                node_type = 'straight'
            # --- Re-introducing detailed turn classification ---
            # These names describe the shape of the corner. The robot's controller
            # will calculate the actual turn direction (left/right) based on its approach.
            elif 'up' in exits_set and 'left' in exits_set:
                node_type = 'Turn-left'
            elif 'up' in exits_set and 'right' in exits_set:
                node_type = 'Turn-right'
            elif 'down' in exits_set and 'left' in exits_set:
                node_type = 'Turn (Down-Left)'
            elif 'down' in exits_set and 'right' in exits_set:
                node_type = 'Turn (Down-Right)'
            else:
                node_type = 'corner' # Fallback for any other 2-exit combo
                
        elif num_exits == 1:
            node_type = 'dead-end'
            
        elif num_exits == 0:
            node_type = 'isolated'

        return {"type": node_type, "exits": exits}

    # def analyze_intersection(self, bird_eye_frame: np.ndarray) -> dict | None:
    #     """
    #     Analyzes an intersection robustly by using the ArUco marker's center
    #     as the reference point for classifying exits.
    #     """
    #     corners, ids, _ = self.aruco_detector.detectMarkers(bird_eye_frame)
    #     if ids is None:
    #         return None

    #     marker_corners = corners[0][0]
    #     marker_center_x = int(np.mean(marker_corners[:, 0]))
    #     marker_center_y = int(np.mean(marker_corners[:, 1]))

    #     processed_frame = bird_eye_frame.copy()
    #     cv2.fillPoly(processed_frame, [np.int32(marker_corners)], (255, 255, 255))

    #     gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
    #     # --- CHANGE 1: Make Canny edge detection slightly more sensitive ---
    #     # Lowering the first threshold allows weaker edges to be considered.
    #     edges = cv2.Canny(blurred, 30, 150, apertureSize=3)
        
    #     # --- CHANGE 2: Make Hough line detection less strict ---
    #     # Lowering the threshold and minLineLength helps detect shorter/weaker lines.
    #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=25)
        
    #     if lines is None:
    #         return {"type": "dead-end", "exits": []}

    #     # The rest of the function remains the same
    #     directions = {'left': False, 'right': False, 'up': False}
    #     angle_threshold = 30

    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         line_mid_x = (x1 + x2) / 2
            
    #         if math.sqrt((line_mid_x - marker_center_x)**2) > 150:
    #             continue

    #         angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    #         angle = (angle + 360) % 360

    #         if (angle < angle_threshold) or (angle > 360 - angle_threshold) or (abs(angle - 180) < angle_threshold):
    #             if line_mid_x < marker_center_x: directions['left'] = True
    #             else: directions['right'] = True
    #         elif (abs(angle - 90) < angle_threshold) or (abs(angle - 270) < angle_threshold):
    #             if min(y1, y2) < marker_center_y: directions['up'] = True
        
    #     exits = [direction for direction, found in directions.items() if found]
    #     num_exits = len(exits)
        
    #     node_type = "unknown"
    #     if num_exits == 3: node_type = '4-way'
    #     elif num_exits == 2: node_type = '3-way'
    #     elif num_exits == 1:
    #         if 'up' in exits: node_type = 'straight'
    #         elif 'left' in exits: node_type = 'Turn-left'
    #         elif 'right' in exits: node_type = 'Turn-right'
    #     elif num_exits == 0:
    #         node_type = 'dead-end'

    #     return {"type": node_type, "exits": exits}

    # def calculate_line_command(self, frame: np.ndarray) -> dict | None:
    #     """
    #     Calculates the robot's driving command based ONLY on the angle of the detected line.
    #     """
    #     _, line_roi_coords = self._get_rois(frame.shape)
    #     (x1, y1, x2, y2) = line_roi_coords
    #     line_roi_frame = frame[y1:y2, x1:x2]
        
    #     gray_roi = cv2.cvtColor(line_roi_frame, cv2.COLOR_BGR2GRAY)
    #     binary_inv = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        
    #     contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     if not contours:
    #         return None
            
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     if cv2.contourArea(largest_contour) < 100:
    #         return None

    #     # --- Angle-Based Logic ---
    #     contour_points = largest_contour.reshape(-1, 2)
        
    #     _, box_y, _, box_h = cv2.boundingRect(contour_points)
    #     y_mid = box_y + box_h / 2
        
    #     upper_points = contour_points[contour_points[:, 1] < y_mid]
    #     lower_points = contour_points[contour_points[:, 1] >= y_mid]

    #     if len(upper_points) == 0 or len(lower_points) == 0:
    #         return None

    #     p_top = tuple(np.mean(upper_points, axis=0).astype(int))
    #     p_bottom = tuple(np.mean(lower_points, axis=0).astype(int))

    #     delta_x = float(p_bottom[0] - p_top[0])
    #     delta_y = float(p_bottom[1] - p_top[1])
    #     angle = math.degrees(math.atan2(delta_x, delta_y))

    #     # Determine command based on the angle
    #     command = "UNKNOWN"
    #     ANGLE_THRESHOLD = 15 # Degrees of tolerance for "FORWARD"
        
    #     if abs(angle) < ANGLE_THRESHOLD:
    #         command = "FORWARD"
    #     elif angle < 0: # Negative angle means tilted left
    #         command = "LEFT"
    #     else: # Positive angle means tilted right
    #         command = "RIGHT"
            
    #     # Calculate the overall centroid for drawing purposes
    #     M = cv2.moments(largest_contour)
    #     if M["m00"] == 0:
    #         return None
    #     cx = int(M["m10"] / M["m00"])
    #     cy = int(M["m01"] / M["m00"])
            
    #     return {
    #         "command": command, 
    #         "contour": largest_contour, 
    #         "centroid": (cx, cy),
    #         "p_top": p_top,
    #         "p_bottom": p_bottom,
    #         "angle": angle
    #     }

# Place this in vision/image_processor.py

    def calculate_line_command(self, frame: np.ndarray, focus_area: str = "FULL") -> dict | None:
        """
        Detects the line in the frame and calculates the positional and angular error
        of the robot relative to it. Includes a selective focus feature to aid in
        precise turning maneuvers.
        """
        # --- Stage 1: Image Preparation & Contour Detection ---

        # Isolate the Region of Interest (ROI) for line following.
        _, line_roi_coords = self._get_rois(frame.shape)
        (x1, y1, x2, y2) = line_roi_coords
        line_roi_frame = frame[y1:y2, x1:x2]

        # Convert to grayscale, apply a blur to reduce noise, then create a binary image.
        gray_roi = cv2.cvtColor(line_roi_frame, cv2.COLOR_BGR2GRAY)
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        # Using adaptiveThreshold for robust performance in varying lighting conditions.
        binary_inv = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)

        # --- Stage 2: Apply Selective Focus if Required ---
        if focus_area in ["LEFT", "RIGHT"]:
            h, w = binary_inv.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            if focus_area == "LEFT":
                # Create a white mask for the left half of the ROI.
                cv2.rectangle(mask, (0, 0), (w // 2, h), 255, -1)
            elif focus_area == "RIGHT":
                # Create a white mask for the right half of the ROI.
                cv2.rectangle(mask, (w // 2, 0), (w, h), 255, -1)

            # Apply the mask to erase contours outside the focus area.
            binary_inv = cv2.bitwise_and(binary_inv, binary_inv, mask=mask)

        # Find all contours (outlines of the line) in the binary image.
        contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Select the largest contour, which is assumed to be the main line.
        largest_contour = max(contours, key=cv2.contourArea)
        # Ignore any contours that are too small to be the line.
        if cv2.contourArea(largest_contour) < 50:
            return None

        # --- Stage 3: Calculate Positional and Angular Errors ---

        # Calculate the contour's moments to find its centroid for positional error.
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        # cx: horizontal position of the line's center in the ROI.
        cx = int(M["m10"] / M["m00"])
        # roi_center_x: horizontal position of the ROI's center.
        roi_center_x = line_roi_frame.shape[1] // 2
        # positional_error: the horizontal distance of the line's center from the robot's view center.
        positional_error = cx - roi_center_x

        # Calculate the line's angle for angular error.
        angle_error = 0.0
        try:
            contour_points = largest_contour.reshape(-1, 2)
            # Find an average point in the upper half and lower half of the contour.
            _, box_y, _, box_h = cv2.boundingRect(contour_points)
            y_mid = box_y + box_h / 2
            upper_points = contour_points[contour_points[:, 1] < y_mid]
            lower_points = contour_points[contour_points[:, 1] >= y_mid]

            if len(upper_points) > 0 and len(lower_points) > 0:
                p_top = tuple(np.mean(upper_points, axis=0).astype(int))
                p_bottom = tuple(np.mean(lower_points, axis=0).astype(int))

                # Calculate the angle of the line connecting these two points.
                delta_x = float(p_bottom[0] - p_top[0])
                # Add a small epsilon to the denominator to prevent division by zero.
                delta_y = float(p_bottom[1] - p_top[1]) + 1e-6
                angle_error = math.degrees(math.atan2(delta_x, delta_y))
        except (ZeroDivisionError, ValueError):
            # In case of any calculation error, default the angular error to zero.
            angle_error = 0.0

        # Return a dictionary containing both calculated errors.
        return {
            "positional_error": positional_error,
            "angle_error": angle_error
        }