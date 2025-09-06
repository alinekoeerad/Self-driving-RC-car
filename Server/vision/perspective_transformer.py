# vision/perspective_transformer.py

import numpy as np
import cv2
import yaml

class PerspectiveTransformer:
    """Handles calculating and applying perspective transformations."""

    def __init__(self, output_size: tuple = (400, 400)):
        self.output_size = output_size
        self.transform_matrix = None
        self.points = []

    def load_perspective_data(self, filename: str = "perspective_data.yml") -> bool:
        """Loads perspective transform data from a file."""
        try:
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
                self.transform_matrix = np.array(data['transform_matrix'])
            print("✅ Perspective transform data loaded successfully.")
            return True
        except FileNotFoundError:
            print("⚠️ Perspective data file not found. Perspective transform is inactive.")
            return False
        except Exception as e:
            print(f"❌ Error loading perspective data: {e}")
            return False

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Applies the perspective transform to a frame IF the matrix exists."""
        # If the transform matrix has not been set, return the original frame.
        if self.transform_matrix is None:
            return frame
        
        return cv2.warpPerspective(frame, self.transform_matrix, self.output_size)

    # --- Other methods remain the same ---
    def reset_points(self):
        self.points = []

    def add_point(self, point: tuple):
        if len(self.points) < 4:
            self.points.append(point)

    def set_transform_from_points(self) -> bool:
        if len(self.points) != 4:
            return False
        width, height = self.output_size
        dst_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        src_points_np = np.float32(self.points)
        self.transform_matrix = cv2.getPerspectiveTransform(src_points_np, dst_points)
        return True

    def save_perspective_data(self, filename: str = "perspective_data.yml"):
        if self.transform_matrix is not None:
            with open(filename, 'w') as f:
                yaml.dump({'transform_matrix': self.transform_matrix.tolist()}, f)
            print(f"💾 Perspective data saved to {filename}")