import cv2
import numpy as np
import math

def generate_aruco_sheet(start_id, end_id, output_filename="aruco_sheet.png"):
    """
    Generates a printable A4 sheet with ArUco markers of a specific range.

    Each marker is 2x2 cm, designed for printing on A4 paper at 300 DPI.
    """
    # --- Constants for A4 Paper and Printing ---
    DPI = 300
    INCH_TO_CM = 2.54

    # A4 paper dimensions in cm
    A4_WIDTH_CM = 21.0
    A4_HEIGHT_CM = 29.7
    
    # Convert A4 dimensions to pixels
    A4_WIDTH_PX = math.floor(A4_WIDTH_CM / INCH_TO_CM * DPI)
    A4_HEIGHT_PX = math.floor(A4_HEIGHT_CM / INCH_TO_CM * DPI)

    # --- Marker and Layout Settings ---
    MARKER_SIZE_CM = 3.0
    MARGIN_CM = 1.0  # Margin from the edge of the paper
    GAP_CM = 0.5     # Gap between markers

    # Convert marker and layout dimensions to pixels
    MARKER_SIZE_PX = math.floor(MARKER_SIZE_CM / INCH_TO_CM * DPI)
    MARGIN_PX = math.floor(MARGIN_CM / INCH_TO_CM * DPI)
    GAP_PX = math.floor(GAP_CM / INCH_TO_CM * DPI)

    # --- ArUco Dictionary ---
    # Using a common dictionary. You can change this if needed.
    # DICT_4X4_50 contains markers with IDs from 0 to 49.
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    except AttributeError:
        print("Error: Your OpenCV version might be old. Trying legacy attribute.")
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


    # Create a blank white A4 sheet
    # Note: OpenCV uses (height, width) order for dimensions
    a4_sheet = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX, 3), dtype=np.uint8) * 255
    print(f"Created a blank A4 sheet of size: {A4_WIDTH_PX}x{A4_HEIGHT_PX} pixels.")
    print(f"Each marker will be {MARKER_SIZE_PX}x{MARKER_SIZE_PX} pixels.")

    # --- Place Markers on the Sheet ---
    current_x = MARGIN_PX
    current_y = MARGIN_PX
    
    total_markers = end_id - start_id + 1
    print(f"Generating {total_markers} markers from ID {start_id} to {end_id}...")

    for marker_id in range(start_id, end_id + 1):
        if marker_id > 49:
            print(f"Warning: Marker ID {marker_id} is outside the range of DICT_4X4_50 (0-49). Skipping.")
            continue

        # Check if the next marker fits horizontally
        if current_x + MARKER_SIZE_PX > A4_WIDTH_PX - MARGIN_PX:
            # Move to the next row
            current_x = MARGIN_PX
            current_y += MARKER_SIZE_PX + GAP_PX

        # Check if the next row fits vertically
        if current_y + MARKER_SIZE_PX > A4_HEIGHT_PX - MARGIN_PX:
            print("Error: Not enough space on the page for all markers. Stopping.")
            break
            
        # Generate the ArUco marker image (it's grayscale)
        marker_image = np.zeros((MARKER_SIZE_PX, MARKER_SIZE_PX), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE_PX, marker_image, 1)

        # Convert grayscale marker to a 3-channel BGR image to place it on the sheet
        marker_bgr = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
        
        # Define the region of interest (ROI) on the A4 sheet
        roi = a4_sheet[current_y : current_y + MARKER_SIZE_PX, current_x : current_x + MARKER_SIZE_PX]
        
        # Place the marker in the ROI
        roi[:, :] = marker_bgr
        
        # Update x-coordinate for the next marker
        current_x += MARKER_SIZE_PX + GAP_PX

    # Save the final image
    cv2.imwrite(output_filename, a4_sheet)
    print(f"\n✅ Successfully created the marker sheet!")
    print(f"File saved as: {output_filename}")

if __name__ == "__main__":
    try:
        start = int(input("Enter the starting ArUco ID (e.g., 0): "))
        end = int(input("Enter the ending ArUco ID (e.g., 49): "))
        
        if start < 0 or end > 49 or start > end:
            print("Invalid range. IDs must be between 0 and 49 for DICT_4X4_50.")
        else:
            generate_aruco_sheet(start, end)
            
    except ValueError:
        print("Invalid input. Please enter numbers only.")