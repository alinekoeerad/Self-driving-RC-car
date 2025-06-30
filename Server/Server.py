import cv2
import numpy as np
import socket

# Video stream path
video_path = "http://192.168.137.48:8080/video"
# Frame dimensions
width = 720
height = 560

def detect_curves(frame, distance_range, sock, esp_ip, esp_port):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Create a mask with zeros (black image)
    mask = np.zeros_like(edges)
    height, width = frame.shape[:2]
    bottom_y = height
    top_y = int(height * (1 - distance_range))
    # Define the region of interest (ROI) polygon vertices
    vertices = np.array([[(20, top_y - 150), (width - 20, top_y - 150), (width - 20, bottom_y - 200), (20, bottom_y - 200)]], dtype=np.int32)
    # Fill the ROI on the mask
    cv2.fillPoly(mask, vertices, 255)
    # Apply the mask to the edges image
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Draw the ROI rectangle on the frame
    cv2.rectangle(frame, (20, top_y - 150), (width - 20, bottom_y - 200), (255, 0, 0), 2)

    # Calculate the midpoint of the ROI
    midpoint_x = (20 + (width - 20)) // 2
    midpoint_y = (top_y - 150 + (bottom_y - 200)) // 2
    
    # Perform Hough Line Transform to detect lines in the masked edges
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    angles = []
    red_lines_points = []

    if lines is not None:
        # Merge similar lines
        lines = merge_lines(lines)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw detected lines on the frame
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 8)

            # Calculate the angle of each line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            if angle < 0:
                angle = 180 + angle
            angles.append(angle)

            # Calculate the deviation angle and second point for drawing the red line
            deviation = angle
            offset_x = int(50 * np.cos(np.radians(deviation)))
            second_point_x = midpoint_x + offset_x
            second_point_y = midpoint_y  # Keep y the same
            
            red_lines_points.append(((midpoint_x, midpoint_y), (second_point_x, second_point_y)))
            
    if angles:
        # Calculate the average angle of the detected lines
        avg_angle = sum(angles) / len(angles)
        print(f"Average Angle: {avg_angle:.2f} degrees")
        # Send the average angle to the ESP device via UDP
        message = f"{avg_angle:.2f}\n"
        sock.sendto(message.encode(), (esp_ip, esp_port))

    if red_lines_points:
        # Calculate the average midpoint and second point for drawing the red line
        avg_midpoint_x = sum(p[0][0] for p in red_lines_points) // len(red_lines_points)
        avg_midpoint_y = sum(p[0][1] for p in red_lines_points) // len(red_lines_points)
        avg_second_point_x = sum(p[1][0] for p in red_lines_points) // len(red_lines_points)
        avg_second_point_y = avg_midpoint_y  # Ensure the y-coordinate is the same for horizontal line
        # Draw the average red line on the frame
        cv2.line(frame, (avg_second_point_x, avg_second_point_y), (avg_midpoint_x, avg_midpoint_y), (0, 0, 255), 8)
    
    # Draw a circle at the midpoint
    cv2.circle(frame, (midpoint_x, midpoint_y), 5, (255, 255, 0), -1)
    
    return frame

def merge_lines(lines, threshold=50):
    if lines is None:
        return None
    
    merged_lines = []
    for line in lines:
        if len(merged_lines) == 0:
            merged_lines.append(line)
        else:
            merged = False
            for merged_line in merged_lines:
                # Check if lines are close enough to be merged
                if np.linalg.norm(np.array(line[0][:2]) - np.array(merged_line[0][:2])) < threshold and \
                   np.linalg.norm(np.array(line[0][2:]) - np.array(merged_line[0][2:])) < threshold:
                    # Merge lines by averaging their coordinates
                    merged_line[0] = [
                        (line[0][0] + merged_line[0][0]) // 2,
                        (line[0][1] + merged_line[0][1]) // 2,
                        (line[0][2] + merged_line[0][2]) // 2,
                        (line[0][3] + merged_line[0][3]) // 2
                    ]
                    merged = True
                    break
            if not merged:
                merged_lines.append(line)
    return np.array(merged_lines)

def main():
    # Open the video stream
    cap = cv2.VideoCapture(video_path)
    
    # Define the distance range and ESP connection details
    distance_range = 0.3
    esp_ip = "192.168.137.23"
    esp_port = 12345
    
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to the desired dimensions
        frame_resized = cv2.resize(frame, (width, height))

        # Rotate the frame 90 degrees clockwise
        image = cv2.rotate(frame_resized, cv2.ROTATE_90_CLOCKWISE)        

        # Detect curves and draw results on the frame
        result_frame = detect_curves(image, distance_range, sock, esp_ip, esp_port)
        
        # Display the result frame
        cv2.imshow('Lane Detection', result_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



