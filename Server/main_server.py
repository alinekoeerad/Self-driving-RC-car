# main_server.py

import cv2
import time
from config import *
from communication.csharp_communicator import CSharpCommunicator
from communication.robot_communicator import RobotCommunicator
from vision.image_processor import ImageProcessor
from vision.camera_calibrator import CameraCalibrator
from vision.perspective_transformer import PerspectiveTransformer
from logic.graph_manager import GraphManager
from logic.robot_controller import RobotController
from vision.stream_reader import VideoStreamReader 

def main():
    """The main entry point for the server."""
    
    # 1. Instantiate all modules
    print("Initializing modules...")
    graph_manager = GraphManager()
    image_processor = ImageProcessor()
    csharp_comm = CSharpCommunicator(CSHARP_HOST, CSHARP_DATA_PORT, CSHARP_COMMAND_PORT)
    robot_comm = RobotCommunicator(FLASK_HOST, FLASK_PORT)
    calibrator = CameraCalibrator()
    transformer = PerspectiveTransformer()

    # 2. Instantiate the main controller, injecting all dependencies
    controller = RobotController(
        graph_manager=graph_manager,
        image_processor=image_processor,
        robot_comm=robot_comm,
        csharp_comm=csharp_comm,
        calibrator=calibrator,
        transformer=transformer
    )

    # 3. Start communication threads
    robot_comm.start()
    csharp_comm.start() # CSharp communicator is now started as well

    # 4. Set up video capture using the threaded reader
    print(f"🎬 Starting threaded video stream from {ESP32_STREAM_URL}...")
    stream_reader = VideoStreamReader(ESP32_STREAM_URL).start()
    
    time.sleep(2.0) # Give the stream a moment to start
    
    print("✅ Server is fully running. Entering main loop...")
    
    # 5. Main application loop
    try:
        while True:
            frame = stream_reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # The controller handles ALL logic, including sending updates to C#
            controller.process_frame(frame)
            
            # Show debug window if enabled
            if SHOW_DEBUG_FRAMES:
                debug_frame = controller.get_processed_frame()
                if debug_frame is not None:
                    cv2.imshow("Debug View", debug_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed. Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\nCTRL+C pressed. Shutting down.")
    finally:
        # 6. Clean up resources gracefully
        print("--- Cleaning up resources ---")
        stream_reader.stop()
        csharp_comm.stop()
        
        # --- DELETE OR COMMENT OUT THIS LINE ---
        # robot_comm.stop()
        # ---------------------------------------

        cv2.destroyAllWindows()
        print("Server shut down gracefully.")
    
if __name__ == "__main__":
    main()