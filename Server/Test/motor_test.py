# File: motor_test.py
# Description: A script to test the robot's motors by sending specific speed values.

import sys
import os
import time

# --- START OF FIX ---
# این بخش به پایتون کمک می‌کند تا پوشه‌های بالاتر را برای پیدا کردن ماژول‌ها جستجو کند
# This adds the parent directory (Server) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --- END OF FIX ---

from communication.robot_communicator import RobotCommunicator

def run_motor_test():
    """
    Initializes the robot communicator and runs a sequence of movements
    by sending direct speed values to each motor.
    """
    robot_comm = RobotCommunicator()
    robot_comm.start()
    print("Robot Communicator started for motor speed test.")
    time.sleep(2)

    # --- دنباله حرکات با سرعت‌های مشخص برای هر موتور ---
    # سرعت‌ها می‌توانند بین -255 (عقب) تا 255 (جلو) باشند
    # command: همیشه باید "DRIVE" باشد
    # payload: یک دیکشنری شامل سرعت موتور چپ و راست
    # duration: مدت زمان اجرای حرکت به ثانیه
    movements = [
        {"name": "Move Forward (Medium Speed)", "command": "DRIVE", "payload": {"left": 200, "right": 200}, "duration": 3},
        {"name": "Move Backward (Medium Speed)", "command": "DRIVE", "payload": {"left": -200, "right": -200}, "duration": 3},
        {"name": "Pivot Turn Right", "command": "DRIVE", "payload": {"left": 200, "right": -200}, "duration": 3},
        {"name": "Pivot Turn Left", "command": "DRIVE", "payload": {"left": -200, "right": 200}, "duration": 3},
        {"name": "Gentle Curve Right", "command": "DRIVE", "payload": {"left": 220, "right": 150}, "duration": 3},
        {"name": "Gentle Curve Left", "command": "DRIVE", "payload": {"left": 150, "right": 220}, "duration": 3},
    ]

    try:
        print("\n--- Starting Motor Speed Test Sequence ---")
        for move in movements:
            print(f"Executing: {move['name']} for {move['duration']} seconds...")
            print(f"  -> Sending payload: {move['payload']}")
            
            # ارسال دستور "DRIVE" همراه با سرعت‌های مشخص
            robot_comm.set_drive_command(move["command"], move["payload"])
            time.sleep(move["duration"])

        # توقف کامل ربات پس از اتمام تست
        print("Stopping the robot.")
        robot_comm.set_drive_command("STOP", None)
        print("--- Motor Speed Test Complete ---")

    except KeyboardInterrupt:
        print("\nTest interrupted by user. Stopping the robot.")
        robot_comm.set_drive_command("STOP", None)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Stopping the robot as a precaution.")
        robot_comm.set_drive_command("STOP", None)

if __name__ == "__main__":
    run_motor_test()