# communication/robot_communicator.py
from flask import Flask, jsonify
import threading
import logging

class RobotCommunicator:
    """
    Runs a Flask server to provide the persistent desired state to the robot.
    """
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
        # --- NEW: The persistent state dictionary ---
        self.robot_state = {
            "command": "STOP",
            "payload": None,
            "led": "OFF" # Default LED state
        }
        self.lock = threading.Lock()

        @self.app.route('/get_command', methods=['GET'])
        def get_command_route():
            # Always return the current, complete state
            with self.lock:
                return jsonify(self.robot_state)

    def set_drive_command(self, command, payload):
        """Thread-safely sets the movement part of the state."""
        with self.lock:
            self.robot_state["command"] = command
            self.robot_state["payload"] = payload

    def set_led_state(self, led_state: str):
        """Thread-safely sets the LED part of the state."""
        with self.lock:
            self.robot_state["led"] = led_state
            print(f"💡 LED state updated to: {led_state}")
    
    def start(self):
        """Starts the Flask server in a separate daemon thread."""
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.thread = threading.Thread(target=self.app.run, kwargs={'host': self.host, 'port': self.port}, daemon=True)
        self.thread.start()
        print(f"🤖 RobotCommunicator (Flask) started on http://{self.host}:{self.port}")