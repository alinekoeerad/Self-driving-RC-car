# config.py

# ⚙️ Network Settings
ESP32_STREAM_URL = "http://192.168.137.104:81/stream"  # TODO: IP address of your ESP32 camera
CSHARP_HOST = '127.0.0.1'
CSHARP_DATA_PORT = 9998
CSHARP_COMMAND_PORT = 9999
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000

# 📷 Camera Calibration Settings
CHESSBOARD_SIZE = (7, 7)  # Number of inner corners (width, height)

# 👁️ Image Processing Settings
LINE_THRESHOLD_MIN = 100 
LINE_THRESHOLD_MAX = 255

# 🤖 Robot Settings
ROBOT_DEFAULT_SPEED = 150 # Default robot movement speed
ROBOT_TURN_SPEED = 120    # Robot speed when turning

# 🖥️ Simulation and Debug Settings
SHOW_DEBUG_FRAMES = True  # Should the OpenCV debug windows be displayed?