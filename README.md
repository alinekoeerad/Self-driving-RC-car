Autonomous Exploration Robot & Vision-Based Topological Mapping
This project is the implementation of an autonomous robotic system designed to explore unknown environments, generate a topological map, and navigate intelligently to user-specified points. The system consists of a mobile robot (RC Car), a central processing server (written in Python), and a controller software (written in C#).

🌟 Key Features
Fully Autonomous Exploration: The robot automatically traverses and maps out the entire accessible environment using an enhanced version of the Frontier-based Exploration algorithm.

Topological Mapping: Instead of grid-based maps, the system constructs a topological graph of the environment where each node represents a key location (an ArUco marker).

Precise Visual Localization: To overcome the cumulative error of odometry, the system uses ArUco visual markers as a "Virtual GPS." These markers provide the robot with precise, absolute position and orientation data.

Intelligent Navigation: Once the map is complete, the system uses the A* search algorithm to calculate the shortest path between the robot and a user-selected destination and guides the robot along that path.

Centralized Control & Monitoring: A desktop application (WPF) allows the user to monitor the robot's camera feed, the live map generation, and system logs in real-time, as well as manage the entire process.

📐 System Architecture
The system follows a Client-Server architecture with the following components:

Central Server (Python):

Processing Core: Responsible for running the main algorithms, including exploration, localization, ArUco marker detection, and path planning (A*).

Image Processing: Uses the OpenCV library to process video frames from the robot's camera and extract spatial information.

Communication Management: Handles communication with the robot (via Serial or Network) and the C# controller (via Sockets).

Controller Application (C# WPF):

User Interface (UI): Provides a graphical dashboard to display the topological map, live video stream, robot status, and system messages.

Server Communication: Sends user commands (e.g., start exploration, set destination) to the server and displays incoming data (e.g., map updates).

Robot (Hardware):

A mobile platform (RC car) equipped with a camera and a microcontroller (like an ESP32 or Raspberry Pi) to execute movement commands received from the server.

💻 Tech Stack
Server-Side:

Language: Python 3

Libraries:

OpenCV-Python: For all computer vision tasks and ArUco marker detection.

NumPy: For numerical and matrix operations.

pyserial: For serial communication with the robot's hardware.

socket: For network communication with the C# controller.

Controller-Side:

Language: C#

Framework: .NET 8 with WPF for the graphical user interface.

Libraries:

Newtonsoft.Json: For serializing and deserializing JSON data exchanged with the server.

🛠️ Setup and Installation
Follow these steps to get the project running:

Prerequisites
Python 3.8 or newer

.NET 8 SDK

Visual Studio 2022

1. Setting up the Python Server
Bash

# 1. Clone the repository
git clone https://github.com/alinekoeerad/Self-driving-RC-car.git
cd Self-driving-RC-car/Server

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt
2. Setting up the C# Controller
Open the Controller/Controller.sln solution file with Visual Studio 2022.

Wait for Visual Studio to restore all the required NuGet packages.

Build the project (from the menu, select Build > Build Solution).

🚀 How to Use
Run the Server: First, start the Python server.

Bash

# From the Server directory
python main_server.py
The server will now wait for connections from the controller and the robot.

Run the Controller: Launch the controller application from Visual Studio (by pressing F5) or by running the executable file located in the bin/Debug folder.

Start the Operation:

Connect to the server from the controller application.

Begin the Autonomous Exploration process. The robot will start moving through the environment, and the map will be drawn live in the application.

Once the exploration is complete, you can select a destination by clicking on one of the nodes on the map.

The system will calculate the shortest path, and the robot will autonomously navigate to the selected destination.

👤 Author
Ali Nekoee Rad

GitHub

LinkedIn

This project was submitted as a Bachelor of Science thesis in Computer Engineering Of Zanjan University, under the supervision of Sajad Haghzad Klidbary, Ph.D., in September 2025.
