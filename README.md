Obstacle Detection and Directional Alert System
This project utilizes a YOLOv5 model for object detection and obstacle avoidance guidance. Using a connected camera, the system detects obstacles in real-time and provides directional alerts on how to avoid them.

Table of Contents
Installation
Usage
How It Works
License
Installation
To set up and run this project, follow these steps:

Clone the Repository
Clone this repository to your local machine:

git clone <repository_url>
cd <repository_name>
Set Up Python Environment
Create and activate a virtual environment:

python3 -m venv yolov5_env
source yolov5_env/bin/activate  # For Linux/macOS
yolov5_env\Scripts\activate     # For Windows
Install Required Libraries
Install the dependencies listed in requirements.txt:

pip install -r requirements.txt
If you don’t have a requirements.txt, install the following libraries manually:

pip install torch
pip install opencv-python
Download YOLOv5 Model Weights
Download the YOLOv5 model weights and place them in the project directory. You can download the weights from Ultralytics’ YOLOv5 GitHub repository.

Usage
Run the following command to start the obstacle detection system:

python obstacle_detection.py
Press q to stop the program and close the camera window.

How It Works
Real-Time Camera Feed: The system accesses the connected camera and processes frames in real time.
Object Detection: Each frame is processed by the YOLOv5 model to detect obstacles.
Directional Alerts: Based on obstacle location, an alert is printed to suggest movement directions to avoid obstacles.
License
This project integrates several open-source libraries and models. Below are details on their licenses:

YOLOv5 Model by Ultralytics
![image](https://github.com/user-attachments/assets/fc89092d-c094-499e-ad47-471a83af74be)

License: GPL-3.0 License
Description: YOLOv5 is licensed under the GPL-3.0, which allows for free use, modification, and distribution under the same license.
PyTorch

License: BSD-3 License
Description: PyTorch, developed by Meta, is licensed under the BSD-3, allowing for free use and modification.
OpenCV

License: Apache 2.0 License
Description: OpenCV, an open-source computer vision library, is licensed under the Apache 2.0 License, enabling flexible usage and distribution.
Note: This project’s code is intended for educational purposes and should comply with the terms of each library’s respective license.

