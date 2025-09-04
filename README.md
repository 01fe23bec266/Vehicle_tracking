# Vehicle_tracking
📌 Project Overview

This project implements a real-time vehicle tracking system using OpenCV without relying on deep learning models.
The system detects, tracks, and counts vehicles from video streams by using background subtraction, contour analysis, and centroid tracking.

It is lightweight, efficient, and works in real-time — making it suitable for traffic monitoring, intelligent transport systems, toll collection, and surveillance.

🎯 Objectives

Detect moving vehicles in a video using background subtraction and contour detection.

Assign unique IDs to each vehicle and track their movement using centroid tracking.

Count vehicles moving in different directions across a predefined line.

Display the results in real-time and save the output to a video file.

📚 Literature Survey

Vehicle Detection and Tracking for Traffic Surveillance Applications – A Review (IJCSE, 2018)

Vehicle Detection and Tracking Techniques: A Concise Review (arXiv, 2014)

Vehicle Detection and Tracking Using Thermal Cameras in Adverse Visibility Conditions (Sensors, 2022)

Vehicle Detection and Vehicle Tracking Applications on Traffic Video Surveillance Systems: A Systematic Literature Review (IJCESEN, 2024)

Object Tracking: An Experimental and Comprehensive Study on Vehicle Object in Video (IJIGSP, 2022)

📂 Dataset & Preprocessing

Dataset Source: CCTV footage, dashcam videos, or open datasets.

Preprocessing Steps:

Frame extraction

Grayscale conversion

Background subtraction (MOG2)

Noise removal (morphological operations)

⚙️ Methodology & Flow

Feature Extraction

Bounding box area, aspect ratio, centroid

Contours and shape info

Frame differencing, background subtraction (MOG2), optical flow, HOG

Vehicle Detection & Tracking

Foreground masks using background subtraction

Bounding boxes around contours

Centroid tracking to assign unique IDs

Vehicle counting across a virtual line

Pipeline

Input Video → Frame Extraction → Preprocessing (Grayscale + Background Subtraction)  
→ Contour Detection → Feature Extraction → Centroid Tracking → Vehicle Counting → Output Video

🛠️ Tools & Libraries

Languages: Python

Libraries: OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib

Development Tools: Jupyter Notebook, Google Colab, GitHub

🚧 Challenges

Occlusion when vehicles overlap

Lighting and weather variations (shadows, rain, fog)

Scale and perspective differences

Real-time performance constraints

✅ Output

Processed video with:

Bounding boxes

Unique IDs

Vehicle counts (upward / downward direction)

Saved output video for later use

👨‍💻 Team

Sai Chetan (01FE23BEC)

Sagar N (01FE23BEC266)

Chetan S (Team 20)

Center for Intelligent Mobility (CIM)

📌 Future Work

Improve occlusion handling

Explore deep learning (YOLO, DeepSORT) for advanced tracking

Deploy in real-time traffic monitoring systems
