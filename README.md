# Real-Time Obstacle Detection System with Angle Tracking

## Overview
This project implements a real-time obstacle detection system using computer vision techniques and the FAST (Features from Accelerated Segment Test) algorithm. The system includes angle tracking capabilities and a graphical user interface built with PyQt5, making it suitable for surveillance and monitoring applications.

## Features
- Real-time obstacle detection using various feature detection algorithms (FAST, SIFT, SURF, etc.)
- Advanced angle change detection with Kalman filtering
- Region of Interest (ROI) based processing
- Real-time video stream processing from RTSP sources
- Comprehensive event logging system
- User-friendly GUI with:
  - Live video feed display
  - Algorithm selection dropdown
  - Start/Stop controls
  - Rotation angle display
- Automatic warning system for significant angle changes

## Requirements
```
python >= 3.6
opencv-python
numpy
PyQt5
filterpy
```

## Installation
Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration
Update the RTSP URL in the `FastAlgorithmApp` class to match your camera settings:
```python
rtsp_url = "rtsp://admin:your_password@your_ip:554"
```

## Usage
1. Run the main script:
```bash
python main.py
```

2. The GUI will appear with the following controls:
   - Feature detector selection dropdown
   - Start/Stop buttons
   - Live video feed display
   - Rotation angle indicator

3. The system will automatically:
   - Detect obstacles in the defined ROI
   - Track angle changes
   - Generate warnings for significant movements
   - Log events to CSV file

## Logging
The system automatically logs events to `engel_tespiti_log.csv` with the following information:
- Date and time
- Event type
- Number of detected keypoints
- Detection and warning times
- Delay duration
- Angle changes
- Event descriptions

## Technical Details
- **Feature Detection**: Implements multiple feature detection algorithms including FAST, SIFT, SURF, BRISK, ORB, etc.
- **Angle Detection**: Uses advanced angle detection with Kalman filtering for stable measurements
- **Multi-threading**: Implements threaded processing for improved performance
- **ROI Processing**: Focuses detection on specific regions for better accuracy

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Author
Furkan Türkoğlu

## Acknowledgments
- OpenCV community
- PyQt5 framework
- FilterPy library contributors
