# Real-Time Security Detection & Tracking System

A concurrent computer vision and data-logging pipeline engineered to process live video streams and track targets with minimal operational latency.

## Key Features & Architecture

- **Multi-Threaded Execution:** Separates heavy computer vision frame processing from the main execution thread. This guarantees continuous, stutter-free frame capture while data tasks run in the background.
- **Low-Latency Logging:** Implements an optimized SQLite database layer to handle real-time tracking event logs concurrently without blocking the live video feed.
- **Object Tracking:** Integrates OpenCV and YOLOv8 pipelines to handle frame-by-frame target identification and tracking.

## Tech Stack

- **Language:** Python
- **Libraries:** OpenCV, Ultralytics (YOLOv8)
- **Database:** SQLite
- **Concurrency:** Python Threading Library
