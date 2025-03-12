# Pull-Up Form Detection and Correction using OpenCV

## Overview

This project uses OpenCV and MediaPipe to analyze and correct pull-up exercise forms. By processing video input, the system detects key body landmarks, calculates angles, and identifies form issues to provide real-time feedback to the user. The results, including repetition count and angles, are logged to a CSV file for further analysis.

---

## Features

- **Pose Detection**: Detects key landmarks such as shoulders, elbows, and wrists using MediaPipe Pose.
- **Angle Calculation**: Computes angles between joints to assess pull-up form.
- **Real-Time Feedback**: Displays visual and textual feedback on the screen during exercise.
- **Repetition Counting**: Tracks the number of correctly performed pull-ups.
- **Data Logging**: Stores angle and repetition data in a CSV file for analysis.

---

## Components

### Software:

- **Python Libraries**:
  - OpenCV: For video processing and visualization.
  - MediaPipe: For pose detection.
  - NumPy: For numerical calculations.
  - pandas: For data handling and logging.

### Hardware:

- A camera (e.g., webcam or smartphone camera) to capture video for analysis.
- A computer with Python installed.

---

## Installation and Setup

### Step 1: Install Python and Dependencies

1. Install Python (version 3.7 or later) from the [official website](https://www.python.org/).
2. Install required libraries using pip:
   ```bash
   pip install opencv-python mediapipe numpy pandas
   ```

### Step 2: Prepare the Video Input

- Use a pre-recorded video of pull-ups or connect to a live camera feed for real-time analysis.

### Step 3: Run the Code

1. Save the provided code as `pull_up_detection.py`.
2. Place the video file (`9.mp4`) in the same directory as the script, or modify the code to use live video.
3. Execute the script:
   ```bash
   python pull_up_detection.py
   ```

---

## Code Details

- **Key Functions**:
  - `calculate_angle(a, b, c)`: Calculates the angle between three points.
  - MediaPipe's `Pose` module is used for detecting body landmarks.
- **Outputs**:
  - Video stream with annotated landmarks and angles.
  - CSV file (`output_data.csv`) containing repetitions and angle data.

---

## How It Works

1. **Pose Detection**:
   - MediaPipe Pose identifies key landmarks such as shoulders, elbows, wrists, hips, knees, and ankles.

2. **Angle Calculation**:
   - Calculates the angles between joints (e.g., elbow and shoulder) to determine form correctness.

3. **Repetition Counting**:
   - Detects upward and downward movements based on angle thresholds to count repetitions.

4. **Feedback**:
   - Displays angle values, stage (up/down), and repetition count on the video.

5. **Data Logging**:
   - Logs data (repetition count, hand angle, shoulder angle) to `output_data.csv` for analysis.

---

## Example CSV Output

| Reps | HandAngle | ShoulderAngle |
|------|-----------|---------------|
| 1    | 45.0      | 30.0          |
| 2    | 50.0      | 35.0          |

---

## Troubleshooting

- **No landmarks detected**:
  - Ensure the camera is properly positioned and captures the user fully.
  - Increase lighting to improve video quality.

- **Incorrect angle values**:
  - Verify the video resolution and quality.
  - Check for consistent poses and avoid occlusions.

- **Script crashes**:
  - Ensure all required libraries are installed.
  - Check the video file path and ensure the file exists.

---

## Future Enhancements

- Add support for additional exercises (e.g., squats, push-ups).
- Integrate a graphical user interface (GUI) for easier interaction.
- Provide detailed feedback for form improvement.
- Enable real-time analysis via live camera feed.

---

## License

This project is licensed under the MIT License. Feel free to use and modify the code as needed.

