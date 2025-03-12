import cv2
import mediapipe as mp
import numpy as np
from cvzone.PoseModule import PoseDetector
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
detector = PoseDetector()

# Initialize the list to store data
data = []

# Define column names for the DataFrame
columns = ['Reps', 'HandAngle', 'ShoulderAngle']

# Create DataFrame from the list and column names
df = pd.DataFrame(data, columns=columns)

# Setup CSV file for writing
csv_filename = 'output_data.csv'

# Define function to calculate angle between two points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    angle2 = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Read video from file or device
video_capture = cv2.VideoCapture('9.mp4')

# Read two consecutive frames from the video
ret, frame1 = video_capture.read()
ret, frame2 = video_capture.read()
frame1 = detector.findPose(frame1)
lmList, bbox = detector.findPosition(frame1)

# Curl counter variables
counter = 0
stage = None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while video_capture.isOpened():
        ret, frame1 = video_capture.read()

        # Convert frames to grayscale

        image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detection
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angles
            angle = calculate_angle(shoulder, elbow, wrist)
            angle2 = calculate_angle(hip, knee, ankle)
            shoulder_angle = calculate_angle(elbow, shoulder, hip)

            # Append data to the list
            data.append([counter, angle, shoulder_angle])

            # Update DataFrame
            df = pd.DataFrame(data, columns=columns)

            # Export DataFrame to CSV
            df.to_csv(csv_filename, index=False)

            # Visualize angles
            cv2.putText(frame1, f"Hand Angle: {angle}",
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # cv2.putText(frame1, f"Leg Angle: {angle2}",
            #             (10, 140),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame1, f"Shoulder Angle: {shoulder_angle}",
                        tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic
            if angle > 150 and angle2 > 170:
                stage = "down"
            if angle < 50 and stage == 'down' and angle2 > 170:
                stage = "up"
                counter += 1
                print(counter)

        except:
            pass

        # Render detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # Render curl counter
        cv2.rectangle(image, (0, 0), (170, 280), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'HAND ANGLES', (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"{int(angle)}",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'SHOULDER ANGLE', (15, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"{int(shoulder_angle)}",
                    (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame with angle information
        cv2.imshow('Video with Angle', image)

        # Update frames
        frame1 = frame2
        ret, frame2 = video_capture.read()

        # Break if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# Release resources and close the window
video_capture.release()
cv2.destroyAllWindows()


