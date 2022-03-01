import cv2
import mediapipe as mp
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('fall_check_4.avi')
out = cv2.VideoWriter('fall_check_4_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (320, 240))

# For Video input:
prevTime = 0
nose_coords = [0, 0]
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = results.pose_landmarks.landmark
            y = landmarks[mp_pose.PoseLandmark.NOSE.value].y * image.shape[1]
            print(f'diff: {y - nose_coords[-2]}')
            if (y - nose_coords[-2]) > 10:
                cv2.putText(image, 'Falling', (10, 220), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            nose_coords.append(y)
        except:
            pass

        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        out.write(image)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        cv2.imshow('BlazePose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
out.release()
cap.release()
cv2.destroyAllWindows()
