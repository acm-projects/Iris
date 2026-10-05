import cv2
import numpy as np
import mediapipe as mp

from keypoints import frame_to_keypoints, new_buffer, add_frame, NUM_SAMPLES

np.set_printoptions(precision=3, suppress=True)   # shorter, cleaner numbers

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

buffer = new_buffer()

cap = cv2.VideoCapture(1)   # 0 = default camera; try 0 if you get a black screen

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        kp = frame_to_keypoints(results)   # the layer's output: normalized (55, 2) array
        add_frame(buffer, kp)

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(frame, f"buffer: {len(buffer)}/{NUM_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Camera test', frame)

        key = cv2.waitKey(1) & 0xFF      # call waitKey only once per loop
        if key == ord('k') and kp is not None:   # press k to print the keypoints
            print("normalized keypoints (55, 2):")
            print("rows 0-12 body, 13-33 left hand, 34-54 right hand")
            print(kp)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()