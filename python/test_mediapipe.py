import cv2
import mediapipe as mp

from keypoints import frame_to_keypoints, new_buffer, add_frame, buffer_to_model_input, NUM_SAMPLES

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

        kp = frame_to_keypoints(results)
        add_frame(buffer, kp)

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(frame, f"buffer: {len(buffer)}/{NUM_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Camera test', frame)

        key = cv2.waitKey(1) & 0xFF      # call waitKey only once per loop
        if key == ord('p'):              # press p to print the model input
            x = buffer_to_model_input(buffer)
            if x is None:
                print("buffer not full yet")
            else:
                print("model input shape:", x.shape)   # you want (1, 55, 100)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()