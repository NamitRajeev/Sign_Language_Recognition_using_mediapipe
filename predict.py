import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

# Paths
BASE_PATH = "C:/Users/namit/OneDrive/Desktop/RSET/Other_projects/ISL_3"
MODEL_PATH = os.path.join(BASE_PATH, "isl_landmark_model.h5")
LABELS_PATH = os.path.join(BASE_PATH, "label_classes.npy")

# Load trained model and labels
model = tf.keras.models.load_model(MODEL_PATH)
label_classes = np.load(LABELS_PATH, allow_pickle=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip for natural selfie view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and flatten coordinates
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            if len(landmark_list) == 63:
                # Predict
                prediction = model.predict(np.array([landmark_list]), verbose=0)
                pred_class = np.argmax(prediction)
                pred_label = label_classes[pred_class]

                # Show result
                cv2.putText(frame, f'Prediction: {pred_label}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("ISL Real-Time Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
