import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

# Load the trained model
model = tf.keras.models.load_model("isl_cnn_model_A_to_M.h5")
class_names = sorted(os.listdir("ISL_custom_data"))  # Make sure this path matches your dataset folder

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
IMAGE_SIZE = 128

print("📷 Webcam started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            x1 = int(x_min * w) - 20
            y1 = int(y_min * h) - 20
            x2 = int(x_max * w) + 20
            y2 = int(y_max * h) + 20

            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)

            # Crop and preprocess the hand region
            hand_img = frame[y1:y2, x1:x2]
            try:
                hand_resized = cv2.resize(hand_img, (IMAGE_SIZE, IMAGE_SIZE))
                hand_normalized = hand_resized / 255.0
                hand_input = np.expand_dims(hand_normalized, axis=0)

                # Predict
                pred = model.predict(hand_input, verbose=0)
                label = class_names[np.argmax(pred)]
                confidence = np.max(pred)

                # Draw results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            except Exception as e:
                print(f"⚠️ Prediction error: {e}")

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display frame
    cv2.imshow("ISL Real-Time Prediction", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
