import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
MODEL_PATH = "isl_landmark_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load label classes
label_classes = np.load("label_classes.npy")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit UI
st.title("🤟 Real-time Sign Language Recognizer")
st.markdown("Uses hand landmarks + a trained model to recognize static signs.")
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Could not access webcam")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = "No hand detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            prediction = model.predict(np.array([landmarks]))[0]
            predicted_label = label_classes[np.argmax(prediction)]
            prediction_text = f"✋ Predicted Sign: **{predicted_label}**"

            # Draw label on screen
            cv2.putText(frame, f"{predicted_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    st.markdown(prediction_text)

cap.release()
