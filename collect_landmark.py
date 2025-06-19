import cv2
import csv
import os
import mediapipe as mp

# 📁 Save CSV file to specific folder
SAVE_DIR = r"C:\Users\namit\OneDrive\Desktop\RSET\Other_projects\ISL_3"
csv_filename = os.path.join(SAVE_DIR, "landmarks.csv")

# 📝 Create the CSV file with header if it doesn't exist
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        header = ['label'] + [f'{coord}_{i}' for i in range(21) for coord in ('x', 'y', 'z')]
        csv_writer.writerow(header)

# 🖐️ MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 📷 OpenCV setup
cap = cv2.VideoCapture(0)
print("🧠 Press a key (e.g. 'a', '1', ...) to label a frame.\n🚪 Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 🔁 Flip and convert image to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    # ✍️ Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 🖼️ Display the image
    cv2.imshow('Hand Capture', image)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key != -1 and results.multi_hand_landmarks:
        try:
            char_label = chr(key).upper()
            if not (char_label.isalpha() or char_label.isdigit()):
                continue

            # 🧠 Extract landmark coordinates
            hand_landmarks = results.multi_hand_landmarks[0]
            data_row = [char_label]
            for lm in hand_landmarks.landmark:
                data_row.extend([lm.x, lm.y, lm.z])

            # 💾 Save to CSV
            with open(csv_filename, mode='a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(data_row)

            print(f"✅ Saved sample for label: {char_label}")
        except ValueError:
            continue

# 🧹 Clean up
cap.release()
cv2.destroyAllWindows()
