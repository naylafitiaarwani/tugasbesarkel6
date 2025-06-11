import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
import uuid
import os

# Load model dan label
model = tf.keras.models.load_model('asl_huruf_model.h5')
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Buffer smoothing prediksi
predictions_buffer = deque(maxlen=10)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        x_list = []
        y_list = []

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

        x_min = max(min(x_list) - 30, 0)
        y_min = max(min(y_list) - 30, 0)
        x_max = min(max(x_list) + 30, w)
        y_max = min(max(y_list) + 30, h)

        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size == 0:
            continue

        # Preprocessing
        hand_img = cv2.resize(hand_img, (64, 64))
        hand_img = hand_img.astype("float32") / 255.0
        hand_img = np.expand_dims(hand_img, axis=0)

        # Predict
        prediction = model.predict(hand_img, verbose=0)
        pred_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        label = labels[pred_index]

        predictions_buffer.append(label)
        smoothed_label = Counter(predictions_buffer).most_common(1)[0][0]

        # Penentuan warna
        color = (0, 255, 0) if smoothed_label in alphabet_labels else (255, 0, 0)
        label_text = f"{smoothed_label} ({confidence:.1f}%)"

        # Tampilkan hasil
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, label_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Deteksi Bahasa Isyarat", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Simpan gambar untuk debugging
        filename = f"gesture_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join("captured", filename)
        os.makedirs("captured", exist_ok=True)
        cv2.imwrite(path, hand_img)
        print(f"Gambar disimpan di {path}")

cap.release()
cv2.destroyAllWindows()
