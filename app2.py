import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model/keypoint_classifier_5.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label names
with open('model/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

def preprocess_landmarks(landmarks):
    base_x, base_y = landmarks[0]
    return [(x - base_x, y - base_y) for x, y in landmarks]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract keypoints (x, y)
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

            if len(landmarks) == 21:
                normalized = preprocess_landmarks(landmarks)
                input_data = np.array([coord for point in normalized for coord in point], dtype=np.float32)
                input_data = np.expand_dims(input_data, axis=0)

                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Get output tensor
                result = interpreter.get_tensor(output_details[0]['index'])[0]
                predicted_id = np.argmax(result)
                predicted_label = labels[predicted_id]

                # Show prediction
                cv2.putText(frame, predicted_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('Hand Sign Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
