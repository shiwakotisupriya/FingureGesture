import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
gesture_data = []
gesture_labels = []
def collect_data(label, gesture_name, num_samples=200):
    collected_samples = 0
    while collected_samples < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                gesture_data.append(features)
                gesture_labels.append(label)
                collected_samples += 1

        cv2.putText(frame, f'Collecting data for gesture: {gesture_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Samples collected: {collected_samples}/{num_samples}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
gesture_names = [
    "call me", "loser", "high-five",
    "good job","dislike","love", 
    "point", "ROCK"
]

for i, gesture in enumerate(gesture_names):
    print(f"Collecting data for gesture '{gesture}'")
    collect_data(label=i, gesture_name=gesture)

gesture_data = np.array(gesture_data)
gesture_labels = np.array(gesture_labels)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(gesture_data, gesture_labels)


gesture_dict = {i: name for i, name in enumerate(gesture_names)}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            features = np.array(features).reshape(1, -1)
            
            gesture = knn.predict(features)
            gesture_text = gesture_dict[gesture[0]]
            cv2.putText(frame, f'Gesture: {gesture_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
