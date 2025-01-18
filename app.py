import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# File to save the trained model
MODEL_FILE = "gesture_model.pkl"

# Hand gesture classes
GESTURES = ["Hi", "Thank You", "Good Morning", "Yes", "No", "Unknown"]

# Helper function: Collect data
def collect_data(label, num_samples=200):
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    data = []
    st.write(f"Collecting data for '{label}' gesture. Show the gesture in front of the camera.")

    while len(data) < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]
                data.append(landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        st.image(frame, channels="BGR", caption="Show the gesture in front of the camera")
        if st.button("Stop"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    return data

# Helper function: Train model
def train_model():
    all_data = []
    all_labels = []
    for idx, gesture in enumerate(GESTURES[:-1]):  # Exclude "Unknown"
        st.write(f"Collecting data for: {gesture}")
        data = collect_data(gesture)
        all_data.extend(data)
        all_labels.extend([idx] * len(data))

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(all_data), np.array(all_labels), test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    st.success(f"Model trained successfully with accuracy: {accuracy:.2f}")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# Helper function: Predict gesture
def predict_gesture(model, landmarks):
    if landmarks is None or model is None:
        return "Unknown"
    prediction = model.predict([landmarks])[0]
    return GESTURES[prediction]

# Gesture Recognition
def recognize_gesture():
    if not os.path.exists(MODEL_FILE):
        st.error("Model not found! Train the model first.")
        return

    model = pickle.load(open(MODEL_FILE, "rb"))
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    st.write("Start showing gestures to the camera...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        landmarks = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        gesture = predict_gesture(model, landmarks)
        st.image(frame, channels="BGR", caption=f"Recognized Gesture: {gesture}")
        if st.button("Stop"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

# Streamlit App
def main():
    st.title("Sign Language Recognition App")
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Train Model", "Gesture Recognition"])

    if app_mode == "Home":
        st.write("Welcome to the Sign Language Recognition App!")
        st.write("Use the sidebar to train the model or start recognizing gestures.")
    elif app_mode == "Train Model":
        st.header("Train the Model")
        train_model()
    elif app_mode == "Gesture Recognition":
        st.header("Gesture Recognition")
        recognize_gesture()

if __name__ == "__main__":
    main()
