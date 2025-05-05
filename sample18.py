import streamlit as st
import numpy as np
import cv2
import joblib
import datetime
import pygame
import time
import threading
from collections import deque, defaultdict
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

#defect counter
defect_counter = defaultdict(int)

stop_video = False
frame_buffer = None
#alert sound and delay
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.mp3")
last_alert_time = 0  
alert_cooldown = 2
prediction_delay = 0.5
grace_period_passed = False
last_detected_defect = None
last_detected_time = time.time()
prediction_interval = 1
last_prediction_time = 0

#logging
log_file_path = "detection_log.txt"
def log_to_file(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

#alert sound with cooldown
#threading for smooth camera feed
def play_alert():
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time > alert_cooldown:
        last_alert_time = current_time
    
    threading.Thread(target=alert_sound.play, daemon=True).start()

#normalizing image captured
def normalize_image(image):
    if image.dtype in [np.float32, np.float64]:
        image = (image - image.min()) / (image.max() - image.min())
    return (image * 255).astype(np.uint8)

#resizing image after capture
def extract_features(image):
    image = cv2.resize(image, (128, 128))
    raw_pixels = image.flatten()
    return raw_pixels

#auto-brightness image
def normalize_brightness(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    return img_eq

#load trainingdata3 (svm)
model = joblib.load("svm_fabric_defect.pkl")
scaler = joblib.load("scaler.pkl")
prediction_buffer = deque(maxlen=5)
classes = ["defect", "defect", "defect", "no defect"]

#prediction confidence (to prevent inconsistency)
def weighted_majority_voting(predictions):
    """Gives higher weight to recent predictions to stabilize output."""
    weights = [1, 2, 3, 4, 5]
    weighted_predictions = {}

    for i, pred in enumerate(predictions):
        if pred not in weighted_predictions:
            weighted_predictions[pred] = 0
        weighted_predictions[pred] += weights[i]

    return max(weighted_predictions, key=weighted_predictions.get)

#defect counter
def summary():
    "Returns a formatted string of total defect counts for UI display."
    return "\n".join([f"{key.capitalize()}: {value}" for key, value in defect_counter.items()])

#reset defect counter
def reset_defect_counter():
    "Resets the defect counter and last detected defect when detection stops."
    global defect_counter, last_detected_defect, last_detected_time
    defect_counter = defaultdict(int)
    last_detected_defect = None
    last_detected_time = time.time()


#predicting using feature extraction tools
def predict_defect(image):
    global prediction_buffer, last_detected_defect, defect_counter, last_detected_time, grace_period_passed, last_prediction_time, current_time

    current_time = time.time()
    if current_time - last_prediction_time < prediction_interval:
        return f"Predicted defect: {last_detected_defect or 'no defect'}", summary()
    else:
        last_prediction_time = current_time
    if not grace_period_passed:
        if current_time - last_detected_time < 3:
            return f"Waiting to stabilize... {3 - int(current_time - last_detected_time)}s", summary()
        else:
            grace_period_passed = True

    #feature extraction
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    lbp_features = lbp_transform(gray_resized).flatten()
    gabor_features = gabor_transform(gray_resized).flatten()
    raw_pixels = gray_resized.flatten()

    features = np.hstack([raw_pixels, lbp_features, gabor_features])
    features = scaler.transform([features])

    decision_scores = model.decision_function(features)
    confidence = np.max(np.array(decision_scores))

    if confidence < 0.67: 
        return "Predicted defect: no defect", summary()

    prediction = model.predict(features)[0]
    prediction_buffer.append(prediction)

    stabilized_prediction = weighted_majority_voting(list(prediction_buffer))
    defect = classes[stabilized_prediction]

    #prediction_delay
    if defect != "no defect":
        if grace_period_passed:    
            if defect != last_detected_defect or (current_time - last_detected_time > 5):
                defect_counter[defect] += 1
                log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"[{log_time}] Counted defect: {defect} (Total: {defect_counter[defect]})"
                log_to_file(log_message)
                play_alert()
                last_detected_time = current_time
            last_detected_defect = defect
    
    if defect != last_detected_defect and defect != "no defect":
        play_alert()
        last_detected_time = current_time   
    last_detected_defect = defect

    print(f"Prediction: {prediction}")
    return f"Predicted defect: {defect}", summary()

#LBP
def lbp_transform(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image 
    radius = 0.5
    n_points = 2 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    return normalize_image(lbp)

#Gabor Filter
def gabor_transform(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image 
    frequency = 0.69
    gabor_response, _ = gabor(gray, frequency=frequency)
    return normalize_image(gabor_response)

#Camera list
def get_available_cameras():
    index = 3
    available_cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        available_cameras.append(str(index))
        cap.release()
        index += 1
    return available_cameras if available_cameras else ["0"]

#opencv
def process_video(camera_index=0):
    global stop_video
    global frame_buffer
    stop_video = False
    cap = cv2.VideoCapture(int(camera_index))
    reset_defect_counter()
    
    while cap.isOpened():
        if stop_video:
            break

        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        prediction, defect_summary = predict_defect(frame)
        yield gray_frame, prediction, defect_summary
    cap.release()

threading.Thread(target=process_video, daemon=True).start()


#stop detection
def stop_camera():
    global stop_video
    stop_video = True
    reset_defect_counter() 

    cv2.destroyAllWindows()

# Streamlit interface
st.title("Cotton Fabric Defect Detection")

st.markdown("We are from Group 1 of 12 STEM 8B of Far Eastern University High School Inc. This is the prototype code made and modified by the coders of this group with reference from x-Ck-x, Prahmodh-Raj1, and tirthajyoti in GitHub. This contains live processing using Local Binary Pattern (LBP), Gabor filter, and Support Vector Machine (SVM) in real-time from the webcam.")

camera_dropdown = st.selectbox("Select Camera", options=get_available_cameras(), index=0)

start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

# Streamlit placeholders
frame_placeholder = st.empty()
prediction_placeholder = st.empty()
defect_summary_placeholder = st.empty()

if start_button:
    reset_defect_counter()
    for frame, prediction, defect_summary in process_video(camera_dropdown):
        frame_placeholder.image(frame, channels="GRAY", use_container_width=True)
        prediction_placeholder.text(prediction)
        defect_summary_placeholder.text(defect_summary)

if stop_button:
    stop_camera()
