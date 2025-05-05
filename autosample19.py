import streamlit as st
import numpy as np
import cv2
import joblib
import pygame
import time
import threading
import serial
from collections import deque, defaultdict
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

arduino = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)

def stop_motor():
    arduino.write(b"STOP\n")

def start_motor():
    arduino.write(b"START\n")

#defect counter
defect_counter = defaultdict(int)

stop_video = False
frame_buffer = None
#alert sound and delay
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.mp3")
last_alert_time = 0  
alert_cooldown = 4
prediction_delay = 4
last_detected_defect = None
last_detected_time = time.time()

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

#load trainingdata3 (svm sgdclassifier)
model = joblib.load("svm_fabric_defect.pkl")
scaler = joblib.load("scaler.pkl")
prediction_buffer = deque(maxlen=5)
classes = ["hole", "tear", "stain", "no defect"]

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

def detect_fabric(image):
    # Example fabric detection: check if the image has sufficient brightness
    # You can replace this logic with a more sophisticated method to detect fabric.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    threshold_brightness = 100  # You can adjust this threshold based on your image
    return avg_brightness > threshold_brightness

#predicting using feature extraction tools
def predict_defect(image):
    global prediction_buffer, last_detected_defect, defect_counter, last_detected_time

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

    if confidence < 0.6: 
        return "Predicted defect: no defect", summary()

    prediction = model.predict(features)[0]
    prediction_buffer.append(prediction)

    #use weighted voting for stabilization
    stabilized_prediction = weighted_majority_voting(list(prediction_buffer))
    defect = classes[stabilized_prediction]
    current_time = time.time()

    if defect in ["hole", "tear", "stain"]:
        if defect != last_detected_defect or (current_time - last_detected_time > prediction_delay):
            defect_counter[defect] += 1
            last_detected_time = current_time
    
    if defect != last_detected_defect:
        play_alert()

    last_detected_defect = defect
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
    index = 2
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

    fabric_detected = False
    delay_time = 2
    
    while cap.isOpened():
        if stop_video:
            break

        ret, frame = cap.read()
        if not ret:
            break
        if detect_fabric(frame):
            if not fabric_detected:  # Check if fabric was not previously detected
                stop_motor()  # Stop the motor when fabric is detected
                fabric_detected = True
                print("Fabric detected. Starting delay before prediction...")
                
                # Introduce a delay before starting defect prediction
                time.sleep(delay_time)
                print("Delay over. Starting defect prediction.")
                
            prediction, defect_summary = predict_defect(frame)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            yield gray_frame, prediction, defect_summary
        else:
            if fabric_detected:  # If fabric was detected previously, resume motor
                start_motor()
                fabric_detected = False  # Reset fabric detection flag

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

st.markdown("We are from Group 1 of 12 STEM 8B of Far Eastern University High School Inc. This is the prototype code made and modified by the coders of this group with reference from x-Ck-x, Prahmodh-Raj1, and tirthajyoti in GitHub. This contains live processing using Fast Fourier Transform (FFT), Local Binary Pattern (LBP), Gabor filter, and Support Vector Machine (SVM) in real-time from the webcam.")

camera_dropdown = st.selectbox("Select Camera", options=get_available_cameras(), index=0)

start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

# Streamlit Output placeholders
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

