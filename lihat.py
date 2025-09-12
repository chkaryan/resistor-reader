import cv2
import numpy as np
import joblib
from gtts import gTTS
import pygame
import os
import RPi.GPIO as GPIO
import time
from sklearn.neighbors import NearestNeighbors

# ===== GPIO Setup =====
BUTTON_PIN = 17  
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  

# ===== Load trained model AND scaler =====
model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")  

# ===== Configuration Parameters =====
CONFIDENCE_THRESHOLD = 0.3  

# ===== Enhanced Label to spoken form mapping =====
spoken_text = {
    "10k": "sepuluh ribu",
    "470": "empat ratus tujuh puluh",
    "680": "enam ratus delapan puluh",
    "unknown": "komponen tidak dikenali atau bukan resistor dalam database"
}

# ===== Enhanced Classification Function with Confidence =====
def classify_with_confidence(features, model, scaler, threshold=CONFIDENCE_THRESHOLD):
    """
    Classify resistor with confidence threshold using scaled features
    Returns: (prediction, confidence_distance, is_confident)
    """
    try:
        # ===== SCALE THE FEATURES FIRST =====
        features_scaled = scaler.transform(features)
        
        # Get distances to nearest neighbors (on scaled data)
        distances, indices = model.kneighbors(features_scaled, n_neighbors=3)
        min_distance = distances[0][0]  
        
        # Calculate additional confidence metrics
        avg_distance = np.mean(distances[0])  
        distance_variance = np.var(distances[0])  
        
        # Determine if prediction is confident
        is_confident = min_distance < threshold
        
        if is_confident:
            prediction = model.predict(features_scaled)[0]
            print(f"Confident prediction: {prediction} (distance: {min_distance:.4f})")
        else:
            prediction = "unknown"
            print(f"Low confidence: {prediction} (distance: {min_distance:.4f}, threshold: {threshold})")
        
        return prediction, min_distance, is_confident
        
    except Exception as e:
        print(f"Error in classification: {e}")
        return "unknown", 1.0, False

# ===== Function to speak resistor value =====
def speak_resistor_value(value_ohm):
    """Enhanced TTS function with error handling"""
    try:
        text = f"Resistor {value_ohm} Ohm" if value_ohm != "komponen bukan resistor atau resistor tidak ter-register" else value_ohm
        filename = "resistor_output.mp3"
        
        tts = gTTS(text=text, lang='id')
        tts.save(filename)

        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        os.remove(filename)
        
    except Exception as e:
        print(f"Error in TTS: {e}")

# ===== Enhanced logging function =====
def log_detection_result(predicted_value, confidence_distance, is_confident, h_mean, s_mean, v_mean):
    """Log detection results for analysis"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}, {predicted_value}, {confidence_distance:.4f}, {is_confident}, {h_mean:.2f}, {s_mean:.2f}, {v_mean:.2f}\n"
    
    try:
        with open("detection_log.csv", "a") as log_file:
            # Write header if file is empty
            if os.path.getsize("detection_log.csv") == 0:
                log_file.write("Timestamp,Prediction,Distance,Confident,H_mean,S_mean,V_mean\n")
            log_file.write(log_entry)
    except Exception as e:
        print(f"Logging error: {e}")

# ===== Camera setup =====
cap = cv2.VideoCapture(0)
current_value = None
current_confidence = 0
last_detection_time = 0

# Set camera resolution for better display
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Sistema deteksi resistor dengan scaled features aktif")
print(f"Threshold confidence: {CONFIDENCE_THRESHOLD}")
print("Tekan button untuk analisis, 'q' untuk keluar\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not detected.")
            break

        # Create a copy for display
        display_frame = frame.copy()
        
        # Resize and extract HSV mean values
        resized = cv2.resize(frame, (200, 200))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)

        # Enhanced prediction with confidence (NOW WITH SCALING)
        features = np.array([[h_mean, s_mean, v_mean]])
        predicted_value, confidence_distance, is_confident = classify_with_confidence(features, model, scaler)
        current_value = spoken_text.get(predicted_value, "error")
        current_confidence = confidence_distance

        # Add text overlay to display frame
        status_text = "CONFIDENT" if is_confident else "UNCERTAIN"
        color = (0, 255, 0) if is_confident else (0, 0, 255)  # Green if confident, Red if uncertain
        
        # Display prediction and confidence on frame
        cv2.putText(display_frame, f"Prediction: {predicted_value}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, f"Status: {status_text}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, f"Distance: {confidence_distance:.4f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, f"HSV: H={h_mean:.1f} S={s_mean:.1f} V={v_mean:.1f}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add instructions
        cv2.putText(display_frame, "Press button for audio or 'q' to quit", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Resistor Detection Camera', display_frame)

        # Display real-time status (optional - for debugging)
        current_time = time.time()
        if current_time - last_detection_time > 2:  # Update every 2 seconds
            status = "CONFIDENT" if is_confident else "UNCERTAIN"
            print(f"Status: {status} | Prediction: {predicted_value} | Distance: {confidence_distance:.4f}")
            last_detection_time = current_time

        # If button pressed (LOW because of pull-up), speak and log
        if GPIO.input(BUTTON_PIN) == GPIO.LOW and current_value:
            print(f"\n=== BUTTON PRESSED ===")
            print(f"Original features: H={h_mean:.2f}, S={s_mean:.2f}, V={v_mean:.2f}")
            print(f"Prediction: {predicted_value}")
            print(f"Confidence Distance: {confidence_distance:.4f}")
            print(f"Threshold: {CONFIDENCE_THRESHOLD}")
            print(f"Confident: {is_confident}")
            print(f"Speaking: {current_value}")
            print("========================\n")
            
            # Log the detection result
            log_detection_result(predicted_value, confidence_distance, is_confident, h_mean, s_mean, v_mean)
            
            # Speak the result
            speak_resistor_value(current_value)
            time.sleep(1.0)  # Enhanced debounce for logging
        
        # Check for 'q' key press to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\nShutting down system...")
    print("Detection log saved to 'detection_log.csv'")

finally:
    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows
    GPIO.cleanup()
    print("System shutdown complete.")
