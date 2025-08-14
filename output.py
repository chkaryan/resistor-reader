import cv2
import numpy as np
import joblib
from gtts import gTTS
import pygame
import os
import RPi.GPIO as GPIO
import time

# ===== GPIO Setup =====
BUTTON_PIN = 17  # GPIO 17 (Pin 11)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Button to GND

# ===== Load trained model =====
model = joblib.load("knn_model.joblib")

# ===== Label to spoken form mapping =====
spoken_text = {
    "10k": "sepuluh ribu",
    "470": "empat ratus tujuh puluh",
    "680": "enam ratus delapan puluh"
}

# ===== Function to speak resistor value =====
def speak_resistor_value(value_ohm):
    text = f"Resistor {value_ohm} Ohm"
    filename = "resistor_output.mp3"
    tts = gTTS(text=text, lang='id')
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    os.remove(filename)

# ===== Camera setup =====
cap = cv2.VideoCapture(0)
current_value = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not detected.")
            break

        # Resize and extract HSV mean values
        resized = cv2.resize(frame, (200, 200))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)

        # Predict resistor value
        features = np.array([[h_mean, s_mean, v_mean]])
        predicted_value = model.predict(features)[0]
        current_value = spoken_text.get(predicted_value, None)

        # If button pressed (LOW because of pull-up), speak
        if GPIO.input(BUTTON_PIN) == GPIO.LOW and current_value:
            print(f"Button pressed → Speaking: {predicted_value} Ohm")
            speak_resistor_value(current_value)
            time.sleep(0.5)  # debounce

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    cap.release()
    GPIO.cleanup()
