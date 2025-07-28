import cv2
import numpy as np
import joblib
from gtts import gTTS
from playsound import playsound
import os

# Load trained model
model = joblib.load("knn_model.joblib")

# Open webcam
cap = cv2.VideoCapture(0)

# Define a function to speak in Bahasa Indonesia
def speak_resistor_value(value_ohm):
    text = f"Resistor {value_ohm} Ohm"
    tts = gTTS(text=text, lang='id')
    filename = "resistor_output.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# Label to spoken form mapping
spoken_text = {
    "10k": "sepuluh ribu",
    "470": "empat ratus tujuh puluh",
    "680": "enam ratus delapan puluh"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize or crop frame as needed
    resized = cv2.resize(frame, (200, 200))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)

    features = np.array([[h_mean, s_mean, v_mean]])
    predicted_value = model.predict(features)[0]

    # Display the result on the frame
    cv2.putText(frame, f"Nilai: {predicted_value} Ohm", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Resistor Reader", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        speak_resistor_value(spoken_text[predicted_value])
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
