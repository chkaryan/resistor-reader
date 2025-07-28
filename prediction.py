import cv2
import numpy as np
from joblib import load

# Load the trained KNN model
model_path = r"C:\Users\LENOVO\Desktop\resistor-reader\knn_model.joblib"
knn_model = load(model_path)

# Start the webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
if not cap.isOpened():
    print("Kamera tidak terdeteksi.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    # Resize the image if needed (optional)
    frame_resized = cv2.resize(frame, (640, 480))

    # Show the live preview
    cv2.imshow("Live Preview", frame_resized)

    # Capture frame on 'c' press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Keluar...")
        break

    elif key == ord('c'):  # Capture image on 'c' press
        # Convert to HSV
        hsv_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

        # Split into H, S, V channels
        h, s, v = cv2.split(hsv_image)

        # Calculate the mean of each channel
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)

        # Prepare feature array
        feature = np.array([[h_mean, s_mean, v_mean]])

        # Make prediction
        prediction = knn_model.predict(feature)
        print(f"Predicted Resistor Value: {prediction[0]}")

        # Display the predicted resistor value on the image
        cv2.putText(frame_resized, f"Predicted: {prediction[0]}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image with the prediction
        cv2.imshow("Predicted Resistor Value", frame_resized)

cap.release()
cv2.destroyAllWindows()
