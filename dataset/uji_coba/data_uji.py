import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from joblib import load
from sklearn.neighbors import KNeighborsClassifier

# === PATH SETUP ===
base_dir = os.path.dirname(os.path.abspath(__file__))  # current script folder
img_dir = os.path.join(base_dir, "images")
csv_file = os.path.join(img_dir, "uji_coba_data.csv")
model_path = "C:/Users/LENOVO/Desktop/resistor-reader/knn_model.joblib"

# === Ensure directories exist ===
os.makedirs(img_dir, exist_ok=True)

# === Load KNN Model ===
knn: KNeighborsClassifier = load(model_path)

# === Initialize camera ===
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Cannot access the camera.")
    exit()

# === Collect Metadata Once ===
label = input("Masukkan nilai resistor (label): ")
lighting = input("Jenis pencahayaan (contoh: natural/white/yellow): ")
angle = input("Sudut kamera (contoh: birds-eye/eye-level/high-angle): ")
bg = input("Warna latar belakang (contoh: putih/hitam/toska): ")
lux = input("Masukkan nilai lux dari lux meter (contoh: 150): ")

print("\nTekan 'c' untuk capture, 'q' untuk keluar.\n")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Gagal mengambil gambar.")
        continue

    cv2.imshow("Preview - Tekan 'c' untuk capture, 'q' untuk keluar", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"uji_{label}_{lighting}_{angle}_{bg}_{timestamp}.jpg"
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, frame)
        print(f"Gambar disimpan sebagai {img_path}")

        # Extract HSV features
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)

        # Predict with KNN
        x_input = np.array([[h_mean, s_mean, v_mean]])
        prediction = knn.predict(x_input)[0]
        proba = knn.predict_proba(x_input).max()
        print(f"Prediksi: {prediction} | Confidence: {proba*100:.2f}%")

        # Auto-confirm and auto-status
        confirmed_prediction = prediction
        status = "Benar" if prediction == label else "Salah"
        print(f"(Status otomatis: {status})")

        # Save to CSV
        data = {
            "h_mean": [h_mean],
            "s_mean": [s_mean],
            "v_mean": [v_mean],
            "label_asli": [label],
            "prediksi": [confirmed_prediction],
            "confidence": [round(proba * 100, 2)],
            "status": [status],
            "lighting": [lighting],
            "angle": [angle],
            "bg": [bg],
            "lux": [lux],
            "img_path": [img_path]
        }
        df_new = pd.DataFrame(data)

        if os.path.exists(csv_file):
            df_new.to_csv(csv_file, mode='a', index=False, header=False)
        else:
            df_new.to_csv(csv_file, index=False)

        print("Data berhasil disimpan ke CSV.\n")

    elif key == ord('q'):
        print("Selesai pengambilan data.")
        break

camera.release()
cv2.destroyAllWindows()
