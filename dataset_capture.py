import cv2
import os
import time
from datetime import datetime

# ==== SETTINGS ====
cam_index = 0  # Webcam is usually 0
burst_count = 75  # Number of images per capture session
delay_between_shots = 0.1  # Delay between burst shots in seconds

# Base path where your dataset will be saved
base_folder = os.path.join(os.path.expanduser("~"), "Desktop", "resistor-reader", "dataset")

# Prompt user to enter all setup variables
resistor_value = input("Masukkan nilai resistor (contoh: 10k, 680, 470): ").strip()
lighting_type = input("Masukkan jenis pencahayaan (contoh: lamp, natural, yellow): ").strip()
camera_angle = input("Masukkan posisi kamera (contoh: top, front, tilted): ").strip()
background_color = input("Masukkan warna latar belakang (contoh: white, black, turquoise): ").strip()
lux_value = input("Masukkan nilai lux saat pengambilan gambar: ").strip()

# Folder name generation
folder_name = f"{resistor_value}_{lighting_type}_{camera_angle}_{background_color}"
folder_path = os.path.join(base_folder, folder_name)

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print("Kamera tidak terdeteksi.")
    exit()

print(f"\n Siap mengambil {burst_count} gambar...")
print("Tekan 's' untuk mulai, atau 'q' untuk keluar.")
cv2.namedWindow("Live Preview")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka kamera.")
        break

    # Show live preview
    cv2.imshow("Live Preview", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Keluar...")
        break

    elif key == ord('s'):
        print("Memulai pengambilan gambar...")
        for i in range(burst_count):
            ret, frame = cap.read()
            if not ret:
                continue

            filename = f"{resistor_value}_{str(i+1).zfill(3)}.jpg"
            save_path = os.path.join(folder_path, filename)

            cv2.imwrite(save_path, frame)
            print(f"Gambar disimpan: {filename}")
            time.sleep(delay_between_shots)

        # Save lux value as text file
        with open(os.path.join(folder_path, "lux.txt"), "w") as f:
            f.write(f"Lux: {lux_value}\n")
            f.write(f"Captured: {datetime.now()}\n")

        print(f"\n Pengambilan selesai. {burst_count} gambar disimpan di: {folder_path}")
        break

cap.release()
cv2.destroyAllWindows()
