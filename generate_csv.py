import cv2
import numpy as np
import os
import pandas as pd

# Change this to match your path
base_dir = r"C:\Users\LENOVO\Desktop\resistor-reader\dataset"
output_csv = os.path.join(base_dir, "resistor_dataset.csv")

data_rows = []

for folder_name in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    try:
        label, lighting, angle, bg = folder_name.split("_")
    except ValueError:
        print(f"Skipping folder with unexpected format: {folder_name}")
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: could not read {img_path}")
                continue

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h_mean = np.mean(h)
            s_mean = np.mean(s)
            v_mean = np.mean(v)

            data_rows.append([h_mean, s_mean, v_mean, label, lighting, angle, bg])

# Save to CSV
df = pd.DataFrame(data_rows, columns=["h_mean", "s_mean", "v_mean", "label", "lighting", "angle", "bg"])
df.to_csv(output_csv, index=False)

print("CSV created:", output_csv)
