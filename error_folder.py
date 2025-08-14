import pandas as pd
import os
import shutil

# Path to your CSV file
csv_path = r"C:\Users\LENOVO\Desktop\resistor-reader\dataset\uji_coba\images\uji_coba_data.csv"
errors_folder = os.path.join(os.path.dirname(csv_path), "errors_only")

# Create errors folder if not exists
os.makedirs(errors_folder, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_path)

# Filter for wrong predictions
errors_df = df[df['status'].str.lower() == 'salah']

print(f"Found {len(errors_df)} error images.")

# Copy each error image
for img_path in errors_df['img_path']:
    if os.path.exists(img_path):
        shutil.copy(img_path, errors_folder)
    else:
        print(f"Warning: File not found {img_path}")

print(f"All error images copied to: {errors_folder}")
