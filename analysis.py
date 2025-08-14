import pandas as pd
import os

# === Load CSV ===
csv_path = r"C:\Users\LENOVO\Desktop\resistor-reader\dataset\uji_coba\images\uji_coba_data.csv"
df = pd.read_csv(csv_path)

# Pastikan 'status' dalam huruf kecil
df['status'] = df['status'].str.lower()

# === Function to calculate accuracy & MoE ===
def calc_stats(group):
    total = len(group)
    correct = (group['status'] == 'benar').sum()
    accuracy = (correct / total) * 100
    moe = ((total - correct) / total) * 100  # Error rate
    return pd.Series({
        'Total': total,
        'Benar': correct,
        'Akurasi (%)': round(accuracy, 2),
        'MoE (%)': round(moe, 2)
    })

# === Per Variable ===
lighting_stats = df.groupby('lighting', group_keys=False).apply(lambda g: calc_stats(g)).reset_index()
angle_stats = df.groupby('angle', group_keys=False).apply(lambda g: calc_stats(g)).reset_index()
bg_stats = df.groupby('bg', group_keys=False).apply(lambda g: calc_stats(g)).reset_index()

# === Per Full Composition ===
composition_stats = df.groupby(['lighting', 'angle', 'bg'], group_keys=False).apply(lambda g: calc_stats(g)).reset_index()

# === Find MAC ===
mac_row = composition_stats.loc[composition_stats['Akurasi (%)'].idxmax()]

# === Find Errors ===
error_samples = df[df['status'] == 'salah'][['label_asli', 'prediksi', 'confidence', 'lighting', 'angle', 'bg', 'lux', 'img_path']]

# === Save to Excel ===
output_excel = os.path.join(os.path.dirname(csv_path), "uji_coba_analysis.xlsx")
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    lighting_stats.to_excel(writer, sheet_name="Lighting Stats", index=False)
    angle_stats.to_excel(writer, sheet_name="Angle Stats", index=False)
    bg_stats.to_excel(writer, sheet_name="Background Stats", index=False)
    composition_stats.to_excel(writer, sheet_name="Composition Stats", index=False)
    pd.DataFrame([mac_row]).to_excel(writer, sheet_name="Most Accurate", index=False)
    error_samples.to_excel(writer, sheet_name="Error Samples", index=False)

print(f"Analysis saved to {output_excel}")
print("\n Most Accurate Composition (MAC):")
print(mac_row)
