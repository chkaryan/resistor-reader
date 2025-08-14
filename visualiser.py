import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Path to CSV ===
csv_path = r"C:\Users\LENOVO\Desktop\resistor-reader\dataset\uji_coba\images\uji_coba_data.csv"
output_dir = os.path.dirname(csv_path)

df = pd.read_csv(csv_path)

# === Helper to calculate stats ===
def calc_stats(group):
    total = len(group)
    benar = (group['status'].str.lower() == 'benar').sum()
    salah = total - benar
    accuracy = (benar / total) * 100 if total > 0 else 0
    error = (salah / total) * 100 if total > 0 else 0
    return pd.Series({'accuracy': accuracy, 'error': error})

# === Helper to annotate bars ===
def annotate_bars(ax):
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 3),
                    textcoords='offset points')

# === Grouped stats ===
lighting_stats = df.groupby('lighting').apply(calc_stats).reset_index()
angle_stats = df.groupby('angle').apply(calc_stats).reset_index()
bg_stats = df.groupby('bg').apply(calc_stats).reset_index()
composition_stats = df.groupby(['lighting', 'angle', 'bg']).apply(calc_stats).reset_index()

# === Save stats to Excel ===
output_excel = os.path.join(output_dir, "data_uji_analysis.xlsx")
with pd.ExcelWriter(output_excel) as writer:
    lighting_stats.to_excel(writer, sheet_name="Lighting Stats", index=False)
    angle_stats.to_excel(writer, sheet_name="Angle Stats", index=False)
    bg_stats.to_excel(writer, sheet_name="Background Stats", index=False)
    composition_stats.to_excel(writer, sheet_name="Composition Stats", index=False)

# === Chart 1: Lighting Accuracy ===
plt.figure(figsize=(6, 4))
ax = sns.barplot(data=lighting_stats, x='lighting', y='accuracy', palette='Blues_r')
annotate_bars(ax)
plt.ylim(0, 100)
plt.title('Akurasi per Pencahayaan')
plt.ylabel('Akurasi (%)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lighting_accuracy.png"), dpi=300)
plt.close()

# === Chart 2: Angle Accuracy ===
plt.figure(figsize=(6, 4))
ax = sns.barplot(data=angle_stats, x='angle', y='accuracy', palette='Greens_r')
annotate_bars(ax)
plt.ylim(0, 100)
plt.title('Akurasi per Sudut Kamera')
plt.ylabel('Akurasi (%)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "angle_accuracy.png"), dpi=300)
plt.close()

# === Chart 3: Background Accuracy ===
plt.figure(figsize=(6, 4))
ax = sns.barplot(data=bg_stats, x='bg', y='accuracy', palette='Oranges_r')
annotate_bars(ax)
plt.ylim(0, 100)
plt.title('Akurasi per Latar Belakang')
plt.ylabel('Akurasi (%)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "background_accuracy.png"), dpi=300)
plt.close()

# === Chart 4: Accuracy per Composition ===
plt.figure(figsize=(12, 6))
comp_labels = composition_stats[['lighting', 'angle', 'bg']].agg(' - '.join, axis=1)
ax = sns.barplot(
    data=composition_stats,
    x='accuracy',
    y=comp_labels,
    palette='Purples_r'
)
for i, (value) in enumerate(composition_stats['accuracy']):
    ax.annotate(f"{value:.2f}%", (value, i), ha='left', va='center', fontsize=8, color='black')

plt.xlabel('Akurasi (%)')
plt.ylabel('Komposisi')
plt.title('Akurasi per Komposisi (Lighting + Posisi Kamera + Backdrop)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_per_composition.png"), dpi=300)
plt.close()

# === Chart 5: KAK & KES Summary ===
kak_row = composition_stats.loc[composition_stats['accuracy'].idxmax()]
kes_row = composition_stats.loc[composition_stats['error'].idxmax()]

kak_kes_df = pd.DataFrame({
    'Komposisi': [
        f"{kak_row['lighting']} - {kak_row['angle']} - {kak_row['bg']}",
        f"{kes_row['lighting']} - {kes_row['angle']} - {kes_row['bg']}"
    ],
    'Accuracy (%)': [kak_row['accuracy'], kes_row['accuracy']],
    'Error (%)': [kak_row['error'], kes_row['error']],
    'Kategori': ['KAK (Akurasi Tertinggi)', 'KES (Error Tertinggi)']
})

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=kak_kes_df,
    x='Kategori', y='Accuracy (%)', hue='Komposisi', dodge=False,
    palette=['green', 'red']
)
annotate_bars(ax)
plt.title('Ringkasan KAK & KES')
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kak_kes_summary.png"), dpi=300)
plt.close()

print(f"Hasil analisis disimpan di: {output_excel}")
print(f"Chart saved in: {output_dir}")
