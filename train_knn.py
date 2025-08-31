import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from scipy.stats import chi2_contingency
import seaborn as sns

# Load dataset
csv_path = r"C:\Users\LENOVO\Desktop\resistor-reader\dataset\training\resistor_dataset.csv"
df = pd.read_csv(csv_path)

# Display first few lines
print("First few lines of dataset:")
print(df.head())

# Features and label
X = df[["h_mean", "s_mean", "v_mean"]].values
y = df["label"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
model_path = r"C:\Users\LENOVO\Desktop\resistor-reader\knn_model.joblib"
dump(knn, model_path)
print("Model saved at:", model_path)

# === 3D Visualization ===
print("\nGenerating 3D KNN decision boundary plot...")

# Convert string labels to numeric for plotting
unique_labels = np.unique(y)
label_to_num = {label: i for i, label in enumerate(unique_labels)}
y_numeric = np.array([label_to_num[label] for label in y])

# Generate a grid of points for decision boundary
h_range = np.linspace(df['h_mean'].min(), df['h_mean'].max(), 20)
s_range = np.linspace(df['s_mean'].min(), df['s_mean'].max(), 20)
v_range = np.linspace(df['v_mean'].min(), df['v_mean'].max(), 20)
grid = np.array(np.meshgrid(h_range, s_range, v_range)).T.reshape(-1, 3)
predictions = knn.predict(grid)
predictions_numeric = np.array([label_to_num[label] for label in predictions])

# Get number of classes for colormap
n_classes = len(unique_labels)

# Create discrete colormap based on number of classes
colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
cmap_discrete = ListedColormap(colors)

# Plot setup (3D scatter)
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot training data points
scatter_train = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_numeric, cmap=cmap_discrete, 
                          s=60, edgecolor='black', linewidth=0.5, alpha=0.8, label='Training Data')

# Plot decision boundary points with transparency
ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], c=predictions_numeric, cmap=cmap_discrete, 
          alpha=0.1, s=8, label='Decision Boundary')

# Labels and title
ax.set_xlabel('Hue Mean')
ax.set_ylabel('Saturation Mean')
ax.set_zlabel('Value Mean')
ax.set_title(f'KNN Decision Boundary in HSV Color Space\n(k={knn.n_neighbors}, Accuracy: {accuracy:.2f})')

# Create legend with actual labels
legend_elements = []
for i, label in enumerate(unique_labels):
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label=f'Class: {label}', 
               markerfacecolor=colors[i], markersize=10)
    )

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# Improve plot appearance
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save the 3D plot
plot_path_3d = r"C:\Users\LENOVO\Desktop\resistor-reader\knn_decision_boundary_3d.png"
plt.savefig(plot_path_3d, dpi=300, bbox_inches='tight')
print(f"3D plot saved at: {plot_path_3d}")

plt.show()

# === Chi-Square Test Analysis ===
print("\nPerforming Chi-Square test analysis...")

def create_bins_and_test(data, labels, feature_name, n_bins=5):
    """Create bins for continuous data and perform chi-square test"""
    # Create bins for the feature
    bins = pd.cut(data, bins=n_bins, labels=[f'Bin_{i+1}' for i in range(n_bins)])
    
    # Create contingency table
    contingency_table = pd.crosstab(bins, labels)
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return contingency_table, chi2, p_value, dof, expected, bins

# Perform chi-square tests for each HSV feature
features = ['h_mean', 's_mean', 'v_mean']
feature_names = ['Hue Mean', 'Saturation Mean', 'Value Mean']
chi2_results = {}

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, (feature, feature_name) in enumerate(zip(features, feature_names)):
    # Perform chi-square test
    contingency_table, chi2, p_value, dof, expected, bins = create_bins_and_test(
        df[feature], df['label'], feature_name, n_bins=5
    )
    
    # Store results
    chi2_results[feature] = {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'contingency_table': contingency_table
    }
    
    # Create heatmap of contingency table
    ax = axes[i]
    sns.heatmap(contingency_table.T, annot=True, fmt='d', cmap='Blues', 
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(f'{feature_name} vs Resistor Class\nχ² = {chi2:.2f}, p = {p_value:.4f}')
    ax.set_xlabel(f'{feature_name} Bins')
    ax.set_ylabel('Resistor Class')

# Summary plot showing all chi-square statistics
ax = axes[3]
chi2_values = [chi2_results[feature]['chi2'] for feature in features]
p_values = [chi2_results[feature]['p_value'] for feature in features]

# Create bar plot for chi-square values
bars = ax.bar(feature_names, chi2_values, color=['skyblue', 'lightgreen', 'lightcoral'])
ax.set_ylabel('Chi-Square Statistic')
ax.set_title('Chi-Square Test Results\nfor HSV Features vs Resistor Classes')
ax.grid(axis='y', alpha=0.3)

# Add p-value annotations on bars
for bar, p_val in zip(bars, p_values):
    height = bar.get_height()
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'p={p_val:.4f}\n{significance}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Save the chi-square analysis plots
plot_path_chi2 = r"C:\Users\LENOVO\Desktop\resistor-reader\chi_square_analysis.png"
plt.savefig(plot_path_chi2, dpi=300, bbox_inches='tight')
print(f"Chi-square analysis saved at: {plot_path_chi2}")

plt.show()

# Print detailed chi-square results
print("\n" + "="*60)
print("CHI-SQUARE TEST RESULTS")
print("="*60)

for feature, feature_name in zip(features, feature_names):
    result = chi2_results[feature]
    print(f"\n{feature_name}:")
    print(f"  Chi-square statistic: {result['chi2']:.4f}")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Degrees of freedom: {result['dof']}")
    
    # Interpretation
    if result['p_value'] < 0.001:
        significance = "highly significant (p < 0.001)"
    elif result['p_value'] < 0.01:
        significance = "very significant (p < 0.01)"
    elif result['p_value'] < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    
    print(f"  Interpretation: The relationship is {significance}")

print("\n" + "="*60)

# === 2D Visualizations ===
print("\nGenerating 2D projection plots...")

# Create 2D projections
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define the three 2D projections
projections = [
    (0, 1, 'Hue Mean', 'Saturation Mean', 'H-S'),
    (0, 2, 'Hue Mean', 'Value Mean', 'H-V'),
    (1, 2, 'Saturation Mean', 'Value Mean', 'S-V')
]

for i, (x_idx, y_idx, x_label, y_label, title_suffix) in enumerate(projections):
    ax = axes[i]
    
    # Create 2D grid for this projection
    x_range = np.linspace(X[:, x_idx].min(), X[:, x_idx].max(), 100)
    y_range = np.linspace(X[:, y_idx].min(), X[:, y_idx].max(), 100)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # For predictions, we need to fill in the third dimension with median values
    grid_2d = np.zeros((xx.ravel().shape[0], 3))
    grid_2d[:, x_idx] = xx.ravel()
    grid_2d[:, y_idx] = yy.ravel()
    
    # Fill the missing dimension with median value
    missing_dim = 3 - x_idx - y_idx - 1
    grid_2d[:, missing_dim] = np.median(X[:, missing_dim])
    
    # Predict on 2D grid
    predictions_2d = knn.predict(grid_2d)
    predictions_2d_numeric = np.array([label_to_num[label] for label in predictions_2d])
    predictions_2d_numeric = predictions_2d_numeric.reshape(xx.shape)
    
    # Plot decision boundary as background
    ax.contourf(xx, yy, predictions_2d_numeric, alpha=0.3, cmap=cmap_discrete, levels=n_classes-1)
    
    # Plot training data points
    scatter = ax.scatter(X[:, x_idx], X[:, y_idx], c=y_numeric, cmap=cmap_discrete, 
                        s=50, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'KNN Decision Boundary ({title_suffix} Projection)\nAccuracy: {accuracy:.2f}')
    ax.grid(True, alpha=0.3)

# Create a single legend for all subplots
legend_elements = []
for i, label in enumerate(unique_labels):
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label=f'{label}Ω', 
               markerfacecolor=colors[i], markersize=10)
    )

fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=len(unique_labels))
plt.tight_layout()

# Save the 2D plots
plot_path_2d = r"C:\Users\LENOVO\Desktop\rr-copy\knn_decision_boundary_2d.png"
plt.savefig(plot_path_2d, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"2D plots saved at: {plot_path_2d}")

plt.show()

print("\nTraining complete with 3D visualization!")