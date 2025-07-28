import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Load dataset
csv_path = r"C:\Users\LENOVO\Desktop\resistor-reader\dataset\resistor_dataset.csv"
df = pd.read_csv(csv_path)

# Features and label
X = df[["h_mean", "s_mean", "v_mean"]]
y = df["label"]

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
