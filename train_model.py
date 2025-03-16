import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Load dataset
dataset_path = "CareerMap- Mapping Tech Roles With Personality & Skills.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

df = pd.read_csv(dataset_path)

# 🔹 Step 1: Handle Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)  # Suppress warning
 # Replace missing values with column mean

# 🔹 Step 2: Remove Duplicates
df.drop_duplicates(inplace=True)

# 🔹 Step 3: Encode the "Role" column
label_encoder = LabelEncoder()
df["Role_encoded"] = label_encoder.fit_transform(df["Role"])
df.drop(columns=["Role"], inplace=True)  # Drop original Role column

# 🔹 Step 4: Normalize Numerical Features
scaler = MinMaxScaler()
feature_columns = df.columns[:-1]
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# 🔹 Step 5: Split Dataset for Training
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["Role_encoded"]), df["Role_encoded"], test_size=0.2, random_state=42, stratify=df["Role_encoded"]
)

# 🔹 Step 6: Train Models
dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42).fit(X_train, y_train)

# 🔹 Step 7: Save Processed Models
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

joblib.dump(dt_model, os.path.join(models_dir, "decision_tree.pkl"))
joblib.dump(rf_model, os.path.join(models_dir, "random_forest.pkl"))
joblib.dump(nn_model, os.path.join(models_dir, "neural_network.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))

print("✅ Data cleaned, models trained, and saved successfully!")
