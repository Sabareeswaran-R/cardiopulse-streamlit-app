import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Data
df = pd.read_csv('heart.csv')
X = df.drop(columns=['target'])
y = df['target']

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model Training
model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Save Model Files
joblib.dump(model, 'random_forest_heart_model.pkl')
joblib.dump(scaler, 'scaler_heart.pkl')

print("Success! Model & Scaler files created successfully!")