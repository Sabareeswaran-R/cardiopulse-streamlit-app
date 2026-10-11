# 🫀 CardioPulse Clinical Cockpit

An interactive clinical decision support web application built with **Streamlit**, powered by a **Random Forest Classifier** trained on the UCI Heart Disease dataset. **CardioPulse** evaluates patient vitals and clinical indicators to generate real-time diagnostic risk assessments and explainable feature impact visualizations.

---

## 📌 Features

- **Clinical Parameter Controls**: Interactive sidebar for adjusting patient vitals (Age, Gender, Resting BP, Serum Cholesterol, ST Depression, Thalassemia, etc.).
- **Diagnostic Risk Stratification**: Binary risk classification (`HIGH RISK` vs. `LOW RISK`) with calibrated disease probability outputs.
- **Dynamic Feature Influence Analysis**: Viridis-styled horizontal bar visualizations dynamically reflecting weighted feature contributions for individual input profiles[cite: 1, 5, 6].
- **Dark Clinical Dashboard UI**: Custom-styled Streamlit interface optimized for readable clinical review.

---

## 🏗️ Project Architecture

```text
├── app.py                         # Streamlit application dashboard
├── train.py                       # Model training & serialization pipeline
├── heart.csv                      # Clinical heart disease dataset
├── random_forest_heart_model.pkl  # Trained Random Forest Classifier model
├── scaler_heart.pkl               # Fitted StandardScaler object
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
