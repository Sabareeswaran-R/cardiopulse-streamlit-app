import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="CardioPulse Clinical Cockpit",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom CSS
st.markdown(
    """
<style>
    .stApp { 
        background-color: #111318;
        color: #FFFFFF; 
    }
    
    section[data-testid="stSidebar"] {
        background-color: #1B1E2B;
        border-right: 1px solid #2B2F40;
    }
    
    label, 
    .stMarkdown p, 
    .stRadio label, 
    .stSlider label, 
    .stSelectbox label, 
    .stNumberInput label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stMarkdownContainer"] p,
    span[data-baseweb="tag"] {
        color: #F3F4F6 !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="select"] span, 
    div[data-baseweb="input"] input {
        color: #FFFFFF !important;
    }

    span[data-baseweb="tag"] {
        background-color: #EF4444 !important;
        border-radius: 6px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #111318 !important;
        border-color: #2B2F40 !important;
        color: #FFFFFF !important;
    }
    
    .header-box {
        background-color: #1B1E2B;
        padding: 20px 25px;
        border-radius: 12px;
        border: 1px solid #2B2F40;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .main-heading {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 28px;
        margin: 0;
    }
    
    .metric-card {
        background-color: #1B1E2B;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2B2F40;
        margin-bottom: 12px;
    }
    .metric-card h3 {
        color: #9CA3AF !important;
        font-size: 13px;
        margin: 0;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-card h2 {
        color: #FFFFFF !important;
        font-size: 24px;
        margin-top: 4px;
        margin-bottom: 0;
        font-weight: 800;
    }

    .sidebar-heading, .output-heading, .graph-heading {
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 22px;
        margin-bottom: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Main Title Display
st.markdown(
    """
<div class='header-box'>
    <span style='font-size: 30px;'>📊</span>
    <div>
        <h1 class='main-heading'>CardioPulse Diagnostic Dashboard</h1>
        <p style='color: #9CA3AF; margin: 0; font-size: 14px;'>Machine Learning Diagnostic Engine | Random Forest Model</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
  return joblib.load("random_forest_heart_model.pkl"), joblib.load(
      "scaler_heart.pkl"
  )


try:
  model, scaler = load_artifacts()
except Exception as e:
  st.error("Model/Scaler files missing! Run `python train.py` first.")
  st.stop()

# INPUTS SECTION
with st.sidebar:
  st.markdown(
      "<h2 class='sidebar-heading'>🔍 Patient Vitals Inputs</h2>",
      unsafe_allow_html=True,
  )

  age = st.slider("Select Age Range", 20, 90, 63)
  sex = st.radio(
      "Select Gender",
      [0, 1],
      format_func=lambda x: "Female" if x == 0 else "Male",
      index=1,
  )
  cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=0)
  trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 160)
  chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 286)
  fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], index=1)
  restecg = st.selectbox("Resting ECG Results", [0, 1, 2], index=0)
  thalach = st.slider("Max Heart Rate (thalach)", 60, 220, 108)
  exang = st.selectbox("Exercise Induced Angina", [0, 1], index=1)
  oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 3.2, step=0.1)
  slope = st.selectbox("Slope", [0, 1, 2], index=1)
  ca = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3, 4], index=2)
  thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3], index=3)

# Feature DataFrame Structure
feature_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
input_df = pd.DataFrame(
    [[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal,
    ]],
    columns=feature_names,
)

# Scaling and Prediction
scaled_input = scaler.transform(input_df)
pred = model.predict(scaled_input)[0]
probs = model.predict_proba(scaled_input)[0]

# --- IMPORTANT TARGET MAP FIX ---
# UCI Dataset Inversion Flag:
# If Class 0 = Heart Disease in your trained dataset, set target_is_inverted = True.
# If Class 1 = Heart Disease, set target_is_inverted = False.
target_is_inverted = True  # <-- Updated to True to fix 0.07% bug

if target_is_inverted:
  high_risk_prob = probs[0] * 100
  low_risk_prob = probs[1] * 100
  is_high_risk = (pred == 0) or (high_risk_prob >= 50)
else:
  high_risk_prob = probs[1] * 100
  low_risk_prob = probs[0] * 100
  is_high_risk = (pred == 1) or (high_risk_prob >= 50)

# Quick Vitals Stat Cards
m1, m2, m3, m4 = st.columns(4)
with m1:
  st.markdown(
      f"<div class='metric-card'><h3>Patient Age</h3><h2>{age} Yrs</h2></div>",
      unsafe_allow_html=True,
  )
with m2:
  st.markdown(
      f"<div class='metric-card'><h3>Cholesterol</h3><h2>{chol} mg/dl</h2></div>",
      unsafe_allow_html=True,
  )
with m3:
  st.markdown(
      f"<div class='metric-card'><h3>Max Heart Rate</h3><h2>{thalach}"
      " bpm</h2></div>",
      unsafe_allow_html=True,
  )
with m4:
  st.markdown(
      f"<div class='metric-card'><h3>Resting BP</h3><h2>{trestbps} mm"
      " Hg</h2></div>",
      unsafe_allow_html=True,
  )

st.divider()

col_out, col_graph = st.columns([1, 1.4])

# OUTPUT SECTION
with col_out:
  st.markdown(
      "<h3 class='output-heading'>🎯 Diagnostic Risk Output</h3>",
      unsafe_allow_html=True,
  )

  if is_high_risk:
    st.markdown(
        f"""
        <div style="background-color: #311319; border: 1px solid #EF4444; padding: 22px; border-radius: 12px;">
            <h2 style="color: #F87171; margin:0;">⚠️ HIGH RISK DIAGNOSED</h2>
            <h4 style="color: #FCA5A5; margin-top:8px;">Heart Disease Probability: <span style="font-size: 26px; color: #EF4444;">{high_risk_prob:.2f}%</span></h4>
            <p style="color: #FECDD3; margin-bottom:0;">Physiological risk factors indicate elevated likelihood of cardiac disease.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        f"""
        <div style="background-color: #062C22; border: 1px solid #10B981; padding: 22px; border-radius: 12px;">
            <h2 style="color: #34D399; margin:0;">✅ LOW RISK DIAGNOSED</h2>
            <h4 style="color: #6EE7B7; margin-top:8px;">Heart Disease Probability: <span style="font-size: 26px; color: #10B981;">{high_risk_prob:.2f}%</span></h4>
            <p style="color: #A7F3D0; margin-bottom:0;">Physiological factors remain within standard healthy metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# DYNAMIC GRAPH SECTION
with col_graph:
  st.markdown(
      "<h3 class='graph-heading'>📊 Feature Influence Analysis</h3>",
      unsafe_allow_html=True,
  )

  base_importance = model.feature_importances_
  scaled_val_abs = np.abs(scaled_input[0])
  dynamic_impact = base_importance * (1 + scaled_val_abs)
  dynamic_impact = dynamic_impact / dynamic_impact.sum()

  feat_df = (
      pd.DataFrame({"Feature": input_df.columns, "Importance": dynamic_impact})
      .sort_values(by="Importance", ascending=True)
      .tail(8)
  )

  norm = plt.Normalize(
      feat_df["Importance"].min(), feat_df["Importance"].max()
  )
  colors = cm.viridis(norm(feat_df["Importance"]))

  plt.clf()
  fig, ax = plt.subplots(figsize=(7, 3.8))
  fig.patch.set_facecolor("#111318")
  ax.set_facecolor("#111318")

  bars = ax.barh(
      feat_df["Feature"], feat_df["Importance"], color=colors, height=0.6
  )

  for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.003,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center",
        ha="left",
        fontsize=9,
        fontweight="bold",
        color="#FFFFFF",
    )

  ax.tick_params(colors="#FFFFFF", labelsize=10)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["bottom"].set_color("#2B2F40")
  ax.spines["left"].set_color("#2B2F40")
  ax.grid(axis="x", linestyle="--", alpha=0.3, color="#9CA3AF")

  st.pyplot(fig)
  plt.close(fig)