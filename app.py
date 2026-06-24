"""
Ames Housing Price Predictor
Streamlit app for the trained regression model.
"""

import streamlit as st
import numpy as np
import joblib

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load('ames_best_model.pkl')
    scaler = joblib.load('ames_scaler.pkl')
    feature_names = joblib.load('ames_feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ============================================================
# APP TITLE
# ============================================================

st.title("🏠 Ames Housing Price Predictor")
st.markdown("Built with Linear Regression + Feature Engineering | R² = 0.90")

# ============================================================
# USER INPUTS (Sidebar)
# ============================================================

st.sidebar.header("House Features")

# Core features (must match your training features)
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
garage_area = st.sidebar.number_input("Garage Area (sq ft)", 0, 1500, 400)
first_flr_sf = st.sidebar.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)
year_built = st.sidebar.number_input("Year Built", 1872, 2026, 2000)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 5, 2)
fireplaces = st.sidebar.slider("Fireplaces", 0, 4, 0)
lot_area = st.sidebar.number_input("Lot Area (sq ft)", 1000, 50000, 8000)
mas_vnr_area = st.sidebar.number_input("Masonry Veneer Area (sq ft)", 0, 1000, 0)
open_porch_sf = st.sidebar.number_input("Open Porch Area (sq ft)", 0, 500, 0)
wood_deck_sf = st.sidebar.number_input("Wood Deck Area (sq ft)", 0, 800, 0)

# ============================================================
# ENGINEER FEATURES (Match training exactly)
# ============================================================

# Build input dictionary
input_dict = {
    'Overall Qual': overall_qual,
    'Gr Liv Area': gr_liv_area,
    'Garage Area': garage_area,
    '1st Flr SF': first_flr_sf,
    'Year Built': year_built,
    'Full Bath': full_bath,
    'Fireplaces': fireplaces,
    'Lot Area': lot_area,
    'Mas Vnr Area': mas_vnr_area,
    'Open Porch SF': open_porch_sf,
    'Wood Deck SF': wood_deck_sf,
    # Add remaining features from your feature_names with defaults
}

# Add engineered features
input_dict['Overall Qual_sq'] = overall_qual ** 2
input_dict['Gr Liv Area_sq'] = gr_liv_area ** 2
input_dict['Qual_x_Area'] = overall_qual * gr_liv_area
input_dict['HouseAge'] = 2026 - year_built

# Ensure all features exist (fill missing with 0)
for feat in feature_names:
    if feat not in input_dict:
        input_dict[feat] = 0

# Reorder to match training
features = np.array([[input_dict[f] for f in feature_names]])

# ============================================================
# SCALE + PREDICT
# ============================================================

features_scaled = scaler.transform(features)
log_pred = model.predict(features_scaled)[0]
prediction = np.expm1(log_pred)

# ============================================================
# DISPLAY RESULT
# ============================================================

st.markdown("---")
st.subheader("Estimated Price")
st.success(f"**${prediction:,.0f}**")

st.caption("Built by Ahmed Iprahin | Junior Data Scientist")

# Show feature breakdown
with st.expander("See feature values"):
    st.write({k: v for k, v in zip(feature_names, features[0])})