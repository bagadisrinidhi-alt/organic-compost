import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import os

DATA_FILE = "crop_compost_requirements_extended.csv"

# ===============================
# Load & Train Model
# ===============================
@st.cache_resource
def load_and_train_model():
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ Dataset file '{DATA_FILE}' not found!")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    le = LabelEncoder()
    df["Crop_encoded"] = le.fit_transform(df["Crop"])

    X = df[["Crop_encoded"]]
    y = df[["Nitrogen (kg/acre)", "Carbon (kg/acre)",
            "Moisture (liters/acre)", "Minerals (kg/acre)"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    rf.fit(X_train, y_train)

    crop_lookup = {c.lower(): c for c in le.classes_}

    return rf, le, crop_lookup, df

def retrain_model():
    st.cache_resource.clear()  # clear cache so model retrains on next call
    return load_and_train_model()

# Load model
rf, le, crop_lookup, df = load_and_train_model()

# ===============================
# Prediction Function
# ===============================
def predict_requirements(crop_name):
    crop_name_lower = crop_name.lower()
    if crop_name_lower not in crop_lookup:
        return None, f"❌ Crop '{crop_name}' not found in dataset."

    actual_crop = crop_lookup[crop_name_lower]
    crop_encoded = le.transform([actual_crop])
    prediction = rf.predict([[crop_encoded[0]]])[0]

    result = {
        "Crop": actual_crop,
        "Nitrogen (kg/acre)": round(prediction[0], 2),
        "Carbon (kg/acre)": round(prediction[1], 2),
        "Moisture (liters/acre)": round(prediction[2], 2),
        "Minerals (kg/acre)": round(prediction[3], 2),
    }
    return result, None

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Crop Nutrient Predictor", page_icon="🌱", layout="wide")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app predicts **Nitrogen, Carbon, Moisture, and Minerals** requirements "
    "for different crops using a Machine Learning model. "
    "You can also add new crop data to improve predictions."
)

# Main Tabs
tab1, tab2 = st.tabs(["🔮 Prediction", "➕ Add New Crop"])

# -------------------------------
# Tab 1: Prediction
# -------------------------------
with tab1:
    st.markdown(
        "<h2 style='text-align:center;'>🌱 Smart Crop Nutrient Predictor</h2>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        crop_name = st.text_input("🔍 Enter crop name:")

    if st.button("🚀 Predict Requirements"):
        if crop_name.strip() == "":
            st.warning("⚠️ Please enter a crop name.")
        else:
            result, error = predict_requirements(crop_name)
            if error:
                st.error(error)
            else:
                st.success(f"Nutrient requirements for **{result['Crop']}**:")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("🌿 Nitrogen", f"{result['Nitrogen (kg/acre)']} kg/acre")
                    st.metric("💧 Moisture", f"{result['Moisture (liters/acre)']} liters/acre")
                with c2:
                    st.metric("🟢 Carbon", f"{result['Carbon (kg/acre)']} kg/acre")
                    st.metric("🪨 Minerals", f"{result['Minerals (kg/acre)']} kg/acre")

# -------------------------------
# Tab 2: Add New Crop
# -------------------------------
with tab2:
    st.markdown("### ➕ Add a New Crop to Dataset")

    with st.form("add_crop_form", clear_on_submit=True):
        crop_name = st.text_input("🌾 Crop Name")
        nitrogen = st.number_input("🌿 Nitrogen (kg/acre)", min_value=0, step=1)
        carbon = st.number_input("🟢 Carbon (kg/acre)", min_value=0, step=1)
        moisture = st.number_input("💧 Moisture (liters/acre)", min_value=0, step=10)
        minerals = st.number_input("🪨 Minerals (kg/acre)", min_value=0, step=1)

        submitted = st.form_submit_button("💾 Save Crop Data")

        if submitted:
            if crop_name.strip() == "":
                st.warning("⚠️ Crop name cannot be empty.")
            else:
                new_row = pd.DataFrame([{
                    "Crop": crop_name.strip().title(),
                    "Nitrogen (kg/acre)": nitrogen,
                    "Carbon (kg/acre)": carbon,
                    "Moisture (liters/acre)": moisture,
                    "Minerals (kg/acre)": minerals,
                }])

                # Append to CSV
                new_row.to_csv(DATA_FILE, mode="a", header=False, index=False)

                st.success(f"✅ Crop '{crop_name}' added successfully!")

                # Retrain model with new data
                rf, le, crop_lookup, df = retrain_model()
