import streamlit as st
import numpy as np
import joblib

# Load model and features
model = joblib.load("wine_model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

st.title("🍷 Wine Quality Predictor")
st.markdown(
    "This app predicts whether a red wine is **Good Quality** (rating ≥ 7) or **Not Good**, based on its chemical properties."
)

input_data = []
st.subheader("Enter Wine Chemical Properties:")
for feature in features:
    value = st.number_input(f"{feature}", format="%.4f")
    input_data.append(value)

if st.button("Predict Wine Quality"):
    input_array = np.array([input_data])
    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0][1]

    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.success("✅ This wine is **Good Quality**!")
    else:
        st.error("❌ This wine is **Not Good Quality**.")

    st.info(f"📊 Confidence Score: **{confidence * 100:.2f}%**")

st.markdown("---")
st.caption("Developed by Keth Lavene Abunda 💻")
