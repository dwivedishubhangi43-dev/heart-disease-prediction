import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Risk Predictor")
st.markdown("---")

# ---------- LOAD MODEL FUNCTION ----------
@st.cache_resource
def load_model():
    """Load model files from models folder"""
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models")
        
        # Load model
        model = pickle.load(open(os.path.join(model_path, "heart_disease_model.pkl"), "rb"))
        
        # Load scaler
        scaler = pickle.load(open(os.path.join(model_path, "heart_scaler.pkl"), "rb"))
        
        # Load columns
        with open(os.path.join(model_path, "model_columns.json"), "r") as f:
            columns = json.load(f)
        
        return model, scaler, columns, True, "Model loaded successfully!"
    
    except FileNotFoundError as e:
        return None, None, None, False, f"Model files not found: {e}"
    except Exception as e:
        return None, None, None, False, f"Error loading model: {e}"

# Load model
model, scaler, columns, model_loaded, message = load_model()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("📊 Model Status")
    if model_loaded:
        st.success("✅ Model Ready!")
        st.metric("Expected Accuracy", "88.2%")
    else:
        st.error("❌ Model Not Loaded")
        st.warning(message)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.write("AI tool for heart disease risk prediction using Machine Learning.")
    st.caption("⚠️ Educational purposes only - Consult a doctor for medical advice.")

# ---------- MAIN INPUT FORM ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Patient Demographics")
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Gender", ["Male", "Female"])
    
    st.subheader("🔬 Clinical Measurements")
    chest_pain_options = {
        "Typical Angina (TA)": "TA",
        "Atypical Angina (ATA)": "ATA", 
        "Non-Anginal Pain (NAP)": "NAP",
        "Asymptomatic (ASY)": "ASY"
    }
    chest_pain_display = st.selectbox("Chest Pain Type", list(chest_pain_options.keys()))
    chest_pain = chest_pain_options[chest_pain_display]
    
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    st.subheader("⚡ Cardiac Function")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    
    ecg_options = {
        "Normal": "Normal",
        "ST-T wave abnormality": "ST",
        "Left Ventricular Hypertrophy": "LVH"
    }
    resting_ecg_display = st.selectbox("Resting ECG", list(ecg_options.keys()))
    resting_ecg = ecg_options[resting_ecg_display]
    
    max_hr = st.slider("Maximum Heart Rate", 60, 202, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
    
    slope_options = {
        "Upsloping": "Up",
        "Flat": "Flat",
        "Downsloping": "Down"
    }
    st_slope_display = st.selectbox("ST Slope", list(slope_options.keys()))
    st_slope = slope_options[st_slope_display]

# ---------- PREDICT BUTTON ----------
st.markdown("---")
predict_btn = st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True)

# ---------- PREDICTION LOGIC ----------
if predict_btn:
    if not model_loaded:
        st.error("❌ Cannot predict: Model files are missing!")
        st.info("📝 Please generate model files from Jupyter Notebook first.")
        st.code("""
# Run this in Jupyter Notebook:
import pickle, json, os
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
model_path = os.path.join(desktop, "Heart-Disease-Prediction", "models")
os.makedirs(model_path, exist_ok=True)
pickle.dump(best_model, open(os.path.join(model_path, "heart_disease_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(model_path, "heart_scaler.pkl"), "wb"))
with open(os.path.join(model_path, "model_columns.json"), "w") as f:
    json.dump(X.columns.tolist(), f)
print("Done!")
        """)
    else:
        # Prepare input data
        input_dict = {
            'Age': age,
            'Sex': 1 if sex == "Male" else 0,
            'ChestPainType': chest_pain,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': 1 if fasting_bs == "Yes" else 0,
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': 1 if exercise_angina == "Yes" else 0,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }
        
        # Convert to DataFrame and encode
        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'])
        
        # Ensure all columns match training data
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[columns]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display result
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        if prediction == 1:
            st.error(f"""
            ### ⚠️ HIGH RISK of Heart Disease
            
            **Probability: {probability[1]*100:.1f}%**
            
            Please consult a healthcare provider for proper evaluation.
            """)
        else:
            st.success(f"""
            ### ✅ LOW RISK of Heart Disease
            
            **Probability: {probability[0]*100:.1f}%**
            
            Maintain a healthy lifestyle with regular exercise and balanced diet.
            """)
        
        # Show risk factors
        st.markdown("---")
        st.subheader("🔍 Key Risk Factors Analysis")
        
        warnings = []
        if age > 50:
            warnings.append("• Age > 50: Regular cardiac screening recommended")
        if chest_pain == "ASY":
            warnings.append("• Asymptomatic chest pain: Requires immediate medical attention")
        if cholesterol > 240:
            warnings.append("• High cholesterol: Consider dietary changes")
        if resting_bp > 140:
            warnings.append("• Elevated blood pressure: Monitor regularly")
        if exercise_angina == "Yes":
            warnings.append("• Exercise-induced angina: Consult cardiologist")
        if st_slope == "Flat":
            warnings.append("• Flat ST slope: Strong indicator of heart disease")
        
        if warnings:
            st.warning("\n".join(warnings))
        else:
            st.info("✅ No critical risk factors identified")

st.markdown("---")
