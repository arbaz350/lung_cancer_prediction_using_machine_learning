import streamlit as st
from datetime import date, datetime
import time
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os

# ---------------------------------------------------------------------
# Streamlit base config
# ---------------------------------------------------------------------
st.set_page_config(layout="wide")

# Dates for the pickers
MIN_DATE = date(1900, 1, 1)
MAX_DATE = date.today()

# ---------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    label {
        color: #c07a7a !important;
        font-weight: bold;
    }
    .toast {
        background-color: #c07a7a;
        color: white;
        padding: 15px;
        border-radius: 10px;
        position: fixed;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        animation: slideUp 0.4s ease-out, fadeOut 0.4s ease-out 3s forwards;
    }
    @keyframes slideUp {
        from { bottom: -100px; opacity: 0; }
        to { bottom: 10px; opacity: 1; }
    }
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Toast session-state
if "show_notification" not in st.session_state:
    st.session_state.show_notification = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("Info")
st.sidebar.write("Group No. 4 PG-DBDA project")
st.sidebar.write("[Tableau Visualizations](https://public.tableau.com/app/profile/akash.vishwakarma5526/viz/lung-cancer-extra-tree-classifier/AgeDistributionBySurvivalStatus?publish=yes)")

# ---------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------
st.markdown("<h1 style='color: #c07a7a;'>Lung Cancer Survival Prediction</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------
with st.form(key="user_info_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input("Name :", placeholder="Enter your name here")
        age = st.number_input("Age :", min_value=0, max_value=120, step=1)
        bmi = st.number_input("BMI :", min_value=0.0, step=0.1, format="%.1f")
        cholesterol = st.number_input("Cholesterol level :", min_value=0.0, step=1.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        family_history = st.selectbox("Family History :", ["Yes", "No"])

    with col2:
        smoking_status = st.selectbox("Smoking Status :", ['Never Smoked', 'Former Smoker', 'Passive Smoker', 'Current Smoker'])
        treatement_type = st.selectbox("Treatment Type :", ['Surgery', 'Radiation', 'Chemotherapy', 'Combined'])
        diagnosis_date = st.date_input("Date of Diagnosis", value=date.today(), min_value=MIN_DATE, max_value=MAX_DATE)
        begining_of_treatment_date = st.date_input("Beginning of treatment date :", value=date.today(), min_value=MIN_DATE, max_value=MAX_DATE)
        end_treatment_date = st.date_input("End of treatment date :", value=date.today(), min_value=MIN_DATE, max_value=MAX_DATE)
        cancer_stage = st.selectbox("Cancer Stage", ["I", "II", "III", "IV"])

    with col3:
        hypertension = st.radio("Hypertension", ["Yes", "No"])
        st.markdown("<br>", unsafe_allow_html=True)
        asthma = st.radio("Asthma", ["Yes", "No"])
        st.markdown("<br>", unsafe_allow_html=True)
        cirrhosis = st.radio("Cirrhosis", ["Yes", "No"])
        st.markdown("<br>", unsafe_allow_html=True)
        other_cancer = st.radio("Other Cancer", ["Yes", "No"])

    submitted = st.form_submit_button("Submit")

# ---------------------------------------------------------------------
# If submitted, show toast then continue
# ---------------------------------------------------------------------
if submitted:
    st.session_state.show_notification = True
    st.session_state.user_name = name
    st.rerun()

if st.session_state.show_notification:
    st.markdown(
        f"""
        <div class="toast">
            <strong>Your form has been successfully submitted ✔️</strong> {st.session_state.user_name}
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(1.5)
    st.session_state.show_notification = False

st.divider()

# ---------------------------------------------------------------------
# Map categorical → numeric
# ---------------------------------------------------------------------
smoking_map = {'Never Smoked': 0, 'Former Smoker': 1, 'Passive Smoker': 2, 'Current Smoker': 3}
treat_map = {'Surgery': 0, 'Radiation': 1, 'Chemotherapy': 2, 'Combined': 3}
stage_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
yn_map = {'Yes': 1, 'No': 0}
gender_map = {'Male': 1, 'Female': 0}

smoking_status_num = smoking_map[smoking_status]
treatement_type_num = treat_map[treatement_type]
cancer_stage_num = stage_map[cancer_stage]
hypertension_num = yn_map[hypertension]
asthma_num = yn_map[asthma]
cirrhosis_num = yn_map[cirrhosis]
other_cancer_num = yn_map[other_cancer]
gender_num = gender_map[gender]
family_history_num = yn_map[family_history]

# ---------------------------------------------------------------------
# Derived numeric features
# ---------------------------------------------------------------------
treatment_delay_days = (begining_of_treatment_date - diagnosis_date).days
treatment_duration_days = (end_treatment_date - begining_of_treatment_date).days

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
MODEL_PATH = "Lung_Cancer_model.sav"

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.stop()
    return joblib.load(path)

model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------
# Feature vector (ensure everything is numeric!)
# ---------------------------------------------------------------------
features = np.array([[
    float(age),
    float(cancer_stage_num),
    float(smoking_status_num),
    float(bmi),
    float(cholesterol),
    float(hypertension_num),
    float(asthma_num),
    float(cirrhosis_num),
    float(other_cancer_num),
    float(treatement_type_num),
    float(gender_num),
    float(family_history_num),
    float(treatment_delay_days),
    float(treatment_duration_days)
]], dtype=float)

# ---------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(features)
        proba = getattr(model, "predict_proba", None)
        if proba is not None:
            probability = model.predict_proba(features)[0][1]  # assuming class 1 is "survived"
        else:
            probability = None

        result_label = "SURVIVED" if prediction[0] == 1 else "NOT SURVIVED"
        if prediction[0] == 1:
            st.success("✅ The model predicts a HIGH chance of survival!")
        else:
            st.error("⚠️ The model predicts a LOW chance of survival.")

        if probability is not None:
            st.write(f"Predicted probability of survival (class=1): **{probability:.2%}**")

        # -----------------------------------------------------------------
        # Dump to MySQL
        # -----------------------------------------------------------------
        data_dict = {
            "name": [name],
            "age": [age],
            "bmi": [bmi],
            "cholesterol": [cholesterol],
            "gender": [gender_num],
            "family_history": [family_history_num],
            "smoking_status": [smoking_status_num],
            "treatement_type": [treatement_type_num],
            "diagnosis_date": [diagnosis_date],
            "begining_of_treatement": [begining_of_treatment_date],
            "end_treatment_date": [end_treatment_date],
            "cancer_stage": [cancer_stage_num],
            "hypertension": [hypertension_num],
            "asthma": [asthma_num],
            "cirrhosis": [cirrhosis_num],
            "other_cancer": [other_cancer_num],
            "treatment_delay_days": [treatment_delay_days],
            "treatment_duration_days": [treatment_duration_days],
            "result": [result_label]
        }

        DATABASE_URI = "mysql+pymysql://root:Shaikh123#@localhost:3306/project"  # <- change if needed
        engine = create_engine(DATABASE_URI)

        df = pd.DataFrame(data_dict)
        df.to_sql(name="output1", con=engine, if_exists="append", index=False)
        st.info("Data inserted successfully into MySQL table `output1`.")

    except Exception as e:
        st.exception(e)

st.write("End")
