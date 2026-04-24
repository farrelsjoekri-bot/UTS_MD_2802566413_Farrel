import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
# =====================
# LOAD MODELS
# =====================


BASE_DIR = Path(__file__).resolve().parent.parent  

clf_path = BASE_DIR / "models" / "best_classmodel.pkl"
reg_path = BASE_DIR / "models" / "best_regmodel.pkl"

clf_model = joblib.load(clf_path)
reg_model = joblib.load(reg_path)

st.set_page_config(page_title="Student Placement Predictor", layout="wide")

st.title("🎓 Student Placement & Salary Prediction")

# =====================
# SIDEBAR
# =====================
st.sidebar.header("Navigation")
task = st.sidebar.radio("Choose Task", ["Classification", "Regression"])

st.sidebar.markdown("---")
st.sidebar.write("Model Info")
st.sidebar.write("• Classification → Placement Status")
st.sidebar.write("• Regression → Salary Prediction")

# =====================
# INPUT FORM
# =====================
st.header("Input Student Data")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
    study_hours = st.number_input("Study Hours per Day", 0.0, 12.0, 3.0)

with col2:
    attendance = st.number_input("Attendance %", 0, 100, 75)
    projects = st.number_input("Projects Completed", 0, 20, 2)
    internships = st.number_input("Internships Completed", 0, 10, 1)

with col3:
    coding = st.slider("Coding Skill", 1, 10, 5)
    communication = st.slider("Communication Skill", 1, 10, 5)
    aptitude = st.slider("Aptitude Skill", 1, 10, 5)

# =====================
# DATAFRAME INPUT
# =====================
input_data = pd.DataFrame({
    "gender": [gender],
    "cgpa": [cgpa],
    "study_hours_per_day": [study_hours],
    "attendance_percentage": [attendance],
    "projects_completed": [projects],
    "internships_completed": [internships],
    "coding_skill_rating": [coding],
    "communication_skill_rating": [communication],
    "aptitude_skill_rating": [aptitude]
})

# =====================
# PREDICTION
# =====================
if st.button("Predict"):
    if task == "Classification":
        pred = clf_model.predict(input_data)[0]

        st.subheader("📌 Placement Prediction")
        if pred == 1:
            st.success("Placed ✅")
        else:
            st.error("Not Placed ❌")

    else:
        pred = reg_model.predict(input_data)[0]

        st.subheader("💰 Salary Prediction")
        st.info(f"Estimated Salary: {pred:.2f} LPA")

# =====================
# VISUALIZATION
# =====================
st.markdown("---")
st.subheader("Input Summary")
st.dataframe(input_data)