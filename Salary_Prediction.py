
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():

    st.title("Employee Salary Prediction")
    st.markdown("### Enter Employee Details")

    # Load trained model
    model = joblib.load("Salary_Prediction.pkl")
    scalar = joblib.load("scaler.pkl")
    
    
    # Numerical Inputs
    age = st.number_input("Age", 18, 65, step=1)

    experience = st.number_input("Experience (Years)", 0, 40, step=1)

    performance = st.slider("Performance Rating", 1, 5, 3)

    certifications = st.number_input("Number of Certifications", 0, 20, step=1)

    overtime = st.number_input("Overtime Hours", 0, 200, step=1)

    tenure = st.number_input("Company Tenure (Years)", 0, 40, step=1)

    projects = st.number_input("Projects Completed", 0, 100, step=1)

    skill = st.slider("Skill Score", 0, 100, 50)

    # Categorical Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])

    education = st.selectbox(
        "Education",
        ["High School", "Bachelor", "Master", "PhD"]
    )

    department = st.selectbox(
        "Department",
        ["HR", "Finance", "IT", "Marketing", "Sales", "Operations"]
    )

    job = st.selectbox(
        "Job Level",
        ["Junior", "Mid", "Senior", "Lead", "Manager"]
    )

    remote = st.selectbox(
        "Remote Work",
        ["No", "Yes"]
    )

    city = st.selectbox(
        "City",
        ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Pune"]
    )

    # Manual Encoding
    gender = {"Male": 1, "Female": 0}[gender]

    education = {
        "High School": 0,
        "Bachelor": 1,
        "Master": 2,
        "PhD": 3
    }[education]

    department = {
        "HR": 0,
        "Finance": 1,
        "IT": 2,
        "Marketing": 3,
        "Sales": 4,
        "Operations": 5
    }[department]

    job = {
        "Junior": 0,
        "Mid": 1,
        "Senior": 2,
        "Lead": 3,
        "Manager": 4
    }[job]

    remote = {
        "No": 0,
        "Yes": 1
    }[remote]

    city = {
        "Delhi": 0,
        "Mumbai": 1,
        "Bangalore": 2,
        "Hyderabad": 3,
        "Chennai": 4,
        "Pune": 5
    }[city]

    input_df1 = pd.DataFrame({
        "Age":[age],
        "Gender":[gender],
        "Education":[education],
        "Experience_Years":[experience],
        "Department":[department],
        "Job_Level":[job],
        "Performance_Rating":[performance],
        "Certifications":[certifications],
        "Overtime_Hours":[overtime],
        "Remote_Work":[remote],
        "City":[city],
        "Company_Tenure":[tenure],
        "Projects_Completed":[projects],
        "Skill_Score":[skill]
    })
    scaled_data=scalar.transform(input_df1 )
    if st.button("Predict Salary"):
        prediction = model.predict(scaled_data)

        st.success(f"Predicted Annual Salary: ₹ {prediction[0]:,.2f} LPA")

if __name__ == "__main__":
    main()