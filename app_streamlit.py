import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition Classifier", layout="wide")


@st.cache_resource
def load_model():
    # Save your trained model once with: joblib.dump(random_forest, "random_forest.pkl")
    return joblib.load("random_forest.pkl")


model = load_model()

st.title("Employee Attrition Classifier")
st.caption("Random Forest classification model: enter the employee's details to classify them as likely to stay or leave.")


def choice(label, options, key=None):
    """Selectbox that shows readable labels but returns the encoded value."""
    picked = st.selectbox(label, list(options.keys()), key=key)
    return options[picked]


LEVEL_0_3 = {"0": 0, "1": 1, "2": 2, "3": 3}

with st.form("attrition_form"):
    st.subheader("Personal")
    c1, c2, c3 = st.columns(3)
    with c1:
        Age_Years = st.number_input("Age", min_value=18.0, max_value=70.0, value=35.0, step=1.0)
        Gender = choice("Gender", {"Female": 0, "Male": 1})
    with c2:
        MaritalStatus = choice("Marital status", {"Divorced": 0, "Married": 1, "Single": 2})
        DistanceFromHome = st.number_input("Distance from home (0-28)", 0.0, 28.0, 5.0)
    with c3:
        Education = choice("Education level", {str(i): i for i in range(5)})
        EducationField = choice(
            "Education field",
            {"H.R": 0, "Life Sciences": 1, "Marketing": 2, "Medical": 3, "Other": 4, "Technical degree": 5},
        )

    st.subheader("Job")
    c1, c2, c3 = st.columns(3)
    with c1:
        BusinessTravel = choice("Business travel", {"Non-travel": 0, "Travel frequently": 1, "Travel rarely": 2})
        Department = choice("Department", {"H.R": 0, "Research": 1, "Sales": 2})
        JobRole = choice("Job role (encoded)", {str(i): i for i in range(9)})
    with c2:
        JobLevel = choice("Job level", {str(i): i for i in range(5)})
        JobInvolvement = choice("Job involvement", LEVEL_0_3)
        OverTime = choice("Works overtime", {"No": 0, "Yes": 1})
    with c3:
        PerformanceRating = choice("Performance rating (encoded)", {"0": 0, "1": 1})
        StockOptionLevel = choice("Stock option level", LEVEL_0_3)
        TrainingTimesLastYear = choice("Training times last year", {str(i): i for i in range(7)})

    st.subheader("Compensation")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        DailyRate = st.number_input("Daily rate (0-885)", 0.0, 885.0, 400.0)
    with c2:
        HourlyRate = st.number_input("Hourly rate (0-70)", 0.0, 70.0, 35.0)
    with c3:
        MonthlyIncome = st.number_input("Monthly income (0-1348)", 0.0, 1348.0, 500.0)
    with c4:
        MonthlyRate = st.number_input("Monthly rate (0-1426)", 0.0, 1426.0, 700.0)
    PercentSalaryHike = st.number_input("Percent salary hike (0-14)", 0.0, 14.0, 7.0)

    st.subheader("Satisfaction and work-life")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        EnvironmentSatisfaction = choice("Environment satisfaction", LEVEL_0_3)
    with c2:
        JobSatisfaction = choice("Job satisfaction", LEVEL_0_3)
    with c3:
        RelationshipSatisfaction = choice("Relationship satisfaction", LEVEL_0_3)
    with c4:
        WorkLifeBalance = choice("Work-life balance", LEVEL_0_3)

    st.subheader("Experience")
    c1, c2, c3 = st.columns(3)
    with c1:
        NumCompaniesWorked = st.number_input("Companies worked in (0-9)", 0, 9, 2, step=1)
        TotalWorkingYears = st.number_input("Total working years (0-39)", 0.0, 39.0, 8.0)
    with c2:
        YearsAtCompany = st.number_input("Years at company (0-36)", 0.0, 36.0, 4.0)
        YearsInCurrentRole = st.number_input("Years in current role (0-18)", 0.0, 18.0, 2.0)
    with c3:
        YearsSinceLastPromotion = st.number_input("Years since last promotion (0-15)", 0.0, 15.0, 1.0)
        YearsWithCurrManager = st.number_input("Years with current manager (0-17)", 0.0, 17.0, 2.0)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    row = {
        "BusinessTravel": BusinessTravel, "DailyRate": DailyRate, "Department": Department,
        "DistanceFromHome": DistanceFromHome, "Education": Education, "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction, "Gender": Gender, "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement, "JobLevel": JobLevel, "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction, "MaritalStatus": MaritalStatus, "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate, "NumCompaniesWorked": NumCompaniesWorked, "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike, "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction, "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears, "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance, "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole, "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager, "Age_Years": Age_Years,
    }
    new_df = pd.DataFrame([row])
    pred = int(model.predict(new_df)[0])

    st.divider()
    if pred == 0:
        st.success("Employee has **not** attrited (likely to stay).")
    else:
        st.error("Employee has **attrited** (likely to leave).")

    if hasattr(model, "predict_proba"):
        p_leave = float(model.predict_proba(new_df)[0][1])
        st.metric("Attrition probability", f"{p_leave:.1%}")
        st.progress(min(max(p_leave, 0.0), 1.0))

    with st.expander("Inputs sent to the model"):
        st.dataframe(new_df.T.rename(columns={0: "Value"}), use_container_width=True)
