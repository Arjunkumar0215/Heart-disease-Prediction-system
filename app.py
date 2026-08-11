import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.title{
    text-align:center;
    font-size:40px;
    color:#C62828;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:20px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.12);
    margin-bottom:20px;
}

.metric{
    background:#ffffff;
    border-radius:10px;
    padding:15px;
    text-align:center;
    box-shadow:0px 2px 8px rgba(0,0,0,0.10);
}

.stButton>button{

    width:100%;
    background:#C62828;
    color:white;
    border:none;
    padding:12px;
    border-radius:8px;
    font-size:18px;
    font-weight:bold;

}

.stButton>button:hover{

    background:#8E0000;
    color:white;

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown("<div class='title'>❤️ Heart Disease Prediction System</div>",
unsafe_allow_html=True)

st.markdown(
"<div class='subtitle'>AI Powered Heart Disease Risk Assessment</div>",
unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
    width=100
)

st.sidebar.title("Dashboard")

st.sidebar.info("""

This application predicts the
risk of heart disease using
Machine Learning.

Model :
✔ Logistic Regression

Feature Scaling :
✔ StandardScaler

""")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("---")

st.sidebar.subheader("Instructions")

st.sidebar.write("""

1. Enter patient details

2. Click Predict

3. View prediction

4. Download report

""")

# =====================================================
# DASHBOARD METRICS
# =====================================================

m1,m2,m3,m4 = st.columns(4)

with m1:
    st.metric("Model","Logistic Regression")

with m2:
    st.metric("Scaling","StandardScaler")

with m3:
    st.metric("Features","16")

with m4:
    st.metric("Target","Heart Disease")

st.markdown("---")

# =====================================================
# PATIENT INFORMATION
# =====================================================

st.markdown("## 👤 Patient Information")

col1,col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=22.5
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=60,
        max_value=180,
        value=120
    )

with col2:

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=80,
        max_value=400,
        value=180
    )

    glucose = st.number_input(
        "Glucose",
        min_value=50,
        max_value=400,
        value=100
    )

    smoking = st.selectbox(
        "Smoking",
        ["No","Yes"]
    )

    alcohol = st.selectbox(
        "Alcohol Consumption",
        ["No","Yes"]
    )

# =====================================================
# ENCODING
# =====================================================

gender = 1 if gender=="Male" else 0

smoking = 1 if smoking=="Yes" else 0

alcohol = 1 if alcohol=="Yes" else 0

# =====================================================
# LIFESTYLE DETAILS
# =====================================================

st.markdown("---")
st.markdown("## 🏃 Lifestyle Details")

col3, col4 = st.columns(2)

with col3:

    exercise_level = st.selectbox(
        "Exercise Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    diabetes = st.selectbox(
        "Diabetes",
        [
            "No",
            "Yes"
        ]
    )

with col4:

    stroke = st.selectbox(
        "Previous Stroke",
        [
            "No",
            "Yes"
        ]
    )

# =====================================================
# SYMPTOMS
# =====================================================

st.markdown("---")
st.markdown("## 🩺 Symptoms")

sym1, sym2, sym3 = st.columns(3)

with sym1:

    fatigue = st.selectbox(
        "Fatigue",
        [
            "No",
            "Yes"
        ]
    )

with sym2:

    chest_pain = st.selectbox(
        "Chest Pain",
        [
            "No",
            "Yes"
        ]
    )

with sym3:

    dizziness = st.selectbox(
        "Dizziness",
        [
            "No",
            "Yes"
        ]
    )

# =====================================================
# ENCODE CATEGORICAL FEATURES
# =====================================================

exercise_level = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}[exercise_level]

diabetes = 1 if diabetes == "Yes" else 0

stroke = 1 if stroke == "Yes" else 0

fatigue = 1 if fatigue == "Yes" else 0

chest_pain = 1 if chest_pain == "Yes" else 0

dizziness = 1 if dizziness == "Yes" else 0

# =====================================================
# AUTO FEATURE ENGINEERING
# =====================================================

cardio_risk = int(
    blood_pressure >= 140 or
    cholesterol >= 240
)

metabolic_syndrome = int(
    bmi >= 30 and
    blood_pressure >= 140 and
    glucose >= 126
)

# =====================================================
# HEALTH SUMMARY
# =====================================================

st.markdown("---")
st.markdown("## 📊 Calculated Health Indicators")

m1, m2 = st.columns(2)

with m1:

    if cardio_risk == 1:
        st.error("⚠️ Cardio Risk : High")
    else:
        st.success("✅ Cardio Risk : Normal")

with m2:

    if metabolic_syndrome == 1:
        st.warning("⚠️ Metabolic Syndrome : Present")
    else:
        st.success("✅ Metabolic Syndrome : Absent")

# =====================================================
# PREPARE INPUT DATA
# =====================================================

input_df = pd.DataFrame({

    "age":[age],

    "gender":[gender],

    "bmi":[bmi],

    "exercise_level":[exercise_level],

    "smoking":[smoking],

    "alcohol":[alcohol],

    "blood_pressure":[blood_pressure],

    "cholesterol":[cholesterol],

    "glucose":[glucose],

    "fatigue":[fatigue],

    "chest_pain":[chest_pain],

    "dizziness":[dizziness],

    "diabetes":[diabetes],

    "stroke":[stroke],

    "cardio_risk":[cardio_risk],

    "metabolic_syndrome":[metabolic_syndrome]

})

# =====================================================
# SHOW INPUT DATA
# =====================================================

with st.expander("📋 View Patient Data"):

    st.dataframe(
        input_df,
        use_container_width=True
    )

st.markdown("---")

predict = st.button(
    "❤️ Predict Heart Disease Risk"
)

# =====================================================
# PREDICTION
# =====================================================

if predict:

    # Scale Input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Prediction Probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.header("🩺 Prediction Result")

    # =============================================
    # RESULT
    # =============================================

    if prediction == 1:

        st.error("❤️ High Risk of Heart Disease")

    else:

        st.success("💚 Low Risk of Heart Disease")

    # =============================================
    # CONFIDENCE
    # =============================================

    st.subheader("Prediction Confidence")

    st.progress(float(probability))

    st.write(f"Probability of Heart Disease : **{probability:.2%}**")

    # =============================================
    # RISK LEVEL
    # =============================================

    st.subheader("Risk Level")

    if probability < 0.30:

        st.success("🟢 LOW RISK")

    elif probability < 0.70:

        st.warning("🟡 MODERATE RISK")

    else:

        st.error("🔴 HIGH RISK")

    # =============================================
    # HEALTH SCORE
    # =============================================

    health_score = int((1 - probability) * 100)

    st.metric(
        "Health Score",
        f"{health_score}/100"
    )

    # =============================================
    # PATIENT SUMMARY
    # =============================================

    st.markdown("---")
    st.subheader("📋 Patient Summary")

    c1, c2 = st.columns(2)

    with c1:

        st.write(f"**Age :** {age}")

        st.write(f"**BMI :** {bmi}")

        st.write(f"**Blood Pressure :** {blood_pressure}")

        st.write(f"**Cholesterol :** {cholesterol}")

        st.write(f"**Glucose :** {glucose}")

    with c2:

        st.write(f"**Smoking :** {'Yes' if smoking else 'No'}")

        st.write(f"**Alcohol :** {'Yes' if alcohol else 'No'}")

        st.write(
            f"**Exercise Level :** "
            f"{['Low','Medium','High'][exercise_level-1]}"
        )

        st.write(f"**Diabetes :** {'Yes' if diabetes else 'No'}")

        st.write(f"**Stroke :** {'Yes' if stroke else 'No'}")

    # =============================================
    # HEALTH RECOMMENDATIONS
    # =============================================

    st.markdown("---")
    st.subheader("💡 Health Recommendations")

    recommendations = []

    if bmi >= 25:
        recommendations.append(
            "🏃 Reduce body weight through a balanced diet and regular exercise."
        )

    if smoking == 1:
        recommendations.append(
            "🚭 Quit smoking to significantly reduce cardiovascular risk."
        )

    if alcohol == 1:
        recommendations.append(
            "🍺 Limit alcohol consumption."
        )

    if blood_pressure >= 140:
        recommendations.append(
            "🩺 Monitor and control your blood pressure."
        )

    if cholesterol >= 240:
        recommendations.append(
            "🥗 Reduce cholesterol with a healthy diet and consult a doctor."
        )

    if glucose >= 126:
        recommendations.append(
            "🩸 Monitor blood sugar regularly."
        )

    if exercise_level == 1:
        recommendations.append(
            "🏃 Increase physical activity to at least 30 minutes daily."
        )

    if fatigue:
        recommendations.append(
            "😴 Persistent fatigue should be evaluated by a physician."
        )

    if chest_pain:
        recommendations.append(
            "🚑 Chest pain requires immediate medical evaluation."
        )

    if dizziness:
        recommendations.append(
            "🩺 Frequent dizziness should not be ignored."
        )

    if diabetes:
        recommendations.append(
            "💊 Maintain good diabetes control."
        )

    if stroke:
        recommendations.append(
            "🧠 Continue regular neurological and cardiac follow-up."
        )

    if len(recommendations) == 0:

        st.success(
            "🎉 Excellent! No major lifestyle risk factors detected. Continue maintaining healthy habits."
        )

    else:

        for rec in recommendations:
            st.write(rec)

    # =============================================
    # WARNING
    # =============================================

    st.info(
        "⚠️ This prediction is generated by a Machine Learning model and should not replace professional medical advice."
    )

    # =====================================================
    # DASHBOARD VISUALIZATION
    # =====================================================

    if predict:

        st.markdown("---")
        st.header("📊 Health Dashboard")

        chart1, chart2 = st.columns(2)

        with chart1:

            bmi_chart = pd.DataFrame({
                "Metric": ["BMI"],
                "Value": [bmi]
            })

            st.bar_chart(
                bmi_chart.set_index("Metric")
            )

        with chart2:

            bp_chart = pd.DataFrame({
                "Metric": ["Blood Pressure"],
                "Value": [blood_pressure]
            })

            st.bar_chart(
                bp_chart.set_index("Metric")
            )

        chart3, chart4 = st.columns(2)

        with chart3:

            chol_chart = pd.DataFrame({
                "Metric": ["Cholesterol"],
                "Value": [cholesterol]
            })

            st.bar_chart(
                chol_chart.set_index("Metric")
            )

        with chart4:

            glucose_chart = pd.DataFrame({
                "Metric": ["Glucose"],
                "Value": [glucose]
            })

            st.bar_chart(
                glucose_chart.set_index("Metric")
            )

        # =====================================================
        # HEALTH INDICATORS
        # =====================================================

        st.markdown("---")
        st.header("🩺 Health Indicators")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "BMI",
                bmi
            )

        with c2:
            st.metric(
                "Blood Pressure",
                blood_pressure
            )

        with c3:
            st.metric(
                "Cholesterol",
                cholesterol
            )

        with c4:
            st.metric(
                "Glucose",
                glucose
            )

        # =====================================================
        # RISK FACTORS
        # =====================================================

        st.markdown("---")
        st.subheader("⚠ Risk Factors")

        risk_data = pd.DataFrame({

            "Risk Factor": [
                "Smoking",
                "Alcohol",
                "Diabetes",
                "Stroke",
                "Fatigue",
                "Chest Pain",
                "Dizziness",
                "Cardio Risk",
                "Metabolic Syndrome"
            ],

            "Present": [
                smoking,
                alcohol,
                diabetes,
                stroke,
                fatigue,
                chest_pain,
                dizziness,
                cardio_risk,
                metabolic_syndrome
            ]

        })

        st.dataframe(
            risk_data,
            use_container_width=True
        )

        # =====================================================
        # REPORT
        # =====================================================

        st.markdown("---")
        st.header("📄 Patient Report")

        report = pd.DataFrame({

            "Feature": [
                "Age",
                "Gender",
                "BMI",
                "Exercise Level",
                "Smoking",
                "Alcohol",
                "Blood Pressure",
                "Cholesterol",
                "Glucose",
                "Fatigue",
                "Chest Pain",
                "Dizziness",
                "Diabetes",
                "Stroke",
                "Cardio Risk",
                "Metabolic Syndrome",
                "Prediction",
                "Probability"
            ],

            "Value": [
                age,
                "Male" if gender == 1 else "Female",
                bmi,
                ["Low", "Medium", "High"][exercise_level - 1],
                "Yes" if smoking else "No",
                "Yes" if alcohol else "No",
                blood_pressure,
                cholesterol,
                glucose,
                "Yes" if fatigue else "No",
                "Yes" if chest_pain else "No",
                "Yes" if dizziness else "No",
                "Yes" if diabetes else "No",
                "Yes" if stroke else "No",
                cardio_risk,
                metabolic_syndrome,
                "Heart Disease" if prediction == 1 else "Healthy",
                f"{probability:.2%}"
            ]

        })

        st.dataframe(
            report,
            use_container_width=True
        )

        # =====================================================
        # DOWNLOAD REPORT
        # =====================================================

        csv = report.to_csv(index=False)

        st.download_button(

            label="⬇ Download Patient Report",

            data=csv,

            file_name="Heart_Disease_Report.csv",

            mime="text/csv"

        )

        # =====================================================
        # FINAL MESSAGE
        # =====================================================

        st.markdown("---")

        if prediction == 1:

            st.error(
                """
    ### 🚨 High Risk Detected

    Please consult a qualified cardiologist as soon as possible.

    Early diagnosis and treatment can significantly reduce complications.
    """
            )

        else:

            st.success(
                """
    ### 🎉 Great News!

    Your predicted risk of heart disease is low.

    Continue maintaining a healthy lifestyle with regular exercise,
    a balanced diet, and routine health checkups.
    """
            )

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("---")

    st.markdown(
        """
        <div style='text-align:center; color:gray;'>
    
        ❤️ Heart Disease Prediction System
    
        Developed using
    
        <b>Python | Streamlit | Scikit-Learn | Logistic Regression</b>
    
        <hr>
    
        This application is intended only for educational purposes and
        should not replace professional medical advice.
    
        </div>
        """,
        unsafe_allow_html=True
    )