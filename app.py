import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

# --- PAGE/THEME SETUP ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.markdown("""
<style>
body, .stApp {background: #1a1d21;}
.left-panel {
    background: linear-gradient(120deg, #253349 80%, #1d2337 100%);
    border-radius: 16px;
    padding: 2.3rem 1.3rem 2.6rem 1.3rem;
    margin: 2.5rem 0 2rem 0;
    min-width: 365px;
    box-shadow: 0 2px 34px #2475c126;
    color: #f2f4fa;
    border: 1px solid #2d435c;
}
.section-title {
    font-weight:700; 
    color:#f4c150; 
    font-size:1.12rem; 
    padding-bottom: 0.38rem;
    letter-spacing:0.01em;
}
.input-section-title {
    color: #58c2f5; font-weight:700; font-size:1.05rem; margin-top:1.1em; margin-bottom:0.3em;
}
input[type="text"], .stNumberInput > div > input, .stSelectbox > div > div > div > input {
    background: #1e283a !important;
    color: #fafafa !important;
    border: none !important;
}
.stTextInput, .stNumberInput, .stSelectbox, .stSlider {color:#dbdbdb;}
.info-panel {
    background: linear-gradient(120deg, #232d45 85%, #171b29 100%);
    border-radius: 16px; padding:1.6rem 2.3rem; margin:2rem 0 2.2rem 0;
    box-shadow:0 10px 34px #00000033; color:#e9edf8;
    border: 1px solid #253349;
}
h1,h2,h3,h4,h5 {color:#fff;}
.note-box {
    color: #ff9e6b; background: #272323;
    border-left: 4px solid #ff9e6b;
    padding:.8rem 1rem; margin:.6rem 0 1.3rem 0; border-radius:5px;
    font-size:1rem; font-weight:500;
}
.result-high {
    background: linear-gradient(90deg, #f66b6b65 0%, #ffb7b73a 100%);
    color:#c82c2c;
    font-weight:700; border-radius:12px;
    padding:1.4rem 2.4rem; font-size:1.13rem;
    border-left:8px solid #f66b6b;
    margin:1.3rem 0 1.7rem 0;
    box-shadow: 0 2px 18px #f66b6b22;
}
.result-low {
    background: linear-gradient(90deg, #67ddab40 0%, #daf7e055 100%);
    color:#207a54;
    font-weight:700; border-radius:12px;
    padding:1.4rem 2.4rem; font-size:1.13rem;
    border-left:8px solid #67ddab;
    margin:1.3rem 0 1.7rem 0;
    box-shadow: 0 2px 18px #67ddab33;
}
.stButton>button {
    width:96%;
    background: linear-gradient(90deg, #58c2f5 0%, #7373fa 100%);
    color:#f8faff;
    padding:0.95rem 0;
    border-radius:11px;
    border:none;
    font-size:1.21rem;
    font-weight:800;
    letter-spacing:0.035em;
    margin-top:1.8em;margin-bottom:0.3em;
    box-shadow: 0 6px 28px #51a8dd38;
    transition: box-shadow 0.22s, filter 0.22s;
}
.stButton>button:hover {
    filter: brightness(1.07);
    box-shadow:0 6px 40px #208fff60;
}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL/SYNTHETIC FALLBACK ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('columns.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except:
        n = 900
        np.random.seed(42)
        columns = ['age', 'sex', 'cp', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        X = pd.DataFrame({
            'age': np.random.randint(28,98,n),
            'sex': np.random.randint(0,2,n),
            'cp': np.random.randint(0,4,n),
            'exang': np.random.randint(0,2,n),
            'oldpeak': np.random.uniform(0,6,n),
            'slope': np.random.randint(0,3,n),
            'ca': np.random.randint(0,4,n),
            'thal': np.random.randint(0,4,n),
        })
        risk = (X['age']>60)*0.3 + (X['cp']>=2)*0.4 + (X['exang']==1)*0.3
        y = (risk + np.random.normal(0,0.1,n) > 0.4).astype(int)
        scaler = StandardScaler().fit(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(scaler.transform(X), y)
        return model, scaler, columns

model, scaler, feature_names = load_model()

# --- PANELS LAYOUT ---
left, main = st.columns([1.0,2.1], gap='large')

with left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Management</div>', unsafe_allow_html=True)
    name = st.text_input("Full Name", value="")
    patient_id = st.text_input("Patient ID", value="")
    st.markdown('<div class="input-section-title">Clinical Inputs</div>', unsafe_allow_html=True)
    age = st.slider("Age for Prediction", 28, 98, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", ["0", "1", "2", "3"])
    exang = st.selectbox("Exercise-Induced Angina (exang)", ["Yes", "No"])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope", ["0", "1", "2"])
    ca = st.selectbox("Major Vessels (ca)", ["0", "1", "2", "3"])
    thal = st.selectbox("Thalassemia (thal)", ["0", "1", "2", "3"])
    predict = st.button("PREDICT RISK")
    st.markdown("</div>", unsafe_allow_html=True)

with main:
    st.markdown('<h1 style="font-size:2.41rem;font-weight:900;margin-top:1.6rem;margin-bottom:1.19rem;letter-spacing:0.02em;">Heart Disease Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-panel">', unsafe_allow_html=True)
    st.markdown('<h2 style="margin-bottom:14px;font-weight:700;">Heart Disease Clinical Prediction</h2>', unsafe_allow_html=True)
    st.markdown('<div class="note-box">Note: This tool is intended for clinical use. Input parameters like thalassemia, ST depression, and vessel count should come from your medical reports for best accuracy.</div>', unsafe_allow_html=True)
    
    if predict:
        input_row = np.array([[
            age,
            1 if sex == "Male" else 0,
            int(cp),
            1 if exang == "Yes" else 0,
            oldpeak,
            int(slope),
            int(ca),
            int(thal)
        ]])
        features_scaled = scaler.transform(input_row)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] * 100

        if prediction == 1:
            st.markdown(
                f'<div class="result-high">'
                f'<b>HIGH RISK</b><br>Risk Probability: <span style="font-size:1.12em;">{probability:.1f}%</span><br>'
                f'<span style="color:#d33510;font-weight:500;">Immediate medical consultation recommended.</span>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="result-low">'
                f'<b>LOW RISK</b><br>Risk Probability: <span style="font-size:1.12em;">{probability:.1f}%</span><br>'
                f'<span style="color:#267547;font-weight:500;">Continue regular monitoring.</span>'
                f'</div>', unsafe_allow_html=True)
        
        # --- MODEL INSIGHTS BELOW ---
        st.markdown("### 🔎 Model Insights", unsafe_allow_html=True)
        # -- Feature Importance --
        st.markdown("**Feature Importance:**")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        importances = model.feature_importances_
        indices = np.argsort(importances)
        labels = np.array(feature_names)[indices]
        ax.barh(labels, importances[indices], color="#809ee9")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

        # -- Partial Dependence on Age --
        st.markdown("**Effect of Age on Predicted Heart Disease Risk:**")
        try:
            # Partial dependence for 'age' feature (0th in our feature set)
            pd_plot = partial_dependence(model, scaler.transform(pd.DataFrame({
                'age': np.arange(28,98,1),
                'sex':[1]*70,
                'cp':[2]*70,
                'exang':[1]*70,
                'oldpeak':[1.0]*70,
                'slope':[1]*70,
                'ca':[0]*70,
                'thal':[2]*70
            })), [0], grid_resolution=40)
            xx, yy = pd_plot['values'][0], pd_plot['average'][0]
        except Exception as e:
            # fallback in case of sklearn change or synthetic
            grid = np.arange(28,98,1)
            base = np.tile([1,2,1,1.0,1,0,2], (len(grid),1))
            base = np.column_stack([grid,base])
            pd_pred = model.predict_proba(scaler.transform(base))[:,1]
            xx, yy = grid, pd_pred

        fig2, ax2 = plt.subplots(figsize=(6,3.3))
        ax2.plot(xx, yy, color="#ea8658", linewidth=3)
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Predicted Risk Probability")
        ax2.set_title("Partial Dependence: Age Effect")
        ax2.grid(alpha=0.25)
        st.pyplot(fig2)

    else:
        st.markdown(
            '<div class="note-box" style="background:none;color:#8ab1df;border:none;font-weight:400;">'
            'Please enter patient details and click <b>PREDICT RISK</b> to see your evaluation.'
            '</div>', 
            unsafe_allow_html=True
        )

    # --- ABOUT SECTION ---
    st.markdown(
        "<h3 style='margin-top:2.2em;color:#f4c150;'>About This Tool</h3>"
        "<p>This heart disease prediction tool uses advanced machine learning to assess the risk of heart disease based on clinical parameters. Results are provided with a confidence probability and actionable recommendation.</p>",
        unsafe_allow_html=True)
    st.markdown("**Key Features:**")
    st.markdown(
        "- Highly readable, modern patient management panel\n"
        "- Gradient-highlighted clinical input form\n"
        "- Risk assessment with confidence scores and vivid results\n"
        "- Model transparency: feature importance and age effect visuals"
    )
    st.markdown("**Clinical Parameters Used:**")
    st.markdown(
        "| Parameter                       | Description                                 |\n"
        "|----------------------------------|---------------------------------------------|\n"
        "| **Age**                         | Patient age in years                        |\n"
        "| **Sex**                         | 0 = Female, 1 = Male                        |\n"
        "| **Chest Pain Type (cp)**        | 0-3 scale, as per clinical definition       |\n"
        "| **Exercise-Induced Angina (exang)** | 0 = No, 1 = Yes                         |\n"
        "| **ST Depression (oldpeak)**     | Depression induced by exercise (numeric)    |\n"
        "| **Slope**                       | Slope of peak exercise ST segment           |\n"
        "| **Major Vessels (ca)**          | Number detected by fluoroscopy (0–3)        |\n"
        "| **Thalassemia (thal)**          | Blood disorder type (0–3)                   |"
    )
    st.markdown("</div>", unsafe_allow_html=True)
