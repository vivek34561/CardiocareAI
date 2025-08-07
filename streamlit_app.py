import os
import io
import unicodedata
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from groq import Groq
from src.mlproject.predict_pipelines import PredictPipeline

# Load environment variables
load_dotenv()

st.set_page_config(page_title="🪀 Heart Risk & Diet AI", layout="wide")

# ------------------------- 🔑 API Key Input -------------------------
st.sidebar.header("🔑 Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")
    st.stop()

client = Groq(api_key=groq_api_key)

st.title("🪀 Heart Disease Predictor & Diet Assistant")

# ------------------------- Session State Init -------------------------
for key in ["predicted", "prediction", "diet_plan_text", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "predicted" else [] if key == "chat_history" else None

# ------------------------- Tabs -------------------------
tab1 = st.container()

with tab1:
    st.subheader("📋 Your Health Profile")

    with st.expander("🏠 Lifestyle & Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("🎂 Age", 20, 90, 45)
            sex = st.radio("♂️ Biological Sex", ["Male", "Female"])
        with col2:
            exang = st.radio("🏃 Chest pain during exercise?", ["No", "Yes"])
            fbs = st.radio("🍬 Fasting blood sugar > 120 mg/dL?", ["No", "Yes"])

    with st.expander("💓 Vitals & Tests", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            trestbps = st.slider("🩺 Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.slider("🧪 Cholesterol Level (mg/dL)", 100, 400, 220)
            thalach = st.slider("❤️ Max Heart Rate Achieved", 60, 210, 150)
        with col2:
            oldpeak = st.slider("📉 ST Depression (Exercise vs Rest)", 0.0, 6.0, 1.0, 0.1)
            restecg = st.selectbox("📈 ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
            slope = st.selectbox("📊 Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])

    with st.expander("🧬 Medical History", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            cp = st.selectbox("💓 Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        with col2:
            ca = st.selectbox("🦠 Number of Major Vessels Colored", [0, 1, 2, 3])
            thal = st.selectbox("🦬 Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    model_input = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
        "ca": ca,
        "thal": ["Normal", "Fixed Defect", "Reversible Defect"].index(thal),
    }

    predict_btn = st.button("🚑 Predict Risk")
    diet_btn = st.button("🥗 Generate Diet Plan")
    report_btn = st.button("🗾 Generate Risk Report")
    lifestyle_btn = st.button("🏃 Lifestyle Suggestions")
    doctor_note_btn = st.button("📄 Generate Doctor's Note")
    language = st.selectbox("🌐 Select Output Language", ["English", "Hindi", "Spanish", "Tamil", "Bengali"])

    def translate_text(text, target_language):
        if target_language == "English":
            return text
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": f"Translate this to {target_language}:\n{text}"}
            ]
        )
        return response.choices[0].message.content.strip()

    if predict_btn:
        pipeline = PredictPipeline()
        prediction = pipeline.predict(model_input)
        st.session_state["prediction"] = prediction
        st.session_state["predicted"] = True

    if st.session_state["predicted"]:
        st.markdown("---")
        if st.session_state["prediction"] == 1:
            st.error("⚠️ **High Risk of Heart Disease Detected!** Consult a cardiologist.")
        else:
            st.success("✅ **Low Risk of Heart Disease. Keep maintaining your health!**")

    if diet_btn:
        if not st.session_state["predicted"]:
            st.warning("⚠️ Please run the prediction first.")
        else:
            prompt = f"""
I’m a {age}-year-old {'male' if model_input['sex'] else 'female'} with:
BP: {trestbps}, Cholesterol: {chol}, Fasting Sugar: {'Yes' if model_input['fbs'] else 'No'}
Max HR: {thalach}, ST Depression: {oldpeak}, Thalassemia: {thal}
Create a heart-healthy diet plan including nutrients, foods to eat/avoid, and sample meals.
"""
            with st.spinner("🍎 Generating diet plan..."):
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a certified medical dietitian."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000
                )
                st.session_state["diet_plan_text"] = response.choices[0].message.content

    if st.session_state["diet_plan_text"]:
        formatted = st.session_state["diet_plan_text"].replace("\n", "<br>")
        st.markdown(f"<div style='background:#068f88;padding:15px;border-radius:10px'>{formatted}</div>", unsafe_allow_html=True)

        def clean_text(text):
            return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in clean_text(st.session_state["diet_plan_text"]).split('\n'):
            pdf.multi_cell(0, 10, line)
        st.download_button("📥 Download Diet Plan", io.BytesIO(pdf.output(dest="S").encode("latin1")), "diet_plan.pdf")

    if report_btn and st.session_state["predicted"]:
        with st.spinner("📊 Analyzing risk factors..."):
            prompt = f"""
You are a cardiologist. Explain why the patient was predicted {'high' if st.session_state['prediction'] else 'low'} risk.
Age: {age}, Sex: {'Male' if model_input['sex'] else 'Female'}, Chol: {chol}, BP: {trestbps}, HR: {thalach}, ST Depression: {oldpeak}, Angina: {exang}, Thal: {thal}
"""
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown("### 🗾 Risk Report")
            st.markdown(translate_text(response.choices[0].message.content.strip(), language))

    if lifestyle_btn and st.session_state["predicted"]:
        with st.spinner("🏃 Generating recommendations..."):
            prompt = f"""
Give daily lifestyle advice on diet, exercise, stress, and sleep for a patient with:
Age: {age}, Sex: {'Male' if model_input['sex'] else 'Female'}, BP: {trestbps}, Chol: {chol}, HR: {thalach}, ST Depression: {oldpeak}
"""
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown("### 🏃 Lifestyle Advice")
            st.markdown(translate_text(response.choices[0].message.content.strip(), language))

    if doctor_note_btn and st.session_state["predicted"]:
        with st.spinner("📄 Drafting summary..."):
            prompt = f"""
Draft a doctor's summary note from patient profile and risk status:
Age: {age}, Sex: {'Male' if model_input['sex'] else 'Female'}, Risk: {'High' if st.session_state['prediction'] else 'Low'},
BP: {trestbps}, Chol: {chol}, HR: {thalach}, ST Depression: {oldpeak}, Angina: {exang}, Thalassemia: {thal}, Vessels: {ca}
"""
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown("### 📄 Doctor's Note")
            st.markdown(translate_text(response.choices[0].message.content.strip(), language))

# ------------------------- Sidebar Chatbot -------------------------
with st.sidebar:
    st.header("💬 Diet & Medical Chatbot")
    user_input = st.text_input("❓ Ask anything")

    if user_input:
        with st.spinner("🧐 Thinking..."):
            messages = [
                {"role": "system", "content": "You are Healthy(B), a multilingual diet and heart health expert."},
                {"role": "user", "content": user_input}
            ]
            response = client.chat.completions.create(model="llama3-70b-8192", messages=messages, max_tokens=300)
            reply = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.markdown(translate_text(reply, language))

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 🪡 Chat History")
        for msg in reversed(st.session_state.chat_history):
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")
