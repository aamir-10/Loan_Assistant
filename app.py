import streamlit as st
import pandas as pd
import joblib
import lime
import lime.lime_tabular
import os
from chatbot_utils import get_relevant_chunks, generate_answer_with_prompt
import streamlit as st


st.set_page_config(page_title="Loan Assistant")

# Load Model & Data

loan_model = joblib.load("loan_model.pkl")
X_train = pd.read_csv("X_train.csv")

# Page Setup

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        padding: 2rem;
        border-radius: 10px;
    }
    .stForm{
        border-radius: 6px !important;
        border: 1px solid #ced4da !important;
        padding: 1.5rem;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 188, 212, 0.1);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    section[data-testid="stSidebar"] {
    background: rgba(0, 188, 212, 0.1);  /* light to medium cyan */
    box-shadow: 2px 0 6px rgba(0, 0, 0, 0.4);
    color: white;
}

section[data-testid="stSidebar"] .css-1v0mbdj,  /* Optional: inner container text color */
section[data-testid="stSidebar"] .css-1cpxqw2 {
    color: white !important;
}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div style='
    border: 2px solid #00bcd4;
    border-radius: 10px;
    padding: 15px 20px;
    text-align: center;
    background-color: rgba(0, 188, 212, 0.1);
    margin-bottom: 20px;
'>
<h1 style='
    color: #00e5ff;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
'>
Loan Approval Assistant
</h1>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<p style='text-align: center; font-style: italic; color: white;'>
Fill in your loan details to check approval chances and ask personalized questions.
</p>
""", unsafe_allow_html=True)


# Sidebar

with st.sidebar:
    st.title("About")
    st.markdown("*Your very own Loan Chatbot!*")
    st.info("""
        This app predicts whether your loan will be approved using a trained ML model, and provides chatbot explanations based on:
        - Your loan details
        - A knowledge base built from past approvals""")
    st.markdown("---")
    st.markdown("~ Input your data to get started!")


# Input Form

st.markdown("""
<h2 style='text-align: center; font-size: 1.5rem; color: white;'>
Enter Your Loan Details
</h2>
""", unsafe_allow_html=True)

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
        gender = st.radio("Gender", ["Male", "Female"], index=None)
        self_employed = st.radio("Self Employed", ["Yes", "No"], index=None)

    with col2:
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_term = st.slider("Loan Term (months)", 12, 480, value=None)
        credit_history = st.selectbox("Credit History", [1.0, 0.0], index=0)
        married = st.radio("Married", ["Yes", "No"], index=None)
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"], index=0)

    c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 1, 1])
    with c3:
        submitted = st.form_submit_button("Predict Loan Approval")


# Prediction

if submitted:
    if (
        gender is None or
        married is None or
        self_employed is None or
        education == "" or
        dependents == "" or
        property_area == "" or
        credit_history is None or
        loan_term is None
    ):
        st.warning("🚫 Please make sure **all fields** are filled before predicting.")
    else:
        input_dict = {
            "Gender": 1 if gender == "Male" else 0,
            "Married": 1 if married == "Yes" else 0,
            "Dependents": int(dependents.replace("3+", "3")),
            "Education": 1 if education == "Graduate" else 0,
            "Self_Employed": 1 if self_employed == "Yes" else 0,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area_Rural": 1 if property_area == "Rural" else 0,
            "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
            "Property_Area_Urban": 1 if property_area == "Urban" else 0,
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

        prediction = loan_model.predict(input_df)[0]
        label = "Approved" if prediction == 1 else "Rejected"

        if label == "Approved":
            st.success(f"✅ Your loan is likely to be **{label}**.")
        else:
            st.markdown(
                f"<div style='background-color: rgba(244, 67, 54, 0.15); color: white; padding: 16px; border-radius: 5px; font-weight: 500; margin-top: 10px;'>❌ Your loan is likely to be <strong>{label}</strong>.</div>",
                unsafe_allow_html=True
            )

        st.markdown("""
<h2 style='text-align: center; font-size: 1.5rem; color: white;'>
Model Explanation with LIME
</h2>
""", unsafe_allow_html=True)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=["Rejected", "Approved"],
            mode="classification"
        )

        exp = explainer.explain_instance(
            data_row=input_df.values[0],
            predict_fn=loan_model.predict_proba,
            num_features=10
        )

        lime_html_path = "lime_explanation.html"
        exp.save_to_file(lime_html_path)

        with open(lime_html_path, 'r', encoding='utf-8') as f:
            lime_html = f.read()

        lime_html = lime_html.replace(
            "<body>",
            "<body style='background-color: white; color: black; padding: 20px;'>"
        )

        st.components.v1.html(lime_html, height=600, scrolling=True)

        st.session_state["user_input"] = input_dict
        st.session_state["prediction"] = label
        st.session_state["chat_history"] = []


# Explanation for chatbot context

def generate_outcome_explanation(user_input, label):
    reasons = []

    if label == "Rejected":
        if user_input['Credit_History'] == 0:
            reasons.append("You have no credit history, which is a strong reason for rejection.")
        if user_input['ApplicantIncome'] < 3000:
            reasons.append("Your income is below the minimum approval threshold of ₹3000.")
        if user_input['CoapplicantIncome'] == 0:
            reasons.append("You don’t have a coapplicant, which limits total household income.")
        if user_input['Self_Employed'] == 1 and user_input['ApplicantIncome'] < 4000:
            reasons.append("Self-employed applicants with lower income are seen as higher risk.")
        if not reasons:
            reasons.append("Based on our model, your profile did not meet the approval criteria.")
    else:
        if user_input['Credit_History'] == 1:
            reasons.append("Your good credit history greatly improved your chances.")
        if user_input['ApplicantIncome'] >= 3000:
            reasons.append("Your income met the minimum threshold.")
        if user_input['CoapplicantIncome'] >= 2000:
            reasons.append("Your coapplicant's income strengthened your application.")
        if user_input['Property_Area_Semiurban'] == 1:
            reasons.append("Applicants from semiurban areas had higher approval rates.")
        if not reasons:
            reasons.append("Your overall profile met the criteria for approval.")

    return "\n".join(reasons)
st.markdown("---")


st.markdown("""
<h2 style='text-align: center; font-size: 1.5rem; color: white;'>
Feature Importance Display
</h2>
""", unsafe_allow_html=True)


st.image("feature_importance.png")

st.markdown("---")
st.markdown("""
<div style="
    display: flex;
    justify-content: center;
">
  <div style="
      background-color: #1e293b;
      padding: 14px 28px;
      border-radius: 3px;
      border: 1px solid #00bcd4;
      font-size: 16.5px;
      font-style: italic;
      color: #e0f7fa;
      box-shadow: 0 4px 12px rgba(0, 188, 212, 0.15);
      transition: transform 0.3s ease;
  ">
    <span style="letter-spacing: 0.5px;">Scroll up to know your loan status!</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Chatbot Section

st.markdown("---")
st.markdown("""
<style>
.chatbot-header {
    display: inline-block;
    text-align: center;
    font-size: 1.6rem;
    color: cyan;
    padding: 15px 25px;
    border-radius: 12px;
    background-color: rgba(0, 188, 212, 0.1);
    transition: all 0.3s ease-in-out;
    cursor: pointer;
    margin-bottom: 25px;
}

.chatbot-header:hover {
    background-color: rgba(0, 188, 212, 0.3);
    color: #ffffff;
    font-size: 1.65rem;        
}
</style>

<div style='text-align: center;'>
    <div class='chatbot-header'>
        🤖 Ask the Chatbot About Your Loan
    </div>
</div>
""", unsafe_allow_html=True)


query = st.chat_input("Ask a question...")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if query:
    kb_chunks = get_relevant_chunks(query, k=5)
    kb_context = "\n".join(kb_chunks)

    user_context = ""
    if "user_input" in st.session_state and "prediction" in st.session_state:
        explanation = generate_outcome_explanation(
            st.session_state["user_input"], st.session_state["prediction"]
        )
        user_context = f"Prediction Outcome: {st.session_state['prediction']}\nExplanation: {explanation}"

    prompt = f"""
You are a helpful AI assistant that explains loan approval decisions.

You have two types of context:
1. General knowledge from past loan approvals
2. User's personal loan input

Your task is to answer the user's question using both, with a focus on the user's unique case.

----
General Knowledge:
{kb_context}

----
User Info:
{user_context}

----
Question: {query}

Answer in full sentences with explanations relevant to this user's profile. Be clear and helpful.
"""

    with st.spinner("Generating answer..."):
        lower_query = query.lower().strip()

        if "why was my loan rejected" in lower_query or "why was it rejected" in lower_query:
            if st.session_state.get("prediction") == "Rejected":
                response = f"Your loan was rejected. Here's why:\n\n{generate_outcome_explanation(st.session_state['user_input'], 'Rejected')}"
            else:
                response = "Your loan was not rejected."
        elif "why was my loan approved" in lower_query or "why was it approved" in lower_query:
            if st.session_state.get("prediction") == "Approved":
                response = f"Your loan was approved. Here's why:\n\n{generate_outcome_explanation(st.session_state['user_input'], 'Approved')}"
            else:
                response = "Your loan was not approved."
        else:
            response = generate_answer_with_prompt(prompt)

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("assistant", response))

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

if st.session_state.chat_history:
    chat_log = ""
    for role, msg in st.session_state.chat_history:
        name = "You" if role == "user" else "Assistant"
        chat_log += f"{name}: {msg}\n\n"

    st.download_button(
        label="⬇️ Download Chat History",
        data=chat_log.encode("utf-8"),
        file_name="loan_chat_history.txt",
        mime="text/plain"
    )
