import streamlit as st
import joblib
import shap
import pandas as pd

model = joblib.load("best_model.pkl")
X_test = pd.read_csv("sample_test_data.csv")  # Replace with your data

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

st.title("Fraud Detection - SHAP Model Explainability")

selected_index = st.slider("Choose a test sample index:", 0, len(X_test) - 1, 0)
shap.initjs()
st_shap = shap.force_plot(explainer.expected_value[1], shap_values[1][selected_index], X_test.iloc[selected_index])
st.components.v1.html(shap.save_html(st_shap), height=400)
