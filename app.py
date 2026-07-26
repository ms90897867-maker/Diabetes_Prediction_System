import numpy as np
import gradio as gr
import joblib

# Load model and scaler
model = joblib.load("diabetes.pkl")
scaler = joblib.load("scaler.pkl")


# Prediction Function
def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age
):

    # Create input array
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]])

    # Scale input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "Diabetic"
    else:
        return "Non-Diabetic"


# Gradio Interface
diabetes_app = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],

    outputs=gr.Textbox(label="Prediction"),

    title="Diabetes Prediction System",

    description="Predict whether a person is Diabetic or Non-Diabetic."
)

diabetes_app.launch()
