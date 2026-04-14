# app.py

import pandas as pd
import numpy as np
import gradio as gr
import pickle
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer

# 1. Load model and scaler
model = load_model("lung_cancer_ann_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class_names = ["NO", "YES"]
feature_order = scaler.feature_names_in_

# 2. Load dataset for LIME background
df = pd.read_csv("survey_lung_cancer.csv")
df.columns = [c.strip() for c in df.columns]
df["GENDER"] = df["GENDER"].map({"FEMALE": 0, "MALE": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})
X = df.drop("LUNG_CANCER", axis=1)[feature_order]
X_scaled = scaler.transform(X)

explainer = LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=feature_order,
    class_names=class_names,
    mode="classification",
    discretize_continuous=False,
)

def model_predict_proba(x):
    probs = model.predict(x)
    return np.hstack([1 - probs, probs])

def predict_with_lime(
    age, gender, smoking, yellow_fingers, anxiety, peer_pressure,
    chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
    shortness_of_breath, swallowing_difficulty, chest_pain
):
    data = {
        "AGE": age,
        "GENDER": 1 if gender else 0,
        "SMOKING": 1 if smoking else 0,
        "YELLOW_FINGERS": 1 if yellow_fingers else 0,
        "ANXIETY": 1 if anxiety else 0,
        "PEER_PRESSURE": 1 if peer_pressure else 0,
        "CHRONIC DISEASE": 1 if chronic_disease else 0,
        "FATIGUE": 1 if fatigue else 0,
        "ALLERGY": 1 if allergy else 0,
        "WHEEZING": 1 if wheezing else 0,
        "ALCOHOL CONSUMING": 1 if alcohol else 0,
        "COUGHING": 1 if coughing else 0,
        "SHORTNESS OF BREATH": 1 if shortness_of_breath else 0,
        "SWALLOWING DIFFICULTY": 1 if swallowing_difficulty else 0,
        "CHEST PAIN": 1 if chest_pain else 0,
    }
    input_df = pd.DataFrame([data])[feature_order]
    scaled_input = scaler.transform(input_df)

    prob_yes = float(model.predict(scaled_input)[0][0])
    pred_label = "YES" if prob_yes >= 0.7 else "NO"
    confidence = prob_yes if pred_label == "YES" else 1 - prob_yes

    try:
        exp = explainer.explain_instance(scaled_input[0], model_predict_proba, num_features=10)
        explanation_text = "\n".join(f"{feat}: {weight:.3f}" for feat, weight in exp.as_list())
    except Exception as e:
        explanation_text = f"LIME explanation not available (error: {type(e).__name__})."

    return f"Prediction: {pred_label} (Confidence: {confidence:.2f})", explanation_text

inputs = [
    gr.Slider(20, 90, value=50, step=1, label="Age"),
    gr.Checkbox(label="Gender (Checked = MALE, Unchecked = FEMALE)"),
    gr.Checkbox(label="Smoking"),
    gr.Checkbox(label="Yellow Fingers"),
    gr.Checkbox(label="Anxiety"),
    gr.Checkbox(label="Peer Pressure"),
    gr.Checkbox(label="Chronic Disease"),
    gr.Checkbox(label="Fatigue"),
    gr.Checkbox(label="Allergy"),
    gr.Checkbox(label="Wheezing"),
    gr.Checkbox(label="Alcohol Consuming"),
    gr.Checkbox(label="Coughing"),
    gr.Checkbox(label="Shortness of Breath"),
    gr.Checkbox(label="Swallowing Difficulty"),
    gr.Checkbox(label="Chest Pain"),
]

outputs = [
    gr.Textbox(label="Prediction"),
    gr.Textbox(label="LIME Explanation (Top 10 features)"),
]

gr.Interface(
    fn=predict_with_lime,
    inputs=inputs,
    outputs=outputs,
    title="Lung Cancer Prediction with Explainability (LIME)",
    description="Check symptoms to predict lung cancer and understand which features contributed to the prediction."
).launch()



