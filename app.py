from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input columns
columns = [ 
    "CBC (Complete Blood Count)", "LFT (Liver Function Test)", "RFT (Renal Function Test)",
    "Blood Glucose Test", "Lipid Profile", "Thyroid Function Test", "HbA1c Test", "PSA Test",
    "X-ray", "MRI", "CT Scan", "Ultrasound", "PET Scan", "Mammography", "Bone Density Test",
    "BP", "Echocardiogram", "ECG", "Holter Monitor", "Cardiac Stress Test", "Angiography",
    "Pulmonary Function Test", "COVID-19 Test", "Mantoux Test (TB Test)", "Intradermal Test",
    "Visual Acuity Test", "Tonometry", "Retinal Examination", "Slit-Lamp Examination",
    "Audiometry", "Tympanometry", "Otoacoustic Emissions Test", "Auditory Brainstem Response (ABR)",
    "Blood Culture", "HIV Test", "Hepatitis Test", "Sputum Test",
    "Pap Smear", "Skin Biopsy", "Colonoscopy", "Endoscopy",
    "EEG", "EMG", "Urinalysis", "Stool Test", "Allergy Test", "Genetic Testing"
]

def preprocess(data):
    df = pd.DataFrame([data])
    replace_dict = {
        "Normal": 0, "Abnormal": 1,
        "Negative": 0, "Positive": 1
    }
    for col in df.columns:
        if col == "BP":
            bp = df.at[0, col]
            try:
                systolic, diastolic = bp.split('/')
                df[col] = (int(systolic) + int(diastolic)) / 2
            except:
                df[col] = 120
        elif col == "Visual Acuity Test":
            va = df.at[0, col]
            try:
                left, right = map(int, va.split('/'))
                df[col] = (left + right) / 2
            except:
                df[col] = 20
        elif col == "Tonometry":
            try:
                df[col] = int(str(df.at[0, col]).replace("mmHg", "").strip())
            except:
                df[col] = 15
        elif isinstance(df.at[0, col], str):
            df[col] = replace_dict.get(df.at[0, col], df.at[0, col])
    
    return df[columns]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        processed = preprocess(data)
        prediction = model.predict(processed)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is up and running!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
