from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# All columns from your list
all_columns = [
    'ID', 'Name', 'Age', 'Blood Group', 'BP', 'CBC (Complete Blood Count)',
    'LFT (Liver Function Test)', 'RFT (Renal Function Test)', 'Blood Glucose Test',
    'Lipid Profile', 'Thyroid Function Test', 'HbA1c Test', 'Blood Culture',
    'X-ray', 'MRI', 'CT Scan', 'Ultrasound', 'PET Scan', 'Mammography',
    'Bone Density Scan', 'Skin Biopsy', 'Needle Biopsy', 'Surgical Biopsy',
    'Bone Marrow Biopsy', 'Lymph Node Biopsy', 'Karyotyping', 'BRCA Gene Test',
    'Whole Genome Sequencing', 'Prenatal Genetic Screening', 'Urinalysis',
    'Urine Culture', 'Urine Cytology', '24-Hour Urine Test', 'Stool Culture',
    'Occult Blood Test', 'Ova and Parasite Exam', 'Calprotectin Test', 'Patch Test',
    'Skin Prick Test', 'Intradermal Test', 'Mantoux Test (TB Test)', 'Visual Acuity Test',
    'Tonometry', 'Retinal Examination', 'Slit-Lamp Examination', 'Audiometry',
    'Tympanometry', 'Otoacoustic Emissions Test', 'Auditory Brainstem Response (ABR)',
    'Disease Present'
]

# Features to use for prediction (exclude ID, Name, Disease Present)
feature_columns = [col for col in all_columns if col not in ['ID', 'Name', 'Disease Present']]

def encode_blood_group(bg):
    mapping = {'O-': 0, 'O+': 1, 'A-': 2, 'A+': 3, 'B-': 4, 'B+': 5, 'AB-': 6, 'AB+': 7}
    return mapping.get(bg, -1)

replace_dict = {
    "Normal": 0, "Abnormal": 1,
    "Negative": 0, "Positive": 1
}

def preprocess(data):
    df = pd.DataFrame([data])

    # Handle special columns
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
        elif col == "Age":
            try:
                df[col] = int(df.at[0, col])
            except:
                df[col] = 30
        elif col == "Blood Group":
            df[col] = encode_blood_group(df.at[0, col])
        elif isinstance(df.at[0, col], str):
            df[col] = replace_dict.get(df.at[0, col], df.at[0, col])

    # Fill missing feature columns with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only features for prediction
    return df[feature_columns]

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
