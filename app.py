from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
model_data = joblib.load("trained_models_and_scaler.joblib")
rf_model = model_data['rf_model']
lr_model = model_data['lr_model']
scaler = model_data['scaler']

# Initialize FastAPI app
app = FastAPI()

# Health check route
@app.get("/")
def read_root():
    return {"message": "API is running ✅"}

# Define input schema
class PatientData(BaseModel):
    Age: float
    CBC: float
    LFT: float
    RFT: float
    Blood_Glucose_Test: float
    Lipid_Profile: float
    Thyroid_Function_Test: float
    HbA1c_Test: float
    Blood_Culture: float
    X_ray: float
    MRI: float
    CT_Scan: float
    Ultrasound: float
    PET_Scan: float
    Mammography: float
    Bone_Density_Scan: float
    Skin_Biopsy: float
    Needle_Biopsy: float
    Surgical_Biopsy: float
    Bone_Marrow_Biopsy: float
    Lymph_Node_Biopsy: float
    Karyotyping: float
    BRCA_Gene_Test: float
    Whole_Genome_Sequencing: float
    Prenatal_Genetic_Screening: float
    Urinalysis: float
    Urine_Culture: float
    Urine_Cytology: float
    Urine_24_Hour: float
    Stool_Culture: float
    Occult_Blood_Test: float
    Ova_and_Parasite_Exam: float
    Calprotectin_Test: float
    Patch_Test: float
    Skin_Prick_Test: float
    Intradermal_Test: float
    Mantoux_Test: float
    Visual_Acuity_Test: float
    Tonometry: float
    Retinal_Exam: float
    Slit_Lamp_Exam: float
    Audiometry: float
    Tympanometry: float
    Otoacoustic_Emissions_Test: float
    ABR_Test: float
    Blood_Group_Encoded: float
    BP_Systolic: float
    BP_Diastolic: float
    LFT_RFT_Ratio: float

# Prediction route
@app.post("/predict")
def predict_disease(data: PatientData):
    try:
        features = np.array([list(data.dict().values())])
        scaled_features = scaler.transform(features)
        prediction = rf_model.predict(scaled_features)
        return {"predicted_disease": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
