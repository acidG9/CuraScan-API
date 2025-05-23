from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load model and scaler
data = joblib.load("trained_models_and_scaler.joblib")
model: RandomForestClassifier = data["rf_model"]
scaler: StandardScaler = data["scaler"]
feature_names = scaler.feature_names_in_

class InputModel(BaseModel):
    Age: float
    CBC_Complete_Blood_Count: float = Field(..., alias="CBC (Complete Blood Count)")
    LFT_Liver_Function_Test: float = Field(..., alias="LFT (Liver Function Test)")
    RFT_Renal_Function_Test: float = Field(..., alias="RFT (Renal Function Test)")
    Blood_Glucose_Test: float = Field(..., alias="Blood Glucose Test")
    Lipid_Profile: float = Field(..., alias="Lipid Profile")
    Thyroid_Function_Test: float = Field(..., alias="Thyroid Function Test")
    HbA1c_Test: float = Field(..., alias="HbA1c Test")
    Blood_Culture: float = Field(..., alias="Blood Culture")
    X_ray: float = Field(..., alias="X-ray")
    MRI: float
    CT_Scan: float = Field(..., alias="CT Scan")
    Ultrasound: float
    PET_Scan: float = Field(..., alias="PET Scan")
    Mammography: float
    Bone_Density_Scan: float = Field(..., alias="Bone Density Scan")
    Skin_Biopsy: float = Field(..., alias="Skin Biopsy")
    Needle_Biopsy: float = Field(..., alias="Needle Biopsy")
    Surgical_Biopsy: float = Field(..., alias="Surgical Biopsy")
    Bone_Marrow_Biopsy: float = Field(..., alias="Bone Marrow Biopsy")
    Lymph_Node_Biopsy: float = Field(..., alias="Lymph Node Biopsy")
    Karyotyping: float
    BRCA_Gene_Test: float = Field(..., alias="BRCA Gene Test")
    Whole_Genome_Sequencing: float = Field(..., alias="Whole Genome Sequencing")
    Prenatal_Genetic_Screening: float = Field(..., alias="Prenatal Genetic Screening")
    Urinalysis: float
    Urine_Culture: float = Field(..., alias="Urine Culture")
    Urine_Cytology: float = Field(..., alias="Urine Cytology")
    Urine_24_Hour_Test: float = Field(..., alias="24-Hour Urine Test")
    Stool_Culture: float = Field(..., alias="Stool Culture")
    Occult_Blood_Test: float = Field(..., alias="Occult Blood Test")
    Ova_and_Parasite_Exam: float = Field(..., alias="Ova and Parasite Exam")
    Calprotectin_Test: float = Field(..., alias="Calprotectin Test")
    Patch_Test: float = Field(..., alias="Patch Test")
    Skin_Prick_Test: float = Field(..., alias="Skin Prick Test")
    Intradermal_Test: float = Field(..., alias="Intradermal Test")
    Mantoux_Test: float = Field(..., alias="Mantoux Test (TB Test)")
    Visual_Acuity_Test: float = Field(..., alias="Visual Acuity Test")
    Tonometry: float
    Retinal_Examination: float = Field(..., alias="Retinal Examination")
    Slit_Lamp_Examination: float = Field(..., alias="Slit-Lamp Examination")
    Audiometry: float
    Tympanometry: float
    Otoacoustic_Emissions_Test: float = Field(..., alias="Otoacoustic Emissions Test")
    Auditory_Brainstem_Response_ABR: float = Field(..., alias="Auditory Brainstem Response (ABR)")
    Blood_Group_Encoded: float = Field(..., alias="Blood Group Encoded")
    BP_Systolic: float
    BP_Diastolic: float
    LFT_RFT_Ratio: float

    class Config:
        allow_population_by_field_name = True

@app.get("/")
def root():
    return {"message": "✅ Server is running. Use POST /predict to get predictions."}

@app.post("/predict")
def predict(input_data: InputModel):
    original_input = input_data.dict(by_alias=True)
    df = pd.DataFrame([original_input], columns=feature_names)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    return {"prediction": int(prediction)}
