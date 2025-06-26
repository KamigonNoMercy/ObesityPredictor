import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ——— Inisialisasi Aplikasi ———
app = FastAPI(
    title="Obesity Prediction API",
    description="API untuk memprediksi tingkat obesitas berdasarkan gaya hidup",
    version="1.0.0"
)

# ——— Schema Input Data ———
class ObesityInput(BaseModel):
    Age: float
    Height: float
    Weight: float
    FCVC: float
    NCP: float
    CH2O: float
    FAF: float
    TUE: float
    Gender: str
    family_history_with_overweight: str
    FAVC: str
    SMOKE: str
    SCC: str
    CAEC: str
    CALC: str
    MTRANS: str

# ——— Load Model dan Metadata ———
with open("best_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]
label_encoder = saved["label_encoder"]
feature_names = saved["feature_names"]
numerical = saved["numerical"]

# ——— Endpoint Root ———
@app.get("/")
def root():
    return {"message": "Obesity Prediction API up and running"}

# ——— Endpoint Prediksi ———
@app.post("/predict")
def predict(data: ObesityInput):
    try:
        # — Step 1: Konversi ke DataFrame
        df = pd.DataFrame([data.dict()])

        # — Step 2: Normalisasi Kategori (seperti saat training)
        df["CAEC"] = df["CAEC"].replace({"Always": "Often", "Frequently": "Often", "no": "Never"})
        df["CALC"] = df["CALC"].replace({"Frequently": "Sometimes"})
        df["MTRANS"] = df["MTRANS"].replace({"Walking": "Other", "Bike": "Other", "Motorbike": "Other"})

        # — Step 3: One-hot Encoding dan Penyesuaian Kolom
        df = pd.get_dummies(df, drop_first=True)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        # — Step 4: Scaling untuk Fitur Numerik
        df[numerical] = scaler.transform(df[numerical])

        # — Step 5: Prediksi
        pred_idx = model.predict(df)[0]
        proba_arr = model.predict_proba(df)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        proba_dict = {
            label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba_arr)
        }

        return {
            "predicted_class": pred_label,
            "probabilities": proba_dict
        }

    except Exception as e:
        return {
            "error": f"Terjadi kesalahan saat prediksi: {str(e)}"
        }
