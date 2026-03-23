from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Any
from groq import Groq
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Internal model loader
from models.loader import load_models

# Load environment vars
load_dotenv()

app = FastAPI(title="PredictLab ML Portfolio")
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Failed to initialize Groq client: {e}")
    groq_client = None

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load models safely
models, scalers = load_models()

# Route: Homepage
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route: Diabetes Predictor
@app.get("/diabetes", response_class=HTMLResponse)
async def read_diabetes(request: Request):
    return templates.TemplateResponse("diabetes.html", {"request": request})

# Route: Stroke Predictor
@app.get("/stroke", response_class=HTMLResponse)
async def read_stroke(request: Request):
    return templates.TemplateResponse("stroke.html", {"request": request})

# Route: Heart Predictor
@app.get("/heart", response_class=HTMLResponse)
async def read_heart(request: Request):
    return templates.TemplateResponse("heart.html", {"request": request})

# Prediction Endpoints
@app.post("/predict/diabetes")
async def predict_diabetes(data: dict):
    if not models.get("diabetes"):
        return JSONResponse(status_code=500, content={"error": "Diabetes model not loaded yet. Coming soon!"})
    try:
        # Construct DataFrame with exactly the same column names the scaler/model expects
        df = pd.DataFrame([{
            "Pregnancies": float(data["pregnancies"]),
            "Glucose": float(data["glucose"]),
            "BloodPressure": float(data["blood_pressure"]),
            "SkinThickness": float(data["skin_thickness"]),
            "Insulin": float(data["insulin"]),
            "BMI": float(data["bmi"]),
            "DiabetesPedigreeFunction": float(data["dpf"]),
            "Age": float(data["age"])
        }])
        
        arr = df
        if scalers.get("diabetes"):
            arr = scalers["diabetes"].transform(arr)
            
        prediction = models["diabetes"].predict(arr)[0]
        label = "Diabetic" if prediction == 1 else "Not Diabetic"
        return {"prediction": int(prediction), "label": label, "confidence": None}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict/stroke")
async def predict_stroke(data: dict):
    if not models.get("stroke"):
        return JSONResponse(status_code=500, content={"error": "Stroke risk model not loaded yet. Coming soon!"})
    try:
        df = pd.DataFrame([{
            "gender": str(data["gender"]),
            "age": float(data["age"]),
            "hypertension": int(data["hypertension"]),
            "heart_disease": int(data["heart_disease"]),
            "ever_married": str(data["ever_married"]),
            "work_type": str(data["work_type"]),
            "Residence_type": str(data["residence_type"]),
            "avg_glucose_level": float(data["avg_glucose_level"]),
            "bmi": float(data["bmi"]),
            "smoking_status": str(data["smoking_status"])
        }])
        
        prediction = models["stroke"].predict(df)[0]
        label = "High Stroke Risk" if prediction == 1 else "Low Stroke Risk"
        return {"prediction": int(prediction), "label": label, "confidence": None}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict/heart")
async def predict_heart(data: dict):
    if not models.get("heart"):
        return JSONResponse(status_code=500, content={"error": "Heart disease model not loaded yet. Coming soon!"})
    try:
        features = [
            float(data["age"]),
            int(data["sex"]),
            int(data["cp"]),
            float(data["trestbps"]),
            float(data["chol"]),
            int(data["fbs"]),
            int(data["restecg"]),
            float(data["thalach"]),
            int(data["exang"]),
            float(data["oldpeak"]),
            int(data["ca"]),
            int(data["thal"])
        ]
        
        arr = np.array(features).reshape(1, -1)
        if scalers.get("heart"):
            arr = scalers["heart"].transform(arr)
        prediction = models["heart"].predict(arr)[0]
        label = "Disease Detected" if prediction == 1 else "No Disease Detected"
        return {"prediction": int(prediction), "label": label}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

# Groq Context Data Setup
class ChatContext(BaseModel):
    current_page: str
    inputs: Optional[dict] = {}
    prediction: Optional[Any] = None

class ChatPayload(BaseModel):
    message: str
    context: ChatContext

# Route: Chatbot
@app.post("/chat")
async def chat(payload: ChatPayload):
    if groq_client is None:
        return JSONResponse(status_code=500, content={"error": "Groq client not configured or missing API key."})
    
    try:
        system_prompt = f"""You are an intelligent assistant embedded in a machine learning portfolio app. You help users understand their prediction results.

Current context:
- Page: {payload.context.current_page}
- User inputs: {payload.context.inputs}
- Model prediction: {payload.context.prediction}

Your responsibilities:
1. Explain the prediction result in plain, simple language
2. Answer questions about the medical condition or real estate topic
3. Clarify what each input field means and how it affects the result
4. Provide helpful, accurate domain information
5. Always recommend consulting a licensed professional for medical/financial decisions
6. Be warm, concise, and non-alarmist

Keep all responses under 130 words unless the user asks for more detail.
Never provide a definitive medical diagnosis. Always encourage professional consultation."""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.message}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
