# 🧪 PredictLab – AI-Powered Disease Prediction Platform

🚀 Live App: https://huggingface.co/spaces/VINAYAKKV/Predict-Ai

PredictLab is a full-stack Machine Learning web application that delivers real-time health risk predictions using trained ML models and an AI assistant.

---

## 🚀 Features

- 🧠 Multiple ML Models
  - Diabetes Prediction
  - Stroke Risk Assessment
  - Heart Disease Detection

- 🤖 AI Chatbot (Llama 3 via Groq)
  - Explains predictions in simple language
  - Answers user queries
  - Provides contextual insights

- 🎨 Modern UI
  - Dark Glassmorphism design
  - Responsive layout
  - Smooth UX

- ⚡ Fast Backend
  - Built with FastAPI
  - High-performance APIs

---

## 🏗️ Tech Stack

- Python, FastAPI
- Scikit-learn, NumPy, Pandas
- HTML, Tailwind CSS, JavaScript
- Groq API (Llama 3)
- Hugging Face Spaces (Deployment)

---

## 📊 Models

| Model        | Purpose |
|-------------|--------|
| Diabetes     | Predicts diabetes risk |
| Stroke       | Estimates stroke probability |
| Heart        | Detects heart disease |

---

## ⚙️ Run Locally

```bash
git clone https://github.com/vinayak533/PredictLab.git
cd PredictLab
pip install -r requirements.txt
uvicorn main:app --reload
