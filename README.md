🚀 PredictLab — AI-Powered Disease Prediction Platform

📌 Summary
PredictLab is a full-stack AI-powered health platform that delivers **real-time disease and stroke risk predictions**. It integrates multiple trained ML models with a **context-aware AI chatbot** to provide users actionable insights in plain language. Users can evaluate risks for **Diabetes, Stroke, and Heart Disease** through an interactive and modern web interface, all without needing technical expertise.

🌐 **Live Demo:** [PredictLab on Hugging Face Spaces](https://huggingface.co/spaces/VINAYAKKV/Predict-Ai)


---

🛠️ Technologies Used

**Backend**  
- Python  
- FastAPI  
- Pandas  
- Scikit-learn  

**AI Chatbot**  
- Groq API  
- Llama 3.1-8B Instant  

**Frontend**  
- HTML, CSS, JavaScript (Vanilla, fetch-based SPA)  
- Jinja2 templates  

**Other / Deployment**  
- Docker (port 7860)  
- Joblib / Pickle for model serialization  
- Hugging Face Spaces for hosting  

---

✨ Features
- Predicts risk for Diabetes, Stroke, and Heart Disease using trained ML models  
- Context-aware AI chatbot explains predictions and answers user questions  
- Interactive UI with **Dark Glassmorphism design** and smooth animations  
- No page reloads (SPA-like experience using JavaScript fetch)  
- Graceful model fallback: missing models only affect specific predictions  
- Dockerized for easy deployment and scaling  
- Modern responsive design for desktop and mobile  

---

⌨️ Keyboard Shortcuts / Commands
- `Ctrl + C` → Stop backend server  
- `Enter` → Execute command in terminal (FastAPI or Docker)  

---

⚙️ Process
1. User opens web app and selects a disease module  
2. Inputs health data (e.g., age, glucose, BMI, blood pressure)  
3. ML model processes inputs and returns **risk prediction**  
4. AI chatbot explains prediction in simple, non-alarmist language  
5. Results are displayed instantly in the interactive dashboard  

---

🏗️ How I Built It
- Designed **full-stack architecture** using FastAPI backend and HTML/JS frontend  
- Implemented ML prediction models for Diabetes, Stroke, and Heart Disease  
- Built **AI assistant** module using Groq API (Llama 3.1-8B Instant)  
- Created REST API endpoints for predictions and chatbot responses  
- Developed **SPA-like UI** using fetch-based forms to prevent page reloads  
- Dockerized the application for deployment on Hugging Face Spaces  

---

📚 What I Learned
- Full-stack AI application development  
- FastAPI backend architecture and API design  
- Integrating context-aware AI chatbots  
- Handling multiple ML models in a single application  
- Frontend integration using vanilla JS and Jinja2 templates  
- Docker deployment and environment management  
- End-to-end project structuring for a professional portfolio  

---

🚀 How It Could Be Improved
- Add **user authentication and project saving**  
- Implement **real-time collaboration features**  
- Exportable PDF / report generation for predictions  
- Advanced visualizations and personalized recommendations  
- Cloud deployment (AWS / GCP / Azure)  
- Integrate additional disease prediction models  

---

▶️ How to Run the Project

**Clone Repository**
```bash
git clone https://github.com/vinayak533/PredictLab.git
cd PredictLab

Backend Setup

pip install -r requirements.txt
cp .env.example .env   # Add your GROQ_API_KEY
uvicorn main:app --reload

Backend runs at: http://127.0.0.1:8000

Docker Setup (Optional)

docker build -t predictlab .
docker run -p 7860:7860 predictlab

Frontend runs at: http://localhost:7860

📂 Project Structure

PredictLab/
├── main.py                  # FastAPI app (routes, prediction, chat)
├── requirements.txt
├── Dockerfile
├── .env.example              # Template for GROQ_API_KEY
├── models/                   # ML models
│   ├── diabetes_model.sav
│   ├── diabetes_scaler.sav
│   ├── heart_model.pkl
│   └── stroke_model.pkl
├── templates/                # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── diabetes.html
│   ├── stroke.html
│   └── heart.html
└── static/                   # CSS and JS
    └── style.css

⭐ About
PredictLab is an AI-powered platform that predicts Diabetes, Stroke, and Heart Disease risks, explains predictions via a context-aware chatbot, and presents results in a modern, interactive dashboard. It is Docker-ready and deployable on Hugging Face Spaces.
