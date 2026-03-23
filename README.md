---
title: PredictLab AI
emoji: 🧬
colorFrom: cyan
colorTo: blue
sdk: docker
pinned: false
---

# PredictLab: Machine Learning Portfolio Web App

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Groq](https://img.shields.io/badge/Groq-Llama%203-f15a24?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

A premium, production-ready full-stack machine learning portfolio application built with FastAPI and customized with an elegant Dark Modern Medical/Tech UI. It showcases predictive models (Pima Diabetes, Stroke Risk, House Price) and automatically integrates a Llama 3-powered AI Assistant to contextualize predictions for users.

---

## 📸 Screenshots

*(Add screenshots of your application here — e.g., the homepage grid and one of the prediction forms with the Chatbot open).*

---

## ✨ Features

- **Three Integrated Machine Learning Models**: Built-in support for Diabetes (SVM), Stroke Risk (Classification), and House Price (Regression) prediction models.
- **Graceful Fallbacks**: The application handles missing models fluidly, displaying visually pleasing error placeholders while keeping the rest of the application fully functional.
- **Premium UI & UX**: Gorgeous dashboard styling using a combination of Tailwind CSS and custom, deeply customized animations. Includes robust micro-interactions and async fetch routines for seamless loading.
- **Context-Aware Groq AI Assistant**: A globally persistent chatbot that understands what page you are on, the exact medical inputs provided, what the model predicted, and gently provides domain expertise based on Llama 3 logic via the Groq API.
- **Fully Responsive**: Looks flawless on Desktop, Tablet, and Mobile.
- **No Page-Reload Architecture**: Entirely JS-fetch driven forms.

---

## 🚀 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo>
   cd ml-portfolio
   ```

2. **Install dependencies:**
   It is recommended to run this within a clean Python Virtual Environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and configure your API keys:
   `GROQ_API_KEY=your_key_here`

4. **Add Trained `.sav` Models:**
   Place your pre-trained models within the `/models` directory using the following exact filenames:
   - `diabetes_model.sav` & `diabetes_scaler.sav`
   - `stroke_model.sav`
   - `house_model.sav`
   If these are omitted, the application will simply show "Model not loaded" for those respective predictors.

5. **Start the Uvicorn Server:**
   ```bash
   uvicorn main:app --reload
   ```

6. **View the Application:**
   Open [http://localhost:8000](http://localhost:8000) in your modern web browser.

---

## 🛠️ How to Add New Models

1. **Create an HTML UI**: Duplicate an existing prediction page in `templates/` and link it in `base.html`. Make sure it submits required payload features.
2. **Update loader config**: Edit `models/loader.py` to recognize your newly appended `.sav` file.
3. **Register Route in `main.py`**: Add a new asynchronous API route under `/predict/yours` handling your specific inference transformation arrays.

---

## 📁 Folder Structure

```text
ml-portfolio/
├── main.py                   # Main FastAPI App, Routes, and Groq Chat implementation.
├── requirements.txt          # Python dependencies.
├── README.md                 # Project instructions.
├── .env                      # Secrets & API Keys.
├── models/                   # Contains ML assets and custom scikit model loader block.
│   └── loader.py             
├── templates/                # Jinja2 Layout Files.
│   ├── base.html             # The global layout, includes Navbar and universal AI Chat widget.
│   ├── index.html            # Landing Dashboard.
│   ├── diabetes.html         # Diabetes prediction form layout.
│   ├── stroke.html           # Stroke prediction layout.
│   └── house.html            # House price prediction interface.
└── static/                   
    └── style.css             # Vanilla CSS enhancements overriding standard Tailwind styling for a premium aesthetic.
```
