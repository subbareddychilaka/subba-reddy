# 🩺 Multi-Disease Prediction System

An AI-powered clinical decision support web application that predicts the risk of four major diseases — **Diabetes**, **Heart Disease**, **Liver Disease**, and **Kidney Disease** — using ensemble machine learning models and real-time biomarker analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Database Setup](#database-setup)
- [Usage](#usage)
- [ML Models](#ml-models)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)

---

## Overview

This system allows healthcare practitioners or individuals to input patient biomarkers and receive an instant AI-based disease risk prediction. Each disease module uses an **ensemble of a Neural Network and an XGBoost classifier**, averaging their probabilities for improved accuracy. All predictions are logged to a MySQL database for record-keeping and statistics.

---

## ✨ Features

- 🔬 **4 Disease Modules** — Diabetes, Heart, Liver, and Kidney disease prediction
- 🤖 **Ensemble ML Models** — Neural Network + XGBoost per disease for better accuracy
- ⚡ **Real-time Feedback** — Color-coded health indicators (Normal / Elevated / Critical) shown as you type
- 💊 **Personalized Precautions** — Disease-specific dietary, exercise, and medical recommendations on positive detection
- 🗄️ **MySQL Integration** — Patient records and prediction results automatically saved to a relational database
- 📊 **Live Dashboard** — Prediction count statistics displayed across all four disease categories
- 🖥️ **Interactive UI** — Built with Streamlit; collapsible sections per disease for a clean experience

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| ML Models | Scikit-learn (Neural Network), XGBoost |
| Data Processing | NumPy, Pandas |
| Model Serialization | Joblib |
| Database | MySQL (via `mysql-connector-python`) |
| Language | Python 3.8+ |

---

## 📁 Project Structure

```
AI-Based-Multi-Disease-Prediction/
│
├── app.py                    # Main Streamlit application
├── train_models.py           # Model training script
├── index.html                # (Supporting HTML file)
│
├── models/
│   ├── diabetes_nn.pkl       # Diabetes Neural Network model
│   ├── diabetes_xgb.pkl      # Diabetes XGBoost model
│   ├── diabetes_scaler.pkl   # Diabetes feature scaler
│   ├── heart_nn.pkl
│   ├── heart_xgb.pkl
│   ├── heart_scaler.pkl
│   ├── liver_nn.pkl
│   ├── liver_xgb.pkl
│   ├── liver_scaler.pkl
│   ├── kidney_nn.pkl
│   ├── kidney_xgb.pkl
│   └── kidney_scaler.pkl
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/subbareddychilaka/AI-Based-Citizen-Grievance-Redressal-System.git
cd AI-Based-Citizen-Grievance-Redressal-System
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install streamlit numpy pandas scikit-learn xgboost joblib mysql-connector-python
```

### 4. Train the Models

If the `.pkl` model files are not present, run the training script first:

```bash
python train_models.py
```

> Make sure the datasets (`heart.csv`, `liver.csv`, `kidney_disease.csv`, and the diabetes dataset) are in the project root before running.

### 5. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🗄️ Database Setup

The app connects to a **MySQL** database to persist patient records and predictions.

### 1. Create the Database

```sql
CREATE DATABASE patient_records;
USE patient_records;
```

### 2. Create Tables

```sql
CREATE TABLE patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    age INT,
    gender VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE diabetes_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    pregnancies FLOAT, glucose FLOAT, blood_pressure FLOAT,
    skin_thickness FLOAT, insulin FLOAT, bmi FLOAT,
    diabetes_pedigree_function FLOAT, age INT,
    prediction_probability FLOAT,
    prediction_result VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE heart_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    age FLOAT, sex FLOAT, chest_pain_type FLOAT, resting_bp FLOAT,
    cholesterol FLOAT, fasting_bs FLOAT, resting_ecg FLOAT,
    max_heart_rate FLOAT, exercise_angina FLOAT, oldpeak FLOAT,
    st_slope FLOAT, ca FLOAT, thal FLOAT,
    prediction_probability FLOAT,
    prediction_result VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE liver_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    age FLOAT, gender FLOAT, total_bilirubin FLOAT, direct_bilirubin FLOAT,
    alkaline_phosphatase FLOAT, alamine_aminotransferase FLOAT,
    aspartate_aminotransferase FLOAT, total_proteins FLOAT,
    albumin FLOAT, albumin_globulin_ratio FLOAT,
    prediction_probability FLOAT,
    prediction_result VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE kidney_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    id_col FLOAT, age FLOAT, blood_pressure FLOAT, specific_gravity FLOAT,
    albumin FLOAT, sugar FLOAT, red_blood_cells FLOAT, pus_cell FLOAT,
    pus_cell_clumps FLOAT, bacteria FLOAT, blood_glucose_random FLOAT,
    blood_urea FLOAT, serum_creatinine FLOAT, sodium FLOAT, potassium FLOAT,
    hemoglobin FLOAT, packed_cell_volume FLOAT, white_blood_cell_count FLOAT,
    red_blood_cell_count FLOAT, hypertension FLOAT, diabetes_mellitus FLOAT,
    coronary_artery_disease FLOAT, appetite FLOAT, peda_edema FLOAT, anemia FLOAT,
    prediction_probability FLOAT,
    prediction_result VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);
```

### 3. Update Credentials

In `app.py`, update the `get_db_connection()` function with your MySQL credentials:

```python
connection = mysql.connector.connect(
    host='localhost',
    user='your_username',
    password='your_password',
    database='patient_records'
)
```

---

## 🚀 Usage

1. Launch the app with `streamlit run app.py`
2. Expand the disease section you want (e.g., **🧪 Diabetes Inputs**)
3. Enter patient biomarker values — real-time color indicators will update as you type
4. Click the **Predict** button for that disease
5. View the prediction probability, result, and (if positive) personalized health precautions
6. The prediction is automatically saved to the MySQL database
7. A live statistics panel at the bottom shows total prediction counts per disease

---

## 🤖 ML Models

Each disease uses an **ensemble of two models**:

| Disease | Features | Models Used |
|---|---|---|
| Diabetes | Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, DPF, Age | NN + XGBoost |
| Heart Disease | Age, Sex, Chest Pain Type, Resting BP, Cholesterol, ECG, Max HR, etc. | NN + XGBoost |
| Liver Disease | Age, Gender, Bilirubin, Alkaline Phosphatase, ALT, AST, Albumin, etc. | NN + XGBoost |
| Kidney Disease | BP, Specific Gravity, Albumin, Creatinine, Hemoglobin, WBC/RBC counts, etc. | NN + XGBoost |

**Prediction formula:**

```
final_probability = (NN_probability + XGBoost_probability) / 2
Positive if final_probability > 0.5
```

---

## 🔮 Future Improvements

- [ ] Add user authentication for doctor/patient login
- [ ] Export predictions as PDF reports
- [ ] Add model explainability (SHAP values) to highlight contributing features
- [ ] Deploy to cloud (AWS / GCP / Streamlit Cloud)
- [ ] Add more disease modules (Cancer, Stroke, etc.)
- [ ] Replace hardcoded DB credentials with environment variables / `.env` file

---

## 👤 Author

**Subba Reddy Chilaka**
[GitHub Profile](https://github.com/subbareddychilaka)

---

## ⚠️ Disclaimer

This application is intended for **educational and research purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for medical decisions.
