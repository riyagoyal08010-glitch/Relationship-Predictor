# 💙 Relationship Probability Predictor  
*A Machine Learning Project with EDA, Baseline Models, and Streamlit App*

This project predicts the **relationship probability** of students using the dataset from the  
**GDGC NIT Jalandhar – AI/ML Inductions Challenge**.  
It includes a complete ML pipeline from exploratory data analysis (EDA) to model training  
and finally a Streamlit-based interactive prediction interface.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![XGBoost](https://img.shields.io/badge/AI-XGBoost-green)

An AI-powered dashboard that predicts the "relationship probability" of a student based on 33 distinct features ranging from academic performance and social habits to digital footprint and lifestyle choices.

## ✨ Features

* **🧠 AI Analysis:** Powered by a trained XGBoost Regressor model (`relationship_predictor.pkl`) to calculate probability scores.
* **📊 Interactive Dashboard:**
    * **Gauge Chart:** Visualizes the success likelihood (0-100%).
    * **Personality Radar:** A spider chart comparing Social, Academic, Digital, Lifestyle, and Ego scores.
* **🌑 Aesthetic Dark Mode:** A custom-styled, glassmorphism-inspired dark UI for a premium look.
* **💾 History Tracking:** Automatically saves predictions to a local SQLite database (`predictions.db`) to compare your score against the average of previous users.
* **🛡️ Robust Inputs:** Features a "Smart Encoder" that handles categorical text inputs safely to prevent model crashes.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Model:** XGBoost (Scikit-Learn API)
* **Visualization:** Plotly Express & Graph Objects
* **Database:** SQLite3 (Local)
* **Data Handling:** Pandas & Numpy

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/riyagoyal08010-glitch/Relationship-Predictor.git]
    cd Relationship-Predictor
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ensure the model is present:**
    Make sure `relationship_predictor.pkl` is in the root directory.

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

```text
├── app.py                      # Main application code
├── relationship_predictor.pkl  # The trained XGBoost model
├── requirements.txt            # List of python libraries
├── predictions.db              # Local database (Generated automatically)
├── .gitignore                  # Files to ignore (e.g., the database)
└── README.md                   # Documentation

---

## 📌 Project Overview
This project aims to build a regression model that predicts how likely a student is to be in a relationship (0–100 scale).  
The workflow includes:

- 🔍 **Exploratory Data Analysis (EDA)**
- 🔧 **Preprocessing** (Label Encoding + Scaling)
- 🤖 **Model Training** (Linear Regression, Random Forest, XGBoost)
- 📊 **Evaluation** using RMSE, MAE, R²
- 📦 **Model Export** using joblib

---

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook contains:

- 📈 Distribution plots  
- 🎻 Violin plots (categorical vs relationship_probability)  
- 📦 Boxplots  
- 🔥 Correlation heatmaps  
- 📊 Train vs Test distribution comparison  
- 🧩 Feature importance plots  

These help in understanding patterns and influential features.

---

## 🔧 Preprocessing Steps

- Removed irrelevant `ID` column  
- Handled numeric and categorical features separately  
- Applied **Label Encoding**  
- Applied **StandardScaler** for numeric features  
- Split into Train/Validation (80/20)  

---

## 🤖 Models Trained

Several beginner-friendly baseline ML models were trained:

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

Metrics used for evaluation:

- RMSE  
- MAE  
- R² Score  

The best model was exported using:

```python
joblib.dump(best_model, "relationship_predictor.pkl")

APP URL : https://riyagoyal08010-glitch-relationship-predictor-app-bjo8lq.streamlit.app/

