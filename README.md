# 💙 Relationship Probability Predictor  
*A Machine Learning Project with EDA, Baseline Models, and Streamlit App*

This project predicts the **relationship probability** of students using the dataset from the  
**GDGC NIT Jalandhar – AI/ML Inductions Challenge**.  
It includes a complete ML pipeline from exploratory data analysis (EDA) to model training  
and finally a Streamlit-based interactive prediction interface.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://riyagoyal08010-glitch-relationship-predictor-app-bjo8lq.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/AI-XGBoost-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

An advanced AI-powered analytics dashboard that predicts relationship compatibility and success probability. Built with **XGBoost** and **Streamlit**, this application analyzes **33 distinct student traits**—covering academic performance, social habits, digital footprint, and lifestyle choices—to generate a precise probability score.

---

## 🚀 Live Demo

👉 **[Click here to launch the Dashboard](https://riyagoyal08010-glitch-relationship-predictor-app-bjo8lq.streamlit.app/)**

---

## 📊 Exploratory Data Analysis (EDA)

Before building the model, extensive analysis was performed on the dataset to understand the factors influencing relationships.

* **Correlation Analysis:** A heatmap revealed that social traits (like *Social Habits Score* and *Popularity*) had a stronger positive correlation with relationship probability than academic traits.
* **Feature Distributions:**
    * **Age:** Most students in the dataset were between 18-22 years old.
    * **Digital Footprint:** High variance in *Instagram Followers* and *Gaming Hours*, indicating diverse digital lifestyles.
* **Categorical Impact:** Factors like *Home State* and *Branch* showed interesting clustering, suggesting that students from certain backgrounds or majors had slightly higher average probabilities.

---

## ⚙️ The Process: From Data to Dashboard

This project followed a structured Data Science lifecycle:

### 1. Data Preprocessing
* **Cleaning:** Handled missing values and standardized numeric scales.
* **Encoding:** Converted categorical text (e.g., "CSE", "Delhi") into numeric codes for machine learning compatibility.
* **Feature Selection:** Analyzed 33 distinct features (F1-F33) to ensure relevant inputs for the model.

### 2. Model Training & Selection
We experimented with multiple algorithms to find the best fit:
* **Linear Regression:** Provided a baseline but failed to capture complex, non-linear patterns (R² Score: ~0.56).
* **Random Forest:** Better performance but prone to overfitting on this dataset.
* **XGBoost Regressor (Winner):** Achieved the highest accuracy (R² Score: **~0.67**) with the lowest Error (RMSE). This model was chosen for deployment.

### 3. Application Development
* Built a **Streamlit** frontend for real-time interaction.
* Integrated a **"Smart Encoder"** to handle user inputs dynamically without crashing.
* Designed a **local SQLite database** to store user predictions for historical comparison.

---

## 📈 Project Outcomes

* **High Accuracy:** The final model successfully predicts relationship probability with a low margin of error (MAE ~4.2).
* **Real-time Insights:** Users get instant feedback not just on their score, but on how their *Social, Academic, and Lifestyle* choices balance out via the Radar Chart.
* **Scalable Deployment:** The app is fully deployed on Streamlit Cloud and accessible globally.

---

## ✨ Key App Features

### 🧠 **AI-Powered Analysis**
* Uses the trained XGBoost model to predict scores instantly.
* Handles non-linear relationships between traits like *Study Hours* vs. *Social Habits*.

### 🎨 **Premium UI/UX**
* **Pure Black Aesthetic:** A custom `#000000` dark theme with neon pink (`#ff0055`) accents.
* **Interactive Visuals:**
    * **Gauge Meter:** Real-time probability speedometer.
    * **Personality Radar:** Spider chart visualizing the 5-dimensional personality footprint.

### 💾 **Smart Data Engine**
* **Local History:** Automatically saves every prediction to `predictions.db`.
* **Benchmarking:** Compares your unique score against the live global average of all previous users.

---

## 📂 Project Structure

```text
relationship-predictor/
├── app.py                       # 🧠 Main Application Logic
├── relationship_predictor.json  # 🤖 Trained XGBoost Model
├── requirements.txt             # 📦 Library Dependencies
├── predictions.db               # 💾 Local Database (Auto-generated)
├── .gitignore                   # 🙈 Files to ignore (DB, Cache)
└── README.md                    # 📄 Project Documentation


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

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://riyagoyal08010-glitch-relationship-predictor-app-bjo8lq.streamlit.app/)

