# 💙 Relationship Probability Predictor  
*A Machine Learning Project with EDA, Baseline Models, and Streamlit App*

This project predicts the **relationship probability** of students using the dataset from the  
**GDGC NIT Jalandhar – AI/ML Inductions Challenge**.  
It includes a complete ML pipeline from exploratory data analysis (EDA) to model training  
and finally a Streamlit-based interactive prediction interface.

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



