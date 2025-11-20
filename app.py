import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

joblib.load(r"C:\Users\HP\Downloads\Relationship_Predictor\relationship_predictor.pkl")

# --- 1. CONFIGURATION & PURE BLACK THEME ---
st.set_page_config(
    page_title="Relationship Predictor",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* PURE BLACK BACKGROUND */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Inputs - Darker Grey for contrast */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] div {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333;
    }
    
    /* Cards / Containers */
    div.css-1r6slb0 {
        background-color: #111111;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Metric Boxes */
    div[data-testid="stMetric"] {
        background-color: #111111;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #888888 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000;
    }
    .stTabs [aria-selected="true"] {
        background-color: #222 !important;
        border-top: 2px solid #ff0055;
    }
    
    /* Buttons - Neon Accent */
    div.stButton > button {
        background: #ff0055;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: #ff3377;
        box-shadow: 0 0 15px rgba(255, 0, 85, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. FEATURE CONFIG ---
FEATURE_CONFIG = {
    "👤 Personal": {
        "F1": {"name": "Age", "type": "num", "min": 17, "max": 30, "default": 20},
        "F2": {"name": "Height (cm)", "type": "num", "min": 140, "max": 200, "default": 170},
        "F3": {"name": "Weight (kg)", "type": "num", "min": 40, "max": 120, "default": 65},
        "F8": {"name": "Home State", "type": "cat", "options": ["Delhi", "Punjab", "UP", "Haryana", "Rajasthan", "HP", "Other"]},
        "F22": {"name": "Beard Length (mm)", "type": "num", "min": 0.0, "max": 50.0, "default": 2.0},
        "F24": {"name": "Shoe Size (US)", "type": "num", "min": 4.0, "max": 15.0, "default": 9.0},
    },
    "🎓 Academic": {
        "F5": {"name": "Branch", "type": "cat", "options": ["CSE", "IT", "ECE", "ME", "BIOTECH", "Civil", "Other"]},
        "F6": {"name": "Study Hours", "type": "num", "min": 0.0, "max": 16.0, "default": 4.0},
        "F9": {"name": "Dropper Status", "type": "cat", "options": ["No drop", "Single drop", "Double drop"]},
        "F10": {"name": "School Board", "type": "cat", "options": ["CBSE", "ICSE", "State Board", "Other"]},
    },
    "🗣️ Social": {
        "F7": {"name": "Passion/Drive", "type": "cat", "options": ["Low passion", "Moderate", "Highly passionate"]},
        "F13": {"name": "Personality", "type": "cat", "options": ["Introvert", "Extrovert", "Ambivert"]},
        "F14": {"name": "Social Score", "type": "num", "min": 0.0, "max": 10.0, "default": 5.0},
        "F15": {"name": "Comm. Skills", "type": "num", "min": 0.0, "max": 10.0, "default": 5.0},
        "F16": {"name": "Public Speaking", "type": "num", "min": 0.0, "max": 10.0, "default": 5.0},
        "F17": {"name": "Popularity", "type": "num", "min": 0.0, "max": 10.0, "default": 5.0},
        "F18": {"name": "Approachability", "type": "num", "min": 0.0, "max": 10.0, "default": 5.0},
        "F19": {"name": "Availability", "type": "num", "min": 0.0, "max": 10.0, "default": 5.0},
    },
    "🧘 Lifestyle": {
        "F4": {"name": "Gym (Days/Week)", "type": "cat_int", "min": 0, "max": 7, "default": 3},
        "F20": {"name": "Ego Score", "type": "num", "min": 0.0, "max": 10.0, "default": 3.0},
        "F21": {"name": "Stress Level", "type": "num", "min": 0.0, "max": 10.0, "default": 5.0},
        "F31": {"name": "Room Cleanliness", "type": "num", "min": 0.0, "max": 10.0, "default": 5.0},
        "F32": {"name": "Sleep Hours", "type": "num", "min": 0.0, "max": 12.0, "default": 7.0},
        "F11": {"name": "Sports Played", "type": "cat_int", "min": 0, "max": 5, "default": 1},
        "F12": {"name": "Clubs Joined", "type": "cat_int", "min": 0, "max": 5, "default": 1},
        "F23": {"name": "Daily Coffee", "type": "cat_int", "min": 0, "max": 10, "default": 1},
        "F30": {"name": "Weekly Food Orders", "type": "cat_int", "min": 0, "max": 20, "default": 2},
    },
    "📱 Digital": {
        "F25": {"name": "Netflix (Hrs/Wk)", "type": "num", "min": 0.0, "max": 50.0, "default": 5.0},
        "F26": {"name": "Insta Followers", "type": "num", "min": 0.0, "max": 50000.0, "default": 500.0},
        "F27": {"name": "Playlists", "type": "num", "min": 0.0, "max": 500.0, "default": 50.0},
        "F28": {"name": "Memes Shared/Day", "type": "cat_int", "min": 0, "max": 50, "default": 5},
        "F29": {"name": "Gaming (Hrs/Wk)", "type": "num", "min": 0.0, "max": 50.0, "default": 2.0},
        "F33": {"name": "Anime Watched", "type": "num", "min": 0.0, "max": 200.0, "default": 10.0},
    }
}
FEATURE_ORDER = [f"F{i}" for i in range(1, 34)]

# --- 3. DATABASE ---
DB_FILE = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, probability REAL)''')
    conn.commit()
    conn.close()

def save_to_db(prob):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO history (timestamp, probability) VALUES (?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), prob))
        conn.commit()
        conn.close()
    except: pass

def get_history():
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql("SELECT * FROM history", conn)
        conn.close()
        return df
    except: return pd.DataFrame()

# --- 4. MAIN APP ---
def main():
    init_db()
    st.title("💘 Relationship Probability Predictor")
    
    # INPUTS
    user_inputs = {}
    tabs = st.tabs(list(FEATURE_CONFIG.keys()))
    for i, (group, features) in enumerate(FEATURE_CONFIG.items()):
        with tabs[i]:
            cols = st.columns(4)
            col_idx = 0
            for f_code, config in features.items():
                with cols[col_idx % 4]:
                    if config['type'] == 'num':
                        user_inputs[f_code] = st.number_input(config['name'], float(config['min']), float(config['max']), float(config['default']), key=f_code)
                    elif config['type'] == 'cat':
                        user_inputs[f_code] = st.selectbox(config['name'], config['options'], key=f_code)
                    elif config['type'] == 'cat_int':
                        user_inputs[f_code] = st.number_input(config['name'], int(config['min']), int(config['max']), int(config['default']), key=f_code)
                col_idx += 1

    st.markdown("---")

    # PREDICT
    if st.button("🔮 Run Analysis"):
        try:
            # 1. Load Model
            if not os.path.exists("relationship_predictor.pkl"):
                st.error("⚠️ Model `relationship_predictor.pkl` not found.")
                st.stop()
            model = joblib.load("relationship_predictor.pkl")

            # 2. Prepare Data
            input_df = pd.DataFrame([user_inputs])
            input_df = input_df[FEATURE_ORDER]

            # 3. Encode Text -> Numbers (Smart Fix)
            for col in input_df.columns:
                if input_df[col].dtype == 'object' or input_df[col].dtype.name == 'category':
                    input_df[col] = input_df[col].astype('category').cat.codes
            
            # 4. Array Conversion
            model_input = input_df.astype(float).values
            
            # 5. PREDICT (With 33 vs 34 Column Fix)
            try:
                prediction = model.predict(model_input)[0]
            except Exception:
                # If 33 columns fail, try 34 (add dummy ID)
                dummy_id = np.zeros((model_input.shape[0], 1))
                model_input_fixed = np.hstack((dummy_id, model_input))
                prediction = model.predict(model_input_fixed)[0]
            
            # Result handling
            if isinstance(prediction, (list, np.ndarray)): prediction = prediction[0]
            prediction = float(max(0, min(100, prediction)))
            save_to_db(prediction)

            # --- DASHBOARD ---
            k1, k2, k3 = st.columns(3)
            with k1: st.metric("❤️ Probability", f"{prediction:.1f}%")
            with k2: 
                status = "Solo" if prediction < 40 else "Mingling" if prediction < 75 else "Taken"
                st.metric("📝 Status", status)
            with k3:
                hist = get_history()
                avg = hist['probability'].mean() if not hist.empty else 0
                st.metric("⚖️ Vs Average", f"{avg:.1f}%", f"{prediction-avg:.1f}%")

            # Graphs
            g1, g2 = st.columns(2)
            with g1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prediction,
                    title={'text': "Likelihood", 'font': {'color': 'white'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': 'white'}, 
                        'bar': {'color': "#ff0055"},
                        'bgcolor': "#333333"
                    }
                ))
                fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
                st.plotly_chart(fig, use_container_width=True)
            
            with g2:
                vals = [min(10, (user_inputs['F14']+user_inputs['F17'])/2), min(10, user_inputs['F6']), 
                        min(10, user_inputs['F25']/5), min(10, user_inputs['F32']), min(10, user_inputs['F20'])]
                fig_rad = go.Figure(go.Scatterpolar(
                    r=vals, 
                    theta=['Social', 'Acad', 'Digi', 'Life', 'Ego'], 
                    fill='toself', 
                    line_color='#ff0055',
                    fillcolor='rgba(255, 0, 85, 0.3)'
                ))
                fig_rad.update_layout(
                    height=300, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(color='white', showline=False)), 
                    font={'color': 'white'}
                )
                st.plotly_chart(fig_rad, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Ensure 'relationship_predictor.pkl' is in the same folder.")

if __name__ == "__main__":
    main()