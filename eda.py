import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS (Pure Black Theme) ---
st.markdown("""
<style>
    /* Black Background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Containers/Cards */
    div.css-1r6slb0 {
        background-color: #111111;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Selectboxes & Inputs */
    .stSelectbox div[data-baseweb="select"] div {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #444;
    }
    
    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000;
    }
    .stTabs [aria-selected="true"] {
        background-color: #222 !important;
        border-top: 2px solid #ff0055;
    }
    
    /* Headings */
    h1, h2, h3 { color: #ffffff !important; }
    p, label { color: #cccccc !important; }
</style>
""", unsafe_allow_html=True)

# --- Smart Data Loading ---
@st.cache_data
def load_data():
    # Fix: Use relative paths so it works on Cloud/GitHub too
    if os.path.exists(r"C:\Users\HP\Downloads\Relationship_Predictor\Input_Data\train.csv"):
        df = pd.read_csv(r"C:\Users\HP\Downloads\Relationship_Predictor\Input_Data\train.csv")
    else:
        st.error("❌ `train.csv` not found.")
        st.info("Please place `train.csv` in the same folder as this script.")
        st.stop()

        
    feature_map = {}
    if os.path.exists(r"C:\Users\HP\Downloads\Relationship_Predictor\Input_Data\feature_lookup.csv"):
        lookup = pd.read_csv(r"C:\Users\HP\Downloads\Relationship_Predictor\Input_Data\feature_lookup.csv")
        feature_map = dict(zip(lookup['feature_code'], lookup['relevance']))
    
        
    return df, feature_map

df, feature_map = load_data()

# --- Helper to Get Real Names ---
def get_name(col):
    return f"{col} ({feature_map.get(col, 'Unknown')})"

# --- MAIN APP ---
def main():
    lookup = pd.read_csv(r"C:\Users\HP\Downloads\Relationship_Predictor\Input_Data\feature_lookup.csv")
    st.title("📊 Interactive EDA (Dark Mode)")
    st.markdown("Explore your dataset with **interactive** charts (Zoom, Pan, Hover).")
   

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Controls")
        st.success(f"✅ Loaded {df.shape[0]} rows")
        st.info("Hover over charts to see details.")
        st.sidebar.dataframe(lookup)
        
    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Target Distribution", 
        "📈 Feature Distributions", 
        "🔥 Correlation Heatmap", 
        "🎻 Bivariate Analysis", "🔢Dataset"
    ])

    # --- 1. Target Analysis ---
    with tab1:
        st.subheader("Target Variable: Relationship Probability")
        
        fig = px.histogram(
            df, 
            x="relationship_probability", 
            nbins=40, 
            marginal="box",  # Adds the boxplot on top automatically
            title="Distribution of Relationship Probability",
            color_discrete_sequence=["#ff0055"] # Neon Pink
        )
        
        # Update layout for dark mode
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 2. Feature Distributions ---
    with tab2:
        st.subheader("Univariate Analysis")
        
        cols = [c for c in df.columns if c not in ['ID', 'relationship_probability']]
        selected_label = st.selectbox("Select Feature:", [get_name(c) for c in cols])
        selected_col = selected_label.split(" ")[0]
        real_name = feature_map.get(selected_col, selected_col)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                fig = px.histogram(
                    df, x=selected_col, 
                    nbins=30, 
                    marginal="violin",
                    title=f"Distribution of {real_name}",
                    color_discrete_sequence=["#00CC96"] # Teal
                )
            else:
                # Bar chart for categories
                count_data = df[selected_col].value_counts().reset_index()
                count_data.columns = [selected_col, 'count']
                fig = px.bar(
                    count_data, x=selected_col, y='count',
                    title=f"Count of {real_name}",
                    color=selected_col,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
            
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### 🔢 Stats")
            stats = df[selected_col].describe()
            st.dataframe(stats, use_container_width=True)

    # --- 3. Heatmap ---
    with tab3:
        st.subheader("Correlation Heatmap")
        
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if 'ID' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['ID'])
            
        corr = numeric_df.corr()
        
        fig = px.imshow(
            corr, 
            text_auto=".2f", 
            aspect="auto",
            color_continuous_scale="RdBu_r", # Red to Blue
            title="Feature Correlations"
        )
        fig.update_layout(
            height=800,
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 4. Bivariate Analysis ---
    with tab4:
        st.subheader("Feature vs Target Relationships")
        
        cat_cols = [c for c in df.columns if df[c].dtype == 'object']
        x_label = st.selectbox("Select Categorical Group (X-Axis):", [get_name(c) for c in cat_cols])
        x_col = x_label.split(" ")[0]
        x_real_name = feature_map.get(x_col, x_col)
        
        plot_type = st.radio("Plot Style:", ["Box Plot", "Violin Plot"], horizontal=True)
        
        if plot_type == "Violin Plot":
            fig = px.violin(
                df, x=x_col, y="relationship_probability", 
                color=x_col, 
                box=True, 
                points="all", # Interactive points!
                title=f"{x_real_name} vs Probability"
            )
        else:
            fig = px.box(
                df, x=x_col, y="relationship_probability", 
                color=x_col, 
                title=f"{x_real_name} vs Probability"
            )
    with tab5:
        st.subheader("Dataset")
        st.dataframe(df)
            
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()