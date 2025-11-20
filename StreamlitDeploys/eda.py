import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Notebook EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Make the plots pop against the background */
    .stPlot {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stApp {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# --- Smart Data Loading ---
@st.cache_data
def load_data():
    # Try to load train.csv
    if os.path.exists("train.csv"):
        df = pd.read_csv("train.csv")
    else:
        st.error("❌ `train.csv` not found.")
        st.info("Please upload `train.csv` to your folder/repository.")
        st.stop()
        
    # Try to load feature names
    feature_map = {}
    if os.path.exists("feature_lookup.csv"):
        lookup = pd.read_csv("feature_lookup.csv")
        feature_map = dict(zip(lookup['feature_code'], lookup['relevance']))
        
    return df, feature_map

df, feature_map = load_data()

# --- Helper to Get Real Names ---
def get_name(col):
    return f"{col} ({feature_map.get(col, 'Unknown')})"

# --- MAIN APP ---
def main():
    st.title("📊 Exploratory Data Analysis (Seaborn)")
    st.markdown("This dashboard replicates the **Seaborn (`sns`)** plots from your research notebook.")

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Visualization Controls")
        st.success(f"Loaded {df.shape[0]} rows")
        
        # Global Plot Settings
        theme = st.selectbox("Seaborn Theme", ["darkgrid", "whitegrid", "ticks", "white"], index=0)
        sns.set_style(theme)
        
    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Target Distribution", 
        "📈 Feature Distributions", 
        "🔥 Correlation Heatmap", 
        "🎻 Bivariate Analysis"
    ])

    # --- 1. Target Analysis (Replicating Notebook Cell) ---
    with tab1:
        st.subheader("Target Variable: Relationship Probability")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        # Notebook Code: sns.histplot(df['relationship_probability'], bins=30, kde=True)
        sns.histplot(df['relationship_probability'], bins=30, kde=True, color="#ff0055", ax=ax)
        ax.set_title("Distribution of Relationship Probability")
        st.pyplot(fig)
        
        st.caption("This plot shows the spread of probability scores across the dataset.")

    # --- 2. Feature Distributions (Replicating the Loop) ---
    with tab2:
        st.subheader("Univariate Analysis")
        
        # Dropdown to replicate the loop behavior in the notebook
        cols = [c for c in df.columns if c not in ['ID', 'relationship_probability']]
        selected_label = st.selectbox("Select Feature to Visualize:", [get_name(c) for c in cols])
        selected_col = selected_label.split(" ")[0]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                # Notebook used histplot for numerics
                sns.histplot(df[selected_col], bins=30, kde=True, ax=ax, color="dodgerblue")
            else:
                # Notebook usually implies countplots for categories
                sns.countplot(x=selected_col, data=df, ax=ax, palette="viridis")
                
            ax.set_title(f"Distribution of {feature_map.get(selected_col, selected_col)}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col2:
            st.write("**Stats:**")
            st.write(df[selected_col].describe())

    # --- 3. Heatmap (Replicating Cell with corr) ---
    with tab3:
        st.subheader("Correlation Heatmap")
        
        # Filter numeric columns like the notebook
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if 'ID' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['ID'])
            
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(16, 12))
        # Notebook Code: sns.heatmap(corr, cmap='coolwarm', annot=False, square=True)
        sns.heatmap(corr, cmap='coolwarm', annot=False, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Full Correlation Heatmap (All Numeric Features)")
        st.pyplot(fig)

    # --- 4. Bivariate Analysis (Violin & Box Plots) ---
    with tab4:
        st.subheader("Feature vs Target Relationships")
        
        # Selectors
        cat_cols = [c for c in df.columns if df[c].dtype == 'object'] # F5, F7, F8, F9 etc
        x_label = st.selectbox("Select X-Axis (Categorical):", [get_name(c) for c in cat_cols])
        x_col = x_label.split(" ")[0]
        
        plot_type = st.radio("Select Plot Type:", ["Violin Plot", "Box Plot"], horizontal=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if plot_type == "Violin Plot":
            # Notebook Code: sns.violinplot(data=df, x=col, y='relationship_probability')
            sns.violinplot(data=df, x=x_col, y='relationship_probability', palette="Set2", ax=ax)
        else:
            # Notebook Code: sns.boxplot(data=df, x=col, y='relationship_probability')
            sns.boxplot(data=df, x=x_col, y='relationship_probability', palette="Set2", ax=ax)
            
        ax.set_title(f"{feature_map.get(x_col, x_col)} vs Relationship Probability")
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    main()