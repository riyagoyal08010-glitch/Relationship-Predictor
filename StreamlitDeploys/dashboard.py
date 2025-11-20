import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive EDA | Relationship Predictor",
    page_icon="📊",
    layout="wide"
)

# --- Custom Styling (Dark Mode Compatible) ---
st.markdown("""
<style>
    .stPlot { border-radius: 10px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\HP\Downloads\train.csv")
    data_path = data.to_csv()
    lookup = pd.read_csv(r"C:\Users\HP\Downloads\feature_lookup.csv")
    lookup_path = lookup.to_csv
    
    if not os.path.exists(data_path):
        return None, None, None
    
    df = pd.read_csv(data_path)
    
    feature_map = {}
    feature_type_map = {}
    
    if os.path.exists(lookup_path):
        lookup = pd.read_csv(lookup_path)
        feature_map = dict(zip(lookup['feature_code'], lookup['relevance']))
        feature_type_map = dict(zip(lookup['feature_code'], lookup['type']))
        
    return df, feature_map, feature_type_map

df, feature_map, feature_type_map = load_data()

# --- Main App ---
def main():
    st.title("📊 Interactive Exploratory Data Analysis")
    st.markdown("Deep dive into the **Relationship Predictor** dataset with interactive charts.")

    if df is None:
        st.error("⚠️ `train.csv` not found. Please place it in the same directory.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Hover over charts to see details. Use the camera icon to download.")
        
        st.divider()
        st.subheader("📁 Feature Dictionary")
        if feature_map:
            lookup_df = pd.DataFrame(list(feature_map.items()), columns=['Code', 'Name'])
            st.dataframe(lookup_df, hide_index=True, height=300)

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Overview", 
        "🎯 Target", 
        "📈 Distributions", 
        "🔥 Heatmap", 
        "⚡ Comparisons"
    ])

    # --- TAB 1: Data Overview ---
    with tab1:
        st.subheader("Dataset at a Glance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        
        with st.expander("View Raw Data", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("Statistical Summary"):
            st.dataframe(df.describe(), use_container_width=True)

    # --- TAB 2: Target Analysis ---
    with tab2:
        st.subheader("Relationship Probability Distribution")
        
        fig = px.histogram(
            df, 
            x="relationship_probability", 
            nbins=40, 
            marginal="box", # Adds a boxplot on top
            title="Target Variable Distribution",
            color_discrete_sequence=["#FF4B4B"] # Streamlit Red
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: Feature Distributions ---
    with tab3:
        st.subheader("Univariate Analysis")
        
        all_features = [c for c in df.columns if c not in ['ID', 'relationship_probability']]
        feature_labels = [f"{col} ({feature_map.get(col, 'Unknown')})" for col in all_features]
        
        selected_label = st.selectbox("Select Feature to Explore:", feature_labels)
        selected_col = selected_label.split(" ")[0]
        real_name = feature_map.get(selected_col, selected_col)
        
        # Detect Type
        is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
        
        if is_numeric:
            fig = px.histogram(
                df, x=selected_col, 
                nbins=30, 
                marginal="violin",
                title=f"Distribution of {real_name}",
                color_discrete_sequence=["#00CC96"] # Teal
            )
        else:
            fig = px.bar(
                df[selected_col].value_counts().reset_index(),
                x="index", y=selected_col,
                labels={"index": real_name, selected_col: "Count"},
                title=f"Count of {real_name}",
                color=selected_col,
                color_continuous_scale="Viridis"
            )
            
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 4: Correlations ---
    with tab4:
        st.subheader("Feature Correlations")
        
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if 'ID' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['ID'])
            
        corr = numeric_df.corr()
        
        fig = px.imshow(
            corr, 
            text_auto=".2f", 
            aspect="auto",
            color_continuous_scale="RdBu_r", # Red-Blue diverging
            title="Correlation Matrix (Numeric Features)"
        )
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 5: Bivariate Analysis ---
    with tab5:
        st.subheader("Feature vs Target Impact")
        
        # Find Categorical Columns
        cat_cols = [c for c in df.columns if df[c].dtype == 'object']
        if not cat_cols: cat_cols = all_features
        
        x_label_options = [f"{col} ({feature_map.get(col, 'Unknown')})" for col in cat_cols]
        x_selected_label = st.selectbox("Select Categorical Group (X-axis):", x_label_options, key="bi_select")
        x_col = x_selected_label.split(" ")[0]
        x_name = feature_map.get(x_col, x_col)
        
        plot_type = st.radio("Visualization Style:", ["Box Plot", "Violin Plot", "Strip Plot"], horizontal=True)
        
        if plot_type == "Box Plot":
            fig = px.box(df, x=x_col, y="relationship_probability", color=x_col, title=f"{x_name} vs Probability")
        elif plot_type == "Violin Plot":
            fig = px.violin(df, x=x_col, y="relationship_probability", box=True, points="all", color=x_col, title=f"{x_name} vs Probability")
        else:
            fig = px.strip(df, x=x_col, y="relationship_probability", color=x_col, title=f"{x_name} vs Probability")
            
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()