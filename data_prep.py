import streamlit as st
import pandas as pd
import numpy as np

# function to clean
def clean_data(df):
    """
    Person 1's full cleaning pipeline - returns cleaned df
    """
    # Filter Active
    df_clean = df[df['Company Status (Active/Inactive)'] == 'Active'].copy()

    # Golden cols imputation
    golden_cols = ['Revenue (USD)', 'Employees Total', 'IT Budget', 'Year Found']
    for col in golden_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Categorical fill
    cat_cols = ['Ownership Type', 'Entity Type', 'Manufacturing Status']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')

    # Outlier capping
    for col in ['Revenue (USD)', 'Employees Total']:
        if col in df_clean.columns:
            upper_limit = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(upper=upper_limit)

    # Convert object columns to string to avoid Arrow serialization issues
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str)

    return df_clean

#Type this in terminal to run: python -m streamlit run data_prep.py  
# UI Setup
st.set_page_config(page_title="AI Company Intelligence", layout="wide")
st.title("Datathon Prototype: Day 1")

# Step 1: Data Loading (with fallback for testing)
uploaded_file = st.file_uploader("Upload Champions Group Excel/CSV", type=['csv', 'xlsx'])

if uploaded_file:
    # Handle both Excel and CSV
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Filter Active Companies Only
    # Based on the Data Dictionary column: "Company Status (Active/Inactive)"
    df_clean = df[df['Company Status (Active/Inactive)'] == 'Active'].copy()

# Missing Value Treatment (Imputation)
    # Filling 'Golden Columns' with the median 
    golden_cols = ['Revenue (USD)', 'Employees Total', 'IT Budget', 'Year Found']
    for col in golden_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Handle Categoricals with 'Unknown' 
    cat_cols = ['Ownership Type', 'Entity Type', 'Manufacturing Status']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')

    # Step 4: Outlier Capping (99th Percentile) 
    for col in ['Revenue (USD)', 'Employees Total']:
        upper_limit = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(upper=upper_limit)
#  Step 5: Initial Quality Report
    st.success("Data Loaded and Cleaned Successfully!")
    st.divider()
    st.subheader("Initial Quality Report (Role 1 Validation)")

    # Row 1: Missing Data & Industry Distribution
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.write("**Missing Data Percentage (%)**")
        # Calculate raw missing percentage to show the judges what we fixed 
        missing_pct = (df.isnull().sum() / len(df)) * 100
        st.write(missing_pct[missing_pct > 0].sort_values(ascending=False).head(5))
        
    with col_q2:
        st.write("**Top 10 Industry Sectors (NAICS)**")
        if 'NAICS Description' in df_clean.columns:
            # Displaying top 10 as requested by the workplan 
            top_naics = df_clean['NAICS Description'].value_counts().head(10)
            st.bar_chart(top_naics)
        else:
            st.warning("NAICS Description column not found.")

    # Row 2: Verification of Cleaning Steps
    st.write("---")
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        # Proving the 'Golden Columns' now have 0 missing values
        st.write("**Post-Cleaning Missing Count**")
        st.write(df_clean[['Revenue (USD)', 'Employees Total', 'Year Found']].isnull().sum())
    
    with v_col2:
        # Proving we created 'Unknown' labels for missing categoricals
        st.write("**Categorical Check**")
        unknown_count = (df_clean['Ownership Type'] == 'Unknown').sum()
        st.write(f"'Unknown' Labels Created: {unknown_count}")
        
    with v_col3:
        # Proving we capped the outliers at the 99th percentile 
        st.write("**Outlier Check (Revenue)**")
        raw_max = df['Revenue (USD)'].max()
        clean_max = df_clean['Revenue (USD)'].max()
        st.write(f"Raw Max: {raw_max:,.0f}")
        st.write(f"Capped Max: {clean_max:,.0f}")

    # Row 3: Final Metrics & Geography
    st.write("---")
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.metric("Active Companies Retained", len(df_clean))
    with m_col2:
        st.write("**Top 5 Countries in Dataset**")
        # Cleaning country names for the final report 
        df_clean['Country'] = df_clean['Country'].astype(str).str.strip().str.upper()
        country_counts = df_clean['Country'].value_counts().head(5)
        st.bar_chart(country_counts)
    st.divider()
    # Step 6: Cache data for the rest of the team
    st.session_state['df_clean'] = df_clean
    
    st.write("### Preview of Cleaned Data")
    st.dataframe(df_clean.head(10))

    # --- PASTE THE DOWNLOAD BUTTON HERE ---
    # It must be lined up with the 'st.write' above it
    csv_data = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Complete Cleaned Dataset",
        data=csv_data,
        file_name="champions_group_cleaned.csv",
        mime="text/csv",
        help="Click here to download the full processed dataset for the team."
    )

else:
    st.info("Waiting for file upload...")