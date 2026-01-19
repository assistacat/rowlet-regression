import streamlit as st
import pandas as pd
from data_prep import clean_data

# set page configuration for wide layout and title
st.set_page_config(page_title="Company Intelligence Prototype", layout="wide")

# main title
st.title("Company Intelligence Prototype")

# Main content layout
col1, col2 = st.columns([2, 1])  # Left wider for overview, right for detail

with col1:
    st.subheader("Cluster Overview")
    st.write("Interactive scatter plot and cluster summary table will appear here once data is processed.")

with col2:
    st.subheader("Company Detail")
    st.write("Select a company to view radar chart, comparison, and AI business insight.")

# Sidebar for data and filters
st.sidebar.header("Data and Filters")

# Placeholder for uploader (tbc)
st.sidebar.header("Upload Data")

# Simple test message
st.write("This is a prototype app skeleton. Data loading will be added next.")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file based on extension
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    # clean data using imported function
    df_clean = clean_data(df_raw)
    
    st.session_state['df_clean'] = df_clean

    st.sidebar.success('Data cleaned and loaded successfully!')
    st.success(f"Data cleaned & ready! Active companies: {len(df_clean)} | Shape: {df_clean.shape}")

    # Show a preview of the first few rows
    st.subheader("Data Preview (First 10 rows)")
    st.dataframe(df_clean.head(10))

    # Optional: Reuse Person 1's download button
    csv_data = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Dataset",
        data=csv_data,
        file_name="champions_group_cleaned.csv",
        mime="text/csv"
    )

    #sidebar filters (currently placeholders)
    st.sidebar.header("Filters")
    st.sidebar.write("Filter options will be added here after data is cleaned.")

    # Dummy placeholders - to connect real ones later
    industry_filter = st.sidebar.multiselect(
        "Industry (NAICS/SIC major)",
        options=["Loading..."],
        default=[]
    )
    country_filter = st.sidebar.multiselect(
        "Country / Region",
        options=["Loading..."],
        default=[]
    )
    revenue_slider = st.sidebar.slider(
        "Revenue Range (USD)",
        min_value=0,
        max_value=1000000000,
        value=(0, 1000000000),
        step=1000000
    )
    employee_slider = st.sidebar.slider(
        "Employee Range",
        min_value=0,
        max_value=10000,
        value=(0, 10000),
        step=100
    )

else:
    st.sidebar.write("Please upload a CSV or Excel file to proceed.")