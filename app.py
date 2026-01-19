import streamlit as st
import pandas as pd

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
    with st.sidebar.spinner('Loading data...'):
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        # cache raw data for the time being (to be replaced with cleaned data later)
        st.session_state.df_raw = df_raw
    st.sidebar.success('Data loaded successfully!')
    st.success(f"Data Loaded! Shape: {st.session_state.df_raw.shape}")

    # Show a preview of the first few rows
    st.subheader("Data Preview (First 5 rows)")
    st.dataframe(st.session_state.df_raw.head(5))

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