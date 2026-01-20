import streamlit as st
import pandas as pd
from data_prep import clean_data

# set page configuration for wide layout and title
st.set_page_config(page_title="Company Intelligence Prototype", layout="wide")

# main title
st.title("Company Intelligence Prototype")

# Sidebar for data and filters
st.sidebar.header("Data and Filters")

# Uploader header
st.sidebar.header("Upload Data")

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

    # Reuse Person 1's download button
    csv_data = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Dataset",
        data=csv_data,
        file_name="champions_group_cleaned.csv",
        mime="text/csv"
    )

    # FILTERS SECTION in sidebar
    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean'].copy()

        st.sidebar.subheader("Data Filters")

        # NAICS Description Filter with Search
        st.sidebar.subheader("Industry Filter")

        if 'NAICS Description' in df.columns:
            all_naics = sorted(df['NAICS Description'].dropna().unique().tolist())

            # Search box
            naics_search = st.sidebar.text_input(
                "Search NAICS Description",
                value="",
                placeholder="Type to filter (e.g. software, manufacturing)",
                key="naics_search"
            )

            # Filter options based on search (case-insensitive, partial match)
            if naics_search:
                search_term = naics_search.lower().strip()
                filtered_naics = [x for x in all_naics if search_term in x.lower()]
            else:
                filtered_naics = all_naics

            # Callback to handle "All" removal
            def handle_naics_change():
                current = st.session_state.naics_multiselect
                if "All" in current and len(current) > 1:
                    # Remove "All" when other items are selected
                    st.session_state.naics_multiselect = [x for x in current if x != "All"]
                elif len(current) == 0:
                    # Reset to "All" if nothing selected
                    st.session_state.naics_multiselect = ["All"]

            # Initialize default
            if 'naics_multiselect' not in st.session_state:
                st.session_state.naics_multiselect = ["All"]
            
            # Multiselect with filtered options + "All" option
            selected_naics = st.sidebar.multiselect(
                "Select NAICS Description(s)",
                options=["All"] + filtered_naics,
                default=st.session_state.naics_multiselect,
                key="naics_multiselect",
                on_change=handle_naics_change
            )
        else:
            st.sidebar.warning("NAICS Description column not found.")
            selected_naics = ["All"]

        # Country
        if 'Country' in df.columns:
            # Callback to handle "All" removal for Country
            def handle_country_change():
                current = st.session_state.country_multiselect
                if "All" in current and len(current) > 1:
                    # Remove "All" when other items are selected
                    st.session_state.country_multiselect = [x for x in current if x != "All"]
                elif len(current) == 0:
                    # Reset to "All" if nothing selected
                    st.session_state.country_multiselect = ["All"]

            # Initialize default
            if 'country_multiselect' not in st.session_state:
                st.session_state.country_multiselect = ["All"]

            country_options = ["All"] + sorted(df['Country'].dropna().unique().tolist())
            selected_countries = st.sidebar.multiselect(
                "Country",
                options=country_options,
                default=st.session_state.country_multiselect,
                key="country_multiselect",
                on_change=handle_country_change
            )
        else:
            selected_countries = ["All"]
            st.sidebar.warning("Country column not found.")

        # Revenue slider
        if 'Revenue (USD)' in df.columns:
            rev_min = int(df['Revenue (USD)'].min(skipna=True))
            rev_max = int(df['Revenue (USD)'].max(skipna=True))
            rev_range = st.sidebar.slider(
                "Revenue Range (USD)",
                min_value=rev_min,
                max_value=rev_max,
                value=(rev_min, rev_max),
                step=100000  # reasonable step size
            )
        else:
            rev_range = (0, 1000000000)
            st.sidebar.warning("Revenue (USD) column not found.")

        # Employees slider
        if 'Employees Total' in df.columns:
            emp_min = int(df['Employees Total'].min(skipna=True))
            emp_max = int(df['Employees Total'].max(skipna=True))
            emp_range = st.sidebar.slider(
                "Employees Total",
                min_value=emp_min,
                max_value=emp_max,
                value=(emp_min, emp_max),
                step=100
            )
        else:
            emp_range = (0, 10000)
            st.sidebar.warning("Employees Total column not found.")

        # Apply filters
        df_filtered = df.copy()
        if "All" not in selected_naics:
            df_filtered = df_filtered[df_filtered['NAICS Description'].isin(selected_naics)]
        if "All" not in selected_countries:
            df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]
        df_filtered = df_filtered[
            (df_filtered['Revenue (USD)'] >= rev_range[0]) &
            (df_filtered['Revenue (USD)'] <= rev_range[1])
        ]
        df_filtered = df_filtered[
            (df_filtered['Employees Total'] >= emp_range[0]) &
            (df_filtered['Employees Total'] <= emp_range[1])
        ]

        st.session_state['df_filtered'] = df_filtered
        st.sidebar.metric("Filtered Companies", len(df_filtered))

    # main layout
    if 'df_filtered' in st.session_state and len(st.session_state['df_filtered']) > 0:
        df_vis = st.session_state['df_filtered']

        # Two columns: left wider for overview, right for detail
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Cluster Overview")
            st.info("Cluster scatter plot and summary table will appear here once clustering is applied.")

        with col_right:
            st.subheader("Company Detail")
            st.info("Select a company to view radar chart, comparison, and AI insights.")
else:
    st.sidebar.info("Upload data to enable filters.")