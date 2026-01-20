import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_prep import clean_data
from insights import build_cluster_profile, ensure_derived_metrics, make_radar_fig
from model import cluster_and_plot

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
            
            # Ensure default values are in the available options
            available_options = ["All"] + filtered_naics
            valid_defaults = [x for x in st.session_state.naics_multiselect if x in available_options]
            if not valid_defaults:
                valid_defaults = ["All"]
            
            # Multiselect with filtered options + "All" option
            selected_naics = st.sidebar.multiselect(
                "Select NAICS Description(s)",
                options=available_options,
                default=valid_defaults,
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

        # Preserve Cluster column if it exists in session state
        if 'df_filtered' in st.session_state and 'Cluster' in st.session_state['df_filtered'].columns:
            # Merge the Cluster column back into the filtered data
            cluster_data = st.session_state['df_filtered'][['DUNS Number', 'Cluster']].copy()
            df_filtered = df_filtered.merge(cluster_data, on='DUNS Number', how='left')
        
        st.session_state['df_filtered'] = df_filtered
        st.sidebar.metric("Filtered Companies", len(df_filtered))

    # main layout
    if 'df_filtered' in st.session_state:
        df_vis = st.session_state['df_filtered']
        
        if len(df_vis) == 0:
            # Show message when no companies match the filters
            st.warning("⚠️ No companies match the current filter criteria. Please adjust your filters.")
        else:
            # Two columns: left wider for overview, right for detail
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.subheader("Cluster Overview")
                
                # Ensure derived metrics (preserve existing columns like Cluster)
                if 'Cluster' not in df_vis.columns:
                    df_vis = ensure_derived_metrics(df_vis)
                    st.session_state['df_filtered'] = df_vis

                # Clustering button
                col_button, col_param = st.columns([1, 1])
                with col_param:
                    n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=4, step=1)
                with col_button:
                    st.write("")  # spacing
                    button_text = "Re-apply K-Means Clustering" if 'Cluster' in df_vis.columns else "Apply K-Means Clustering"
                    if st.button(button_text, type="primary"):
                        try:
                            # Select numeric features for clustering
                            feature_cols = ['Revenue (USD)', 'Employees Total', 'IT Spend per Employee', 
                                          'Revenue per Employee', 'Tech Intensity Score', 'Company Age']
                            available_features = [col for col in feature_cols if col in df_vis.columns]
                            
                            if len(available_features) >= 2:
                                # Prepare data
                                X = df_vis[available_features].fillna(0)
                                
                                # Standardize features
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Use model.py clustering function
                                artifacts, fig = cluster_and_plot(
                                    df_vis,
                                    X_scaled,
                                    feature_cols=available_features,
                                    n_clusters=int(n_clusters),
                                    random_state=42
                                )
                                
                                # Update session state with clustered dataframe
                                st.session_state['df_filtered'] = artifacts.df_clustered.copy()
                                st.session_state['pca_fig'] = fig  # Store for later visualization
                                st.success(f"✅ Created {n_clusters} clusters with {len(available_features)} features!")
                                st.rerun()
                            else:
                                st.error(f"Not enough features. Found: {available_features}")
                        except Exception as e:
                            st.error(f"Clustering failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

                if 'Cluster' in df_vis.columns:
                    profile_df = build_cluster_profile(
                        df_vis,
                        cluster_col="Cluster",
                        metrics=[
                            "Revenue (USD)",
                            "Employees Total",
                            "Company Age",
                            "Revenue per Employee",
                            "IT Spend per Employee",
                            "Tech Intensity Score"
                        ],
                        extra_group_cols=["Country", "NAICS Description"]
                    )

                    # Select key columns for display (avoid overwhelming the table)
                    display_cols = [
                        "Cluster", 
                        "Cluster Size",
                        "Revenue (USD)_mean",
                        "Employees Total_mean",
                        "Company Age_mean",
                        "Top Country",
                        "Top NAICS Description"
                    ]
                    
                    # Only show columns that exist
                    available_cols = [col for col in display_cols if col in profile_df.columns]
                    
                    # Rename columns for better readability
                    display_df = profile_df[available_cols].copy()
                    display_df.columns = [
                        col.replace("_mean", "").replace("Top ", "").replace(" (USD)", "($)")
                        for col in display_df.columns
                    ]
                    
                    st.dataframe(
                        display_df.style.format({
                            col: "{:,.0f}" for col in display_df.columns 
                            if display_df[col].dtype in ['float64', 'int64'] and col != 'Cluster'
                        }),
                        width='stretch',
                        height=300
                    )

                    st.caption("📊 Cluster profiles showing size, average metrics, and top characteristics")
                    
                    # Display PCA scatter plot if available
                    if 'pca_fig' in st.session_state:
                        st.plotly_chart(st.session_state['pca_fig'], width='stretch')
                else:
                    st.info("Clustering not yet applied — summary table will appear here once clustering is applied.")

            with col_right:
                st.subheader("Company Detail")
                
                # Company selector – use DUNS Number as unique ID
                if 'DUNS Number' in df_vis.columns:
                    # Find a readable name column - try multiple variations
                    possible_name_cols = ['Company Name', 'Company', 'Name', 'Company Sites']
                    display_col = None
                    for col in possible_name_cols:
                        if col in df_vis.columns:
                            display_col = col
                            break
                    
                    # If no name column found, use DUNS Number
                    if display_col is None:
                        display_col = 'DUNS Number'
                    
                    # Create clean list of options (avoid duplicates) and sort by DUNS Number
                    if display_col == 'DUNS Number':
                        company_list = df_vis[['DUNS Number']].drop_duplicates().sort_values('DUNS Number').reset_index(drop=True)
                    else:
                        company_list = df_vis[[display_col, 'DUNS Number']].drop_duplicates().sort_values('DUNS Number').reset_index(drop=True)
                    
                    # Create combined display: "DUNS: 123456 - Company Name" (DUNS first for sorting)
                    if display_col != 'DUNS Number':
                        # Limit company name length to prevent cutoff, put DUNS first
                        company_list['short_name'] = company_list[display_col].astype(str).str[:50]
                        company_list['display'] = company_list['DUNS Number'].astype(str) + ' | ' + company_list['short_name']
                    else:
                        company_list['display'] = company_list['DUNS Number'].astype(str)
                    
                    company_options = company_list['display'].values.tolist()

                    # Use markdown to add custom CSS for wider selectbox
                    st.markdown("""
                        <style>
                        div[data-baseweb="select"] > div {
                            min-width: 100% !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    # Dropdown
                    selected_display = st.selectbox(
                        "Select Company",
                        options=company_options,
                        index=0 if company_options else None,
                        key="company_select"
                    )

                    if selected_display:
                        # Get corresponding DUNS from the display string
                        selected_idx = company_list[company_list['display'] == selected_display].index[0]
                        selected_duns = company_list.loc[selected_idx, 'DUNS Number']
                        
                        if display_col != 'DUNS Number':
                            selected_name = company_list.loc[selected_idx, display_col]
                            st.write(f"{selected_name} (DUNS: {selected_duns})")
                        else:
                            st.write(f"DUNS: {selected_duns}")
                            st.caption("Note: No company name column found in dataset")

                        # Radar chart placeholder
                        if 'Cluster' in df_vis.columns:
                            try:
                                fig = make_radar_fig(
                                    df_vis,
                                    company_id=selected_duns,
                                    id_col="DUNS Number",
                                    cluster_col="Cluster",
                                    metrics=[
                                        "Revenue (USD)",
                                        "Employees Total",
                                        "Company Age",
                                        "Revenue per Employee",
                                        "IT Spend per Employee",
                                        "Tech Intensity Score"
                                    ],
                                    company_name_col=display_col if display_col != 'DUNS Number' else None
                                )
                                # Update figure layout for better sizing and centering
                                fig.update_layout(
                                    height=500,  # Reduced from 600 to 500 to fit the container better
                                    margin=dict(l=80, r=80, t=40, b=80),  # Increased margins to prevent text cutoff
                                    autosize=True,
                                    # Keep legend at the bottom to save horizontal space
                                    legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=-0.15,  # Pushed slightly lower to avoid overlapping the chart
                                        xanchor="center",
                                        x=0.5
                                    )
                                )
                                st.plotly_chart(fig, width='stretch')
                            except Exception as e:
                                st.error(f"Failed to create radar chart: {str(e)}")
                        else:
                            st.info("Clustering not yet applied — radar chart will appear here once clusters exist.")
                else:
                    st.warning("No 'DUNS Number' column found — cannot select companies.")
else:
    st.sidebar.info("Upload data to enable filters.")