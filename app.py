import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from data_prep import clean_data
from insights import build_cluster_profile, ensure_derived_metrics, make_radar_fig, compute_anomaly_score
from model_day1 import cluster_and_plot
from groq import Groq

# Initalise Groq client
client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

# set page configuration for wide layout and title
st.set_page_config(page_title="Company Intelligence Prototype", layout="wide")

# Custom theme and styling
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
.stButton>button { background-color: #4CAF50; color: white; }
</style>
""", unsafe_allow_html=True)

# Main header with description
st.markdown("""
### 🎯 AI-Driven Company Intelligence Prototype
Upload firmographic data → filter → cluster → explore peers → get AI insights.  
**Built for the Champions Group Datathon** – turns raw data into actionable intelligence.
""")

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

    # FILTERS SECTION in sidebar
    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean'].copy()

        st.sidebar.subheader("Data Filters")

        if 'NAICS Description' in df.columns:
            all_naics = sorted(df['NAICS Description'].dropna().unique().tolist())

# NAICS Description Filter (Cleaned Up)
        #st.sidebar.subheader("Industry Filter")

        if 'NAICS Description' in df.columns:
            all_naics = sorted(df['NAICS Description'].dropna().unique().tolist())

            # --- DELETED SEARCH BOX SECTON HERE --- 

            # Callback to handle "All" removal
            def handle_naics_change():
                current = st.session_state.naics_multiselect
                if "All" in current and len(current) > 1:
                    st.session_state.naics_multiselect = [x for x in current if x != "All"]
                elif len(current) == 0:
                    st.session_state.naics_multiselect = ["All"]

            # Initialize default
            if 'naics_multiselect' not in st.session_state:
                st.session_state.naics_multiselect = ["All"]
            
            # Ensure default values are in the available options
            # CHANGE 1: Use 'all_naics' here instead of 'filtered_naics'
            available_options = ["All"] + all_naics 
            
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
                    st.session_state.country_multiselect = [x for x in current if x != "All"]
                elif len(current) == 0:
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

        # --- REVENUE FILTER (Synced Slider + Inputs) ---
        if 'Revenue (USD)' in df.columns:
            rev_min_global = int(df['Revenue (USD)'].min(skipna=True))
            rev_max_global = int(df['Revenue (USD)'].max(skipna=True))
            
            # 1. Initialize session state keys if they don't exist
            if 'rev_slider' not in st.session_state:
                st.session_state['rev_slider'] = (rev_min_global, rev_max_global)
            if 'rev_min_input' not in st.session_state:
                st.session_state['rev_min_input'] = rev_min_global
            if 'rev_max_input' not in st.session_state:
                st.session_state['rev_max_input'] = rev_max_global

            st.sidebar.markdown("**Revenue Range (USD)**")
            
            # 2. Define Sync Functions
            def update_rev_slider():
                # When input box changes, update the slider state
                mn = st.session_state['rev_min_input']
                mx = st.session_state['rev_max_input']
                # Basic validation: ensure min <= max
                if mn > mx:
                    mn, mx = mx, mn
                    st.session_state['rev_min_input'] = mn
                    st.session_state['rev_max_input'] = mx
                st.session_state['rev_slider'] = (mn, mx)

            def update_rev_inputs():
                # When slider moves, update the input box states
                mn, mx = st.session_state['rev_slider']
                st.session_state['rev_min_input'] = mn
                st.session_state['rev_max_input'] = mx

            # 3. Create Widgets
            col_r1, col_r2 = st.sidebar.columns(2)
            with col_r1:
                st.number_input("Min Revenue", min_value=rev_min_global, max_value=rev_max_global, key='rev_min_input', on_change=update_rev_slider, label_visibility="collapsed")
            with col_r2:
                st.number_input("Max Revenue", min_value=rev_min_global, max_value=rev_max_global, key='rev_max_input', on_change=update_rev_slider, label_visibility="collapsed")

            st.sidebar.slider(
                "Revenue Slider",
                min_value=rev_min_global,
                max_value=rev_max_global,
                key='rev_slider',
                step=100000,
                on_change=update_rev_inputs,
                label_visibility="collapsed"
            )
            
            # 4. Use the synchronized values for filtering
            rev_start, rev_end = st.session_state['rev_slider']
        else:
            rev_start, rev_end = (0, 1000000000)
            st.sidebar.warning("Revenue (USD) column not found.")

        # --- EMPLOYEES FILTER (Synced Slider + Inputs) ---
        if 'Employees Total' in df.columns:
            emp_min_global = int(df['Employees Total'].min(skipna=True))
            emp_max_global = int(df['Employees Total'].max(skipna=True))

            # 1. Initialize session state keys
            if 'emp_slider' not in st.session_state:
                st.session_state['emp_slider'] = (emp_min_global, emp_max_global)
            if 'emp_min_input' not in st.session_state:
                st.session_state['emp_min_input'] = emp_min_global
            if 'emp_max_input' not in st.session_state:
                st.session_state['emp_max_input'] = emp_max_global

            st.sidebar.markdown("**Employees Total**")
            
            # 2. Define Sync Functions
            def update_emp_slider():
                mn = st.session_state['emp_min_input']
                mx = st.session_state['emp_max_input']
                if mn > mx:
                    mn, mx = mx, mn
                    st.session_state['emp_min_input'] = mn
                    st.session_state['emp_max_input'] = mx
                st.session_state['emp_slider'] = (mn, mx)

            def update_emp_inputs():
                mn, mx = st.session_state['emp_slider']
                st.session_state['emp_min_input'] = mn
                st.session_state['emp_max_input'] = mx

            # 3. Create Widgets
            col_e1, col_e2 = st.sidebar.columns(2)
            with col_e1:
                st.number_input("Min Employees", min_value=emp_min_global, max_value=emp_max_global, key='emp_min_input', on_change=update_emp_slider, label_visibility="collapsed")
            with col_e2:
                st.number_input("Max Employees", min_value=emp_min_global, max_value=emp_max_global, key='emp_max_input', on_change=update_emp_slider, label_visibility="collapsed")

            st.sidebar.slider(
                "Employee Slider",
                min_value=emp_min_global,
                max_value=emp_max_global,
                key='emp_slider',
                step=10,
                on_change=update_emp_inputs,
                label_visibility="collapsed"
            )
            
            # 4. Use the synchronized values for filtering
            emp_start, emp_end = st.session_state['emp_slider']
        else:
            emp_start, emp_end = (0, 10000)
            st.sidebar.warning("Employees Total column not found.")

        # Apply filters
        df_filtered = df.copy()
        if "All" not in selected_naics:
            df_filtered = df_filtered[df_filtered['NAICS Description'].isin(selected_naics)]
        if "All" not in selected_countries:
            df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]
        
        # Numeric range filtering using PRECISE start/end values
        df_filtered = df_filtered[
            (df_filtered['Revenue (USD)'] >= rev_start) &
            (df_filtered['Revenue (USD)'] <= rev_end) &
            (df_filtered['Employees Total'] >= emp_start) &
            (df_filtered['Employees Total'] <= emp_end)
        ]

        # DAY 3: Handle Edge-Cases Gracefully
        if len(df_filtered) == 0:
            # Handle empty filters by showing a warning and stopping execution for visualizations
            st.warning("⚠️ No companies match the current filter criteria. Please adjust your filters.")
        elif len(df_filtered) < 100:
            # Add warning for small data as requested by workplan
            st.warning(f"⚠️ Small Data Warning: Only {len(df_filtered)} rows selected. Clustering may be less accurate.")

        # Preserve Cluster column if it exists in session state
        if 'df_filtered' in st.session_state and 'Cluster' in st.session_state['df_filtered'].columns:
            # Merge the Cluster column back into the filtered data
            cluster_data = st.session_state['df_filtered'][['DUNS Number', 'Cluster']].drop_duplicates().copy()
            df_filtered = df_filtered.merge(cluster_data, on='DUNS Number', how='left')
        
        # Ensure cleaned + filtered data flows correctly
        st.session_state['df_filtered'] = df_filtered
        st.sidebar.metric("Filtered Companies", len(df_filtered))

    # Quick stats row
    st.markdown("---")
    cols = st.columns(4)
    cols[0].metric("Filtered Companies", len(st.session_state['df_filtered']))
    cols[1].metric("Clusters Found", len(st.session_state['df_filtered']['Cluster'].unique()) if 'Cluster' in st.session_state['df_filtered'].columns else 0)
    cols[2].metric("Avg Revenue", f"${st.session_state['df_filtered']['Revenue (USD)'].mean():,.0f}" if 'Revenue (USD)' in st.session_state['df_filtered'].columns else "N/A")
    cols[3].metric("Avg Employees", f"{st.session_state['df_filtered']['Employees Total'].mean():,.0f}" if 'Employees Total' in st.session_state['df_filtered'].columns else "N/A")
    st.markdown("---")

    # Show a preview of the first few rows
    st.subheader("Data Preview (Filtered)")
    st.dataframe(st.session_state['df_filtered'].head(10))

    # Reuse Person 1's download button
    csv_data = st.session_state['df_filtered'].to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Filtered Dataset",
        data=csv_data,
        file_name="champions_group_filtered.csv",
        mime="text/csv"
    )

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
                st.subheader("📊 Cluster Benchmarking & Distribution")
                
                # Always ensure derived metrics exist (Company Age, Revenue per Employee, etc.)
                df_vis = ensure_derived_metrics(df_vis)
                st.session_state['df_filtered'] = df_vis

                # Elbow Method Section (Optional - for finding optimal k)
                with st.expander("🔍 Advanced: Elbow Method (Find Optimal Clusters)", expanded=False):
                    st.caption("Use the elbow method to find the optimal number of clusters. Look for the 'elbow' where inertia reduction slows.")
                    if st.button("📊 Compute Elbow Chart", key="elbow_btn", type="secondary"):
                        with st.spinner("Computing elbow curve..."):
                            try:
                                from model import compute_elbow_inertia
                                
                                # Prepare data same as clustering
                                feature_cols = ['Revenue (USD)', 'Employees Total', 'IT Spend per Employee', 
                                                'Revenue per Employee', 'Tech Intensity Score', 'Company Age']
                                available_features = [col for col in feature_cols if col in df_vis.columns]
                                
                                if len(available_features) >= 2:
                                    X = df_vis[available_features].fillna(0)
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)
                                    
                                    # Compute elbow (k from 2 to 10)
                                    elbow_data = compute_elbow_inertia(X_scaled, k_min=2, k_max=10, random_state=42)
                                    
                                    # Plot elbow chart
                                    import plotly.graph_objects as go
                                    fig_elbow = go.Figure()
                                    fig_elbow.add_trace(go.Scatter(
                                        x=list(elbow_data.keys()),
                                        y=list(elbow_data.values()),
                                        mode='lines+markers',
                                        name='Inertia',
                                        line=dict(color='#4CAF50', width=3),
                                        marker=dict(size=10)
                                    ))
                                    fig_elbow.update_layout(
                                        title="Elbow Method: Finding Optimal Number of Clusters",
                                        xaxis_title="Number of Clusters (k)",
                                        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
                                        template="plotly_dark",
                                        height=400
                                    )
                                    st.plotly_chart(fig_elbow, width='stretch')
                                    
                                    # Calculate elbow point heuristic (simple: max second derivative)
                                    k_values = list(elbow_data.keys())
                                    inertias = list(elbow_data.values())
                                    if len(inertias) >= 3:
                                        # Calculate rate of change
                                        deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
                                        second_deltas = [deltas[i] - deltas[i+1] for i in range(len(deltas)-1)]
                                        optimal_k_idx = second_deltas.index(max(second_deltas)) + 2  # +2 because we start from k=2
                                        optimal_k = k_values[optimal_k_idx]
                                        st.success(f"💡 Suggested optimal k: **{optimal_k}** (based on maximum curvature)")
                                    else:
                                        st.info("Use the chart above to visually identify the 'elbow' point")
                                else:
                                    st.error(f"Not enough features for elbow analysis. Found: {available_features}")
                            except Exception as e:
                                st.error(f"Elbow computation failed: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                
                # Clustering button
                col_button, col_param = st.columns([1, 1])
                with col_param:
                    n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=4, step=1)
                with col_button:
                    st.write("")  # spacing
                    button_text = "Re-apply K-Means Clustering" if 'Cluster' in df_vis.columns else "Apply K-Means Clustering"
                    if st.button(button_text, type="primary"):
                        # Select numeric features for clustering
                        feature_cols = ['Revenue (USD)', 'Employees Total', 'IT Spend per Employee', 
                                        'Revenue per Employee', 'Tech Intensity Score', 'Company Age']
                        available_features = [col for col in feature_cols if col in df_vis.columns]
                        
                        # Create cache key to prevent redundant clustering
                        cache_key = (len(df_vis), tuple(sorted(available_features)), int(n_clusters), tuple(sorted(df_vis['DUNS Number'].unique())))
                        
                        # Check if we need to re-cluster
                        if st.session_state.get('cluster_cache_key') == cache_key:
                            st.info("✅ Using existing clusters (same data and parameters). Change filters or k to re-cluster.")
                        else:
                            try:
                                if len(available_features) >= 2:
                                    # Prepare data
                                    X = df_vis[available_features].fillna(0)
                                    
                                    # Standardize features
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)
                                    
                                    # Use model_day1.py clustering function
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
                                    st.session_state['cluster_cache_key'] = cache_key  # Save cache key
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
                    
                    # Top Anomalies / Watch List
                    st.markdown("---")
                    st.markdown("### 🚨 Top Anomalies / Watch List")
                    st.caption("Companies with unusual patterns compared to their cluster")
                    
                    with st.spinner("Analyzing anomalies..."):
                        anomalies = []
                        # Sample companies for speed - check first 50 or all if fewer
                        sample_size = min(50, len(df_vis))
                        sample_companies = df_vis['DUNS Number'].unique()[:sample_size]
                        
                        for duns in sample_companies:
                            try:
                                anomaly_df = compute_anomaly_score(df_vis, company_id=duns)
                                # Get high anomalies (z-score > 2 or < -2)
                                high_anomalies = anomaly_df[anomaly_df['Z-Score'].abs() > 2]
                                
                                if not high_anomalies.empty:
                                    avg_score = high_anomalies['Z-Score'].abs().mean()
                                    flags = high_anomalies['Metric'].tolist()
                                    
                                    # Get company name
                                    company_row = df_vis[df_vis['DUNS Number'] == duns].iloc[0]
                                    name = duns
                                    for col in ['Company Name', 'Company', 'Name', 'Company Sites']:
                                        if col in df_vis.columns:
                                            name = company_row[col]
                                            break
                                    
                                    anomalies.append((name, duns, avg_score, flags))
                            except:
                                continue
                        
                        if anomalies:
                            # Sort by score descending and take top 5
                            anomalies.sort(key=lambda x: x[2], reverse=True)
                            for name, duns, score, flags in anomalies[:5]:
                                st.markdown(f"{name} (DUNS: {duns}) – Anomaly Score: {score:.2f}")
                                st.caption(f"🔍 Unusual metrics: {', '.join(flags)}")
                        else:
                            st.info("✅ No significant anomalies detected in filtered data.")
                else:
                    st.info("Clustering not yet applied — summary table will appear here once clustering is applied.")

            with col_right:
                st.subheader("🔍 Selected Company vs Cluster Comparison")
                
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

                        # Company Information Table
                        st.subheader("📊 Company Information")
                        selected_row = df_vis[df_vis['DUNS Number'] == selected_duns].iloc[0]
                        
                        # Define important fields to display
                        important_fields = [
                            'Revenue (USD)',
                            'Employees Total',
                            'Company Age',
                            'Year Found',
                            'Revenue per Employee',
                            'IT Spend per Employee',
                            'Tech Intensity Score',
                            'Country',
                            'NAICS Description',
                            'Ownership Type',
                            'Entity Type',
                            'Cluster'
                        ]
                        
                        # Build table data with available fields
                        table_data = []
                        for field in important_fields:
                            if field in df_vis.columns:
                                value = selected_row[field]
                                # Format numeric values and ensure all values are strings
                                if field in ['Revenue (USD)', 'Employees Total', 'Year Found']:
                                    value = f"{value:,.0f}" if pd.notna(value) and value != 'N/A' else str(value)
                                elif field in ['Revenue per Employee', 'IT Spend per Employee', 'Company Age']:
                                    value = f"{value:,.2f}" if pd.notna(value) and value != 'N/A' else str(value)
                                elif field == 'Tech Intensity Score':
                                    value = f"{value:.3f}" if pd.notna(value) and value != 'N/A' else str(value)
                                else:
                                    # Convert all other values to string
                                    value = str(value) if pd.notna(value) else 'N/A'
                                
                                table_data.append({'Metric': field, 'Value': value})
                        
                        # Display as dataframe
                        if table_data:
                            info_df = pd.DataFrame(table_data)
                            st.dataframe(info_df, width='stretch', hide_index=True)

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
                                st.caption("📈 Normalized metrics (0–1 scale) – higher = better relative performance")
                            except Exception as e:
                                st.error(f"Failed to create radar chart: {str(e)}")
                            
                            # AI Business Insight Section
                            with st.expander("AI Business Insight", expanded=False):
                                # Initialize insights storage in session state
                                if 'ai_insights' not in st.session_state:
                                    st.session_state['ai_insights'] = {}
                                
                                # Check if insight already exists for this company
                                has_existing_insight = selected_duns in st.session_state['ai_insights']
                                
                                # Display existing insight if available
                                if has_existing_insight:
                                    insight_data = st.session_state['ai_insights'][selected_duns]
                                    st.markdown(insight_data['text'])
                                    st.caption(f"🕒 Generated on: {insight_data['timestamp']}")
                                    st.markdown("---")
                                    generate_button = st.button("🔄 Regenerate Insight", key=f"regenerate_{selected_duns}", type="secondary")
                                else:
                                    generate_button = st.button("✨ Generate Insight", key=f"generate_{selected_duns}", type="primary")
                                
                                # Generate or regenerate insight
                                if generate_button:
                                    with st.spinner("Analyzing company..."):
                                        try:
                                            # Get selected row as dict
                                            selected_row = df_vis[df_vis['DUNS Number'] == selected_duns].iloc[0]
                                            row_dict = selected_row.to_dict()

                                            # Get cluster average (re-compute profile or use existing)
                                            profile_df = build_cluster_profile(df_vis)
                                            cluster_value = selected_row['Cluster']  # Use cluster value as-is
                                            
                                            # Find matching cluster in profile
                                            cluster_match = profile_df[profile_df['Cluster'] == cluster_value]
                                            if cluster_match.empty:
                                                st.warning(f"Cluster {cluster_value} not found in profile data.")
                                            else:
                                                cluster_mean = cluster_match.iloc[0].to_dict()

                                                # Optional anomaly info
                                                try:
                                                    anomaly_df = compute_anomaly_score(df_vis, company_id=selected_duns)
                                                    # Get top anomalies (z-score > 2 or < -2)
                                                    high_anomalies = anomaly_df[anomaly_df['Z-Score'].abs() > 2]
                                                    if not high_anomalies.empty:
                                                        flags = high_anomalies['Metric'].tolist()
                                                        avg_score = high_anomalies['Z-Score'].abs().mean()
                                                        anomaly_text = f"Anomaly score: {avg_score:.1f} | Flags: {', '.join(flags)}"
                                                    else:
                                                        anomaly_text = "No anomalies detected."
                                                except:
                                                    anomaly_text = "Anomaly analysis unavailable."

                                                # Prompt (action-oriented, commercial focus)
                                                prompt = f"""
                                                    You are an expert business analyst specializing in firmographic and operational intelligence.
                                                    Company data: {row_dict}
                                                    Cluster average: {cluster_mean}
                                                    {anomaly_text}

                                                    Provide a concise 120–180 word professional summary:
                                                    - Typical company profile
                                                    - Key strengths and competitive advantages
                                                    - Potential risks or areas for concern
                                                    - How it compares to similar companies in its cluster

                                                    End with 1–2 actionable commercial recommendations for different stakeholders:
                                                    - Sales/upsell opportunity
                                                    - Investment/acquisition interest
                                                    - Risk management note

                                                    Tone: confident, data-driven, business-oriented. No hallucinations — stick to provided data.
                                                    """

                                                response = client.chat.completions.create(
                                                    model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768" for faster/cheaper
                                                    messages=[{"role": "user", "content": prompt}],
                                                    max_tokens=300,
                                                    temperature=0.6
                                                )

                                                summary = response.choices[0].message.content.strip()
                                                
                                                # Save insight with timestamp to session state
                                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                                st.session_state['ai_insights'][selected_duns] = {
                                                    'text': summary,
                                                    'timestamp': timestamp
                                                }
                                                
                                                st.rerun()
                                        except Exception as e:
                                            st.error(f"Insight generation failed: {str(e)}")
                                            st.caption("Check GROQ_API_KEY in secrets or try again.")
                        else:
                            st.info("Clustering not yet applied — radar chart will appear here once clusters exist.")
                else:
                    st.warning("No 'DUNS Number' column found — cannot select companies.")
else:
    st.sidebar.info("Upload data to enable filters.")