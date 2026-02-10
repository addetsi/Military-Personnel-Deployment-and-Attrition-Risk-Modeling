# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config
from deployment_simulator import DeploymentSimulator

# Page configuration
st.set_page_config(
    page_title="Military HR Analytics",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    """Load personnel data."""
    df = pd.read_csv(config.PROCESSED_DATA_DIR / 'features_engineered.csv')
    return df


@st.cache_resource
def initialize_simulator():
    """Initialize deployment simulator."""
    simulator = DeploymentSimulator(
        data_path=config.PROCESSED_DATA_DIR / 'features_engineered.csv',
        attrition_model_path=config.MODELS_DIR / 'attrition_classifier.pkl',
        readiness_model_path=config.MODELS_DIR / 'readiness_regressor.pkl',
        readiness_scaler_path=config.MODELS_DIR / 'readiness_scaler.pkl'
    )
    return simulator


def main():
    """Main dashboard application."""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Workforce Overview", "Attrition Analysis", "Readiness Assessment", "Deployment Simulator"]
    )
    
    # Load data
    df = load_data()
    
    # Global filters in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    
    branches = ['All'] + sorted(df['service_branch'].unique().tolist())
    selected_branch = st.sidebar.selectbox("Service Branch", branches)
    
    ranks = ['All'] + sorted(df['rank'].unique().tolist())
    selected_rank = st.sidebar.selectbox("Rank", ranks)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_branch != 'All':
        filtered_df = filtered_df[filtered_df['service_branch'] == selected_branch]
    if selected_rank != 'All':
        filtered_df = filtered_df[filtered_df['rank'] == selected_rank]
    
    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df)}")
    
    # Route to selected page
    if page == "Workforce Overview":
        page_workforce_overview(filtered_df)
    elif page == "Attrition Analysis":
        page_attrition_analysis(filtered_df)
    elif page == "Readiness Assessment":
        page_readiness_assessment(filtered_df)
    else:
        page_deployment_simulator(df)


def page_workforce_overview(df):
    """Page 1: Workforce Overview."""
    st.title("Workforce Overview")
    st.markdown("---")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_personnel = len(df)
    avg_readiness = df['readiness_score'].mean()
    high_risk_count = (df['attrition_risk'] == 'HIGH_RISK').sum()
    high_risk_pct = (high_risk_count / len(df)) * 100
    low_readiness = (df['readiness_score'] < 70).sum()
    
    col1.metric("Total Personnel", f"{total_personnel:,}")
    col2.metric("Avg Readiness", f"{avg_readiness:.1f}")
    col3.metric("High Risk", f"{high_risk_count}", f"{high_risk_pct:.1f}%")
    col4.metric("Low Readiness", f"{low_readiness}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personnel by Service Branch")
        branch_counts = df['service_branch'].value_counts()
        fig = px.bar(
            x=branch_counts.index,
            y=branch_counts.values,
            labels={'x': 'Branch', 'y': 'Count'},
            color=branch_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Personnel by Rank")
        rank_counts = df['rank'].value_counts()
        fig = px.pie(
            values=rank_counts.values,
            names=rank_counts.index,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition Risk Distribution")
        risk_counts = df['attrition_risk'].value_counts()
        colors = {'HIGH_RISK': '#e74c3c', 'MEDIUM_RISK': '#f39c12', 'LOW_RISK': '#2ecc71'}
        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            labels={'x': 'Risk Level', 'y': 'Count'},
            color=risk_counts.index,
            color_discrete_map=colors
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Readiness Score Distribution")
        fig = px.histogram(
            df,
            x='readiness_score',
            nbins=30,
            labels={'readiness_score': 'Readiness Score'},
            color_discrete_sequence=['#3498db']
        )
        fig.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="Target: 80")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def page_attrition_analysis(df):
    """Page 2: Attrition Analysis."""
    st.title("Attrition Risk Analysis")
    st.markdown("---")
    
    # High-risk personnel table
    st.subheader("High-Risk Personnel")
    high_risk_df = df[df['attrition_risk'] == 'HIGH_RISK'][
        ['age', 'rank', 'MOS', 'years_of_service', 'months_until_contract_end',
         'performance_review_score', 'health_index', 'civilian_job_offers']
    ].sort_values('months_until_contract_end')
    
    st.dataframe(
        high_risk_df.head(20),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = high_risk_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download High-Risk Personnel List",
        csv,
        "high_risk_personnel.csv",
        "text/csv"
    )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition Risk by Branch")
        risk_branch = pd.crosstab(df['service_branch'], df['attrition_risk'], normalize='index') * 100
        fig = px.bar(
            risk_branch,
            barmode='stack',
            labels={'value': 'Percentage (%)', 'service_branch': 'Branch'},
            color_discrete_map={'HIGH_RISK': '#e74c3c', 'MEDIUM_RISK': '#f39c12', 'LOW_RISK': '#2ecc71'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Attrition Risk by MOS")
        mos_risk = df.groupby('MOS')['attrition_risk'].apply(
            lambda x: (x == 'HIGH_RISK').sum() / len(x) * 100
        ).sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=mos_risk.values,
            y=mos_risk.index,
            orientation='h',
            labels={'x': 'High-Risk %', 'y': 'MOS'},
            color_discrete_sequence=['#e74c3c']
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contract Pressure Analysis")
        fig = px.box(
            df,
            x='attrition_risk',
            y='months_until_contract_end',
            color='attrition_risk',
            labels={'months_until_contract_end': 'Months Until Contract End'},
            color_discrete_map={'HIGH_RISK': '#e74c3c', 'MEDIUM_RISK': '#f39c12', 'LOW_RISK': '#2ecc71'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Performance by Risk Level")
        fig = px.violin(
            df,
            x='attrition_risk',
            y='performance_review_score',
            color='attrition_risk',
            labels={'performance_review_score': 'Performance Score'},
            color_discrete_map={'HIGH_RISK': '#e74c3c', 'MEDIUM_RISK': '#f39c12', 'LOW_RISK': '#2ecc71'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def page_readiness_assessment(df):
    """Page 3: Readiness Assessment."""
    st.title("Readiness Assessment")
    st.markdown("---")
    
    # Readiness metrics
    col1, col2, col3, col4 = st.columns(4)
    
    ready = (df['readiness_score'] >= 80).sum()
    limited = ((df['readiness_score'] >= 70) & (df['readiness_score'] < 80)).sum()
    not_ready = (df['readiness_score'] < 70).sum()
    avg_readiness = df['readiness_score'].mean()
    
    ready_pct = (ready / len(df)) * 100
    limited_pct = (limited / len(df)) * 100
    not_ready_pct = (not_ready / len(df)) * 100
    
    col1.metric("Ready (≥80)", f"{ready}", f"{ready_pct:.1f}%")
    col2.metric("Limited (70-79)", f"{limited}", f"{limited_pct:.1f}%")
    col3.metric("Not Ready (<70)", f"{not_ready}", f"{not_ready_pct:.1f}%")
    col4.metric("Average Score", f"{avg_readiness:.1f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Readiness by Branch")
        fig = px.box(
            df,
            x='service_branch',
            y='readiness_score',
            color='service_branch',
            labels={'readiness_score': 'Readiness Score', 'service_branch': 'Branch'}
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Target: 80")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Readiness by Rank")
        fig = px.violin(
            df,
            x='rank',
            y='readiness_score',
            color='rank',
            labels={'readiness_score': 'Readiness Score'}
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Low readiness alerts
    st.subheader("Low Readiness Alerts (<70)")
    low_readiness_df = df[df['readiness_score'] < 70][
        ['age', 'rank', 'MOS', 'years_of_service', 'readiness_score',
         'training_score_average', 'health_index', 'days_since_last_training']
    ].sort_values('readiness_score')
    
    st.dataframe(
        low_readiness_df.head(20),
        use_container_width=True,
        height=300
    )
    
    # Download button
    csv = low_readiness_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Low Readiness Personnel",
        csv,
        "low_readiness_personnel.csv",
        "text/csv"
    )
    
    st.markdown("---")
    
    # Heatmap: Readiness by Branch x Rank
    st.subheader("Readiness Heatmap: Branch × Rank")
    heatmap_data = df.pivot_table(
        values='readiness_score',
        index='rank',
        columns='service_branch',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Branch", y="Rank", color="Avg Readiness"),
        color_continuous_scale='RdYlGn',
        aspect="auto",
        text_auto='.1f'
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def page_deployment_simulator(df):
    """Page 4: Deployment Simulator."""
    st.title("Deployment Scenario Simulator")
    st.markdown("Simulate deployment scenarios to predict workforce impacts and optimize personnel selection.")
    st.markdown("---")
    
    # Initialize simulator
    simulator = initialize_simulator()
    
    # Input controls
    st.subheader("Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_personnel = st.number_input(
            "Personnel Needed",
            min_value=1,
            max_value=200,
            value=50,
            step=5
        )
        
        duration = st.slider(
            "Duration (months)",
            min_value=3,
            max_value=12,
            value=6,
            step=1
        )
    
    with col2:
        min_readiness = st.slider(
            "Minimum Readiness Score",
            min_value=60,
            max_value=95,
            value=80,
            step=5
        )
        
        strategy = st.selectbox(
            "Selection Strategy",
            ["balanced", "readiness", "low_risk"],
            format_func=lambda x: x.title()
        )
    
    with col3:
        all_mos = sorted(df['MOS'].unique().tolist())
        selected_mos = st.multiselect(
            "MOS Requirements (optional)",
            all_mos,
            default=None
        )
        
        allow_high_risk = st.checkbox("Allow HIGH_RISK personnel", value=False)
    
    # Run simulation button
    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            results = simulator.simulate_deployment(
                n_personnel=n_personnel,
                duration_months=duration,
                min_readiness=min_readiness,
                mos_requirements=selected_mos if selected_mos else None,
                allow_high_risk=allow_high_risk,
                strategy=strategy
            )
        
        st.success("Simulation complete!")
        
        # Display results
        st.markdown("---")
        st.subheader("Simulation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Selected Personnel", results['n_selected'])
        col2.metric("Avg Readiness", f"{results['avg_readiness']:.1f}")
        col3.metric("Expected Attrition", 
                   f"{results['expected_attrition_count']} ({results['expected_attrition_rate']:.1f}%)")
        col4.metric("Replacements Needed", results['replacement_needed'])
        
        # Risk distribution
        st.markdown("---")
        st.subheader("Risk Distribution")
        risk_dist = results['risk_distribution']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=list(risk_dist.values()),
                names=list(risk_dist.keys()),
                color=list(risk_dist.keys()),
                color_discrete_map={'HIGH_RISK': '#e74c3c', 'MEDIUM_RISK': '#f39c12', 'LOW_RISK': '#2ecc71'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Risk Breakdown:**")
            for risk, count in risk_dist.items():
                pct = (count / results['n_selected']) * 100 if results['n_selected'] > 0 else 0
                st.write(f"{risk}: {count} ({pct:.1f}%)")
        
        # Selected personnel table
        st.markdown("---")
        st.subheader("Selected Personnel")
        selected_df = pd.DataFrame(results['selected_personnel'])
        
        if len(selected_df) > 0:
            st.dataframe(selected_df, use_container_width=True, height=400)
            
            # Download button
            csv = selected_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Deployment List",
                csv,
                f"deployment_list_{n_personnel}pax_{duration}mo.csv",
                "text/csv"
            )
        else:
            st.warning("No personnel selected. Try adjusting the parameters.")
        
        # Alternative scenarios
        if results['alternatives']:
            st.markdown("---")
            st.subheader("Alternative Scenarios")
            
            for i, alt in enumerate(results['alternatives'], 1):
                with st.expander(f"Alternative {i}: {alt['scenario']}"):
                    for key, value in alt.items():
                        if key != 'scenario':
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")


if __name__ == "__main__":
    main()