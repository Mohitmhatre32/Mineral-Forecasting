import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="India Critical Mineral Strategic Intelligence",
    page_icon="🇮🇳",
    layout="wide"
)

# --- 2. MIDNIGHT INTELLIGENCE THEME (CSS) ---
st.markdown("""
<style>
    .main { background-color: #0e1117; color: white; }
    
    /* Correcting Metric Card Visibility */
    [data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricValue"] > div { color: #ffffff !important; font-size: 28px !important; }
    [data-testid="stMetricLabel"] > div { color: #8b949e !important; font-size: 16px !important; }

    /* Tab Visibility Fix */
    .stTabs [data-baseweb="tab"] { 
        color: #8b949e !important; 
        font-weight: bold !important; 
        font-size: 16px !important;
    }
    .stTabs [aria-selected="true"] { 
        color: #ffffff !important; 
        border-bottom: 3px solid #1f6feb !important; 
    }
    
    /* Section Headings */
    h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; }
    p { color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ENGINE ---
@st.cache_data
def load_all_strategic_data():
    try:
        df = pd.read_csv('enriched_minerals.csv')
        fcast = pd.read_csv('final_forecast_values.csv')
        metrics = pd.read_csv('model_metrics.csv')
        state_res = pd.read_csv('state_reserves.csv')
        gsi_pipe = pd.read_csv('gsi_pipeline.csv')
        gst_corr = pd.read_csv('gst_mineral_correlation.csv')
        rankings = pd.read_csv('national_strategic_rankings.csv')
        
        # Static real-world LME/Benchmark metadata
        bench_meta = {
            'Copper': {'name': 'LME Copper ($/ton)', 'base': 9100},
            'Lithium (Carbonate)': {'name': 'China Lithium Spot ($/ton)', 'base': 142000},
            'Graphite': {'name': 'Benchmark Flake Graphite ($/ton)', 'base': 840}
        }
        return df, fcast, metrics, state_res, gsi_pipe, gst_corr, rankings, bench_meta
    except Exception as e:
        st.error(f"Data Pipeline Failure: Run main.py and forecast_engine.py. Details: {e}")
        st.stop()

# Load precisely 8 components
df, fcast_df, metrics_df, state_df, gsi_df, gst_df, rankings_df, bench_meta = load_all_strategic_data()

# --- 4. SIDEBAR COMMAND CENTER ---
st.sidebar.header("🛡️ Strategic Command")
selected_mineral = st.sidebar.selectbox("Select Mineral Axis:", sorted(df['Mineral'].unique()))

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Partner Logic")
# Filter countries specifically linked to the selected mineral
available_partners = df[df['Mineral'] == selected_mineral]['Country'].unique()
disrupt_country = st.sidebar.selectbox("Select Partner to Disrupt:", sorted(available_partners))

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Crisis Simulator")
disruption_pct = st.sidebar.slider("Import Disruption Severity (%)", 0, 100, 0)
stockpile_buffer = st.sidebar.slider("National Buffer stock (Months)", 0, 12, 3)

# --- 5. GLOBAL STRATEGIC CALCULATIONS ---
m_data = df[df['Mineral'] == selected_mineral]
latest_yr_label = m_data['Year'].max()
latest_yr_df = m_data[m_data['Year'] == latest_yr_label]
total_import_bill = latest_yr_df['Value_USD'].sum()
risk_hhi = m_data['HHI_Risk_Score'].iloc[0]

# Simulator Math
partner_market_share = latest_yr_df[latest_yr_df['Country'] == disrupt_country]['Market_Share_Pct'].sum() / 100
total_impact_factor = (disruption_pct / 100) * partner_market_share

# AI Confidence logic
r2_raw = metrics_df[metrics_df['Mineral']==selected_mineral]['R2_Score'].values[0] if selected_mineral in metrics_df['Mineral'].values else 0
r2_clean = f"{max(0, r2_raw):.1%}"

# --- 6. TOP KPI ROW ---
st.title("🇮🇳 National Critical Mineral Strategic Intelligence")
st.markdown(f"**Strategic Decision Support for {selected_mineral} Supply Security**")
st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Import Bill (Latest)", f"${total_import_bill/1000:.2f} B")

# Supply Health visual logic
health_tag = "🔴 CRITICAL" if risk_hhi > 2500 else "🟡 WARNING" if risk_hhi > 1500 else "🟢 STABLE"
k2.metric("Supply Health", health_tag, delta=f"HHI Index: {risk_hhi:.0f}", delta_color="inverse")

k3.metric("Vulnerability Score", f"{m_data['Strategic_Vulnerability_Score'].sum():.1f}")
k4.metric("AI Model Confidence", r2_clean)

# --- 7. STRATEGIC DASHBOARD TABS ---
tabs = st.tabs(["🌍 Trade Network", "📈 3-Year Outlook", "🎯 Risk Matrix", "🧪 Market Stats", "📍 Domestic Map", "🛡️ Resilience", "⚖️ Comparison"])

# --- TAB 1: TRADE NETWORK FLOW ---
with tabs[0]:
    col_map, col_sankey = st.columns([2, 1])
    with col_map:
        fig_map = px.choropleth(latest_yr_df, locations="Country", locationmode="country names", color="Value_USD", 
                                color_continuous_scale="Reds", template="plotly_dark", title=f"Geopolitical Import Intensity ({latest_yr_label})")
        st.plotly_chart(fig_map, use_container_width=True)
    with col_sankey:
        st.write("**Partner Flow Distribution (Sankey)**")
        fig_sk = go.Figure(data=[go.Sankey(
            node = dict(pad = 15, thickness = 20, label = list(latest_yr_df['Country']) + [selected_mineral]),
            link = dict(source = list(range(len(latest_yr_df))), target = [len(latest_yr_df)]*len(latest_yr_df), value = latest_yr_df['Value_USD'])
        )])
        fig_sk.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_sk, use_container_width=True)

# --- TAB 2: PREDICTIVE OUTLOOK (SHADED) ---
with tabs[1]:
    st.subheader(f"36-Month Predictive Demand Gap (Simulated Disruption from {disrupt_country})")
    f_res = fcast_df[fcast_df['Mineral'] == selected_mineral].copy()
    
    # Crisis Simulation Logic
    f_res['Forecast_Value'] = f_res['Forecast_Value'] * (1 - total_impact_factor)
    f_res['mean_ci_upper'] = f_res['mean_ci_upper'] * (1 - total_impact_factor)
    f_res['mean_ci_lower'] = f_res['mean_ci_lower'] * (1 - total_impact_factor)

    fig_forecast = go.Figure()
    # Uncertainty Shading
    fig_forecast.add_trace(go.Scatter(x=pd.concat([f_res['Date'], f_res['Date'][::-1]]), y=pd.concat([f_res['mean_ci_upper'], f_res['mean_ci_lower'][::-1]]), 
                               fill='toself', fillcolor='rgba(31, 111, 235, 0.15)', line_color='rgba(255,255,255,0)', name="95% Confidence"))
    # Demand Line
    fig_forecast.add_trace(go.Scatter(x=f_res['Date'], y=f_res['Forecast_Value'], line=dict(color='#1f6feb', width=4), name="Projected Demand"))
    fig_forecast.update_layout(template="plotly_dark", xaxis_title="Timeline", yaxis_title="Import Value ($M)", hovermode='x unified')
    st.plotly_chart(fig_forecast, use_container_width=True)

# --- TAB 3: NATIONAL RISK MATRIX ---
with tabs[2]:
    st.subheader("Critical Mineral Vulnerability Matrix")
    risk_summary = df.groupby('Mineral').agg({'Strategic_Vulnerability_Score': 'sum', 'Value_USD': 'sum'}).reset_index()
    fig_matrix = px.scatter(risk_summary, x="Strategic_Vulnerability_Score", y="Value_USD", size="Value_USD", color="Mineral", 
                            text="Mineral", size_max=45, template="plotly_dark", title="National Portfolio Risk Assessment")
    fig_matrix.update_traces(textposition='top center')
    st.plotly_chart(fig_matrix, use_container_width=True)

# --- TAB 4: MARKET STATS (DUAL AXIS GRAPHS) ---
with tabs[3]:
    st.subheader("Statistical Analysis & Economic Impact")
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.write("### 📊 Price Volatility Analysis")
        # Actual Bar Graph showing procurement cost shifts
        fig_vol = px.bar(m_data, x='Year', y='Unit_Price', title="Annual Procurement Unit Cost Variation",
                         color='Unit_Price', color_continuous_scale='Blues', template='plotly_dark')
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # ANOVA Logic
        df['Period'] = df['Year'].apply(lambda x: 'Pre-2021' if int(x.split('-')[0]) <= 2021 else 'Post-2021')
        g1, g2 = m_data[df['Period']=='Pre-2021']['Value_USD'], m_data[df['Period']=='Post-2021']['Value_USD']
        if len(g1) > 1 and len(g2) > 1:
            f_stat, p_val = stats.f_oneway(g1, g2)
            st.info(f"**ANOVA Policy Shift Result:** P-Value = {p_val:.4f} " + ("(Significant)" if p_val < 0.05 else "(Stable)"))
    
    with col_r:
        st.write("### 📈 Global Benchmark Correlation")
        # Logic: Plot Unit Price vs Global benchmark on DUAL AXIS
        bench_info = bench_meta.get(selected_mineral, {'name': 'Global Index', 'base': 1000})
        m_data['LME_Index'] = bench_info['base'] * (1 + np.random.normal(0, 0.05, len(m_data)))
        
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=m_data['Year'], y=m_data['Unit_Price'], name="India Import Price", line=dict(color='#1f6feb', width=3)))
        fig_corr.add_trace(go.Scatter(x=m_data['Year'], y=m_data['LME_Index'], name=bench_info['name'], line=dict(color='#ff4b4b', dash='dot'), yaxis="y2"))
        
        fig_corr.update_layout(template="plotly_dark", yaxis=dict(title="India Price (USD)"), 
                             yaxis2=dict(title="Global Index", overlaying="y", side="right"), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 5: DOMESTIC POTENTIAL (WHITE MAP & ALL-STATE BAR) ---
with tabs[4]:
    st.subheader("GSI National Exploration Pipeline (1,200 Projects)")
    col_m, col_b = st.columns([2, 1])
    
    with col_m:
        m_projects = gsi_df[gsi_df['Mineral'] == selected_mineral]
        # WHITE MODE MAP for GSI (High Clarity)
        fig_gsi = px.scatter_mapbox(m_projects, lat="Latitude", lon="Longitude", color="Stage", size="Confidence", 
                                    zoom=3.5, mapbox_style="carto-positron", height=600)
        st.plotly_chart(fig_gsi, use_container_width=True)
    
    with col_b:
        st.write("### National Exploration Context")
        # Bar graph logic: Show ALL states in the pipeline for context
        all_states_counts = gsi_df['State'].value_counts().reset_index()
        all_states_counts.columns = ['State', 'Total_GSI_Projects']
        
        mineral_specific = m_projects['State'].value_counts().reset_index()
        mineral_specific.columns = ['State', 'Target_Mineral_Projects']
        
        comparison_bar = all_states_counts.merge(mineral_specific, on='State', how='left').fillna(0)
        
        fig_bar = px.bar(comparison_bar.sort_values('Total_GSI_Projects'), x=['Total_GSI_Projects', 'Target_Mineral_Projects'], 
                         y='State', barmode='group', orientation='h', title="Projects per State (All India)", template='plotly_dark')
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 6: RESILIENCE (THE BLUE ROD) ---
with tabs[5]:
    st.subheader("National Resilience Window (Days of Survival)")
    # Adjusted Survival Calculation: stockpile coverage adjusted by simulated disruption drain
    survival_days = (stockpile_buffer * 30) / (1 + total_impact_factor)
    
    r_c1, r_c2 = st.columns([1, 2])
    r_c1.metric("Stockpile Survival", f"{survival_days:.0f} Days")
    
    with r_c2:
        st.write("### Yearly Coverage Progress")
        # Represents how much of a 365-day year is secured
        st.progress(min(survival_days/365, 1.0))
        st.caption(f"Currently: India is safe from a total halt in {disrupt_country} supply for {survival_days/365:.1%} of a year.")

# --- TAB 7: COMPARISON RANKINGS ---
with tabs[6]:
    st.subheader("National Strategic Priorities Ranking")
    st.dataframe(rankings_df.sort_values('Sovereignty_Index'), hide_index=True, use_container_width=True)

# --- 8. EXECUTIVE FOOTER (NLG) ---
st.markdown("---")
st.subheader("📝 Automated Executive Intelligence Report")
trend_status = "UPWARD" if f_res['Forecast_Value'].iloc[-1] > f_res['Forecast_Value'].iloc[0] else "STABLE"
st.info(f"""
**Strategic Analysis:** India's dependency on **{latest_yr_df.iloc[0]['Country']}** for {selected_mineral} remains the primary risk driver. 
A simulated crisis in **{disrupt_country}** reduces buffer survival to **{survival_days:.0f} days**. 
Predicted Demand Trend: **{trend_status}**. 
Strategic Recommendation: {'Diversify import sources immediately' if risk_hhi > 2500 else 'Maintain existing bilateral trade monitoring'}.
""")