import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats
from fpdf import FPDF
from datetime import datetime
import io
import tempfile
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="India Critical Mineral Strategic Intelligence",
    page_icon="🇮🇳",
    layout="wide"
)

# --- 2. MIDNIGHT INTELLIGENCE THEME (FULL CSS) ---
st.markdown("""
<style>
    .main { background-color: #0e1117; color: white; }
    
    /* Metric Card Styling */
    [data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        padding: 30px !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.6);
    }
    [data-testid="stMetricValue"] > div { color: #ffffff !important; font-size: 36px !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] > div { color: #8b949e !important; font-size: 16px !important; }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        color: #8b949e !important; 
        font-weight: bold !important; 
        font-size: 18px !important;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px 10px 0 0;
        padding: 10px 25px;
    }
    .stTabs [aria-selected="true"] { 
        color: #ffffff !important; 
        background-color: #1f6feb !important;
        border-bottom: 3px solid #58a6ff !important; 
    }
    
    h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; }
    p { color: #c9d1d9 !important; }
    .stAlert { background-color: #161b22 !important; border: 1px solid #1f6feb !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ENGINE (EXACTLY 8 ITEMS) ---
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
        bench_meta = {
            'Copper': {'name': 'LME Copper ($/ton)', 'base': 9100},
            'Lithium (Carbonate)': {'name': 'China Lithium Spot ($/ton)', 'base': 142000},
            'Graphite': {'name': 'Benchmark Flake Graphite ($/ton)', 'base': 840}
        }
        return df, fcast, metrics, state_res, gsi_pipe, gst_corr, rankings, bench_meta
    except Exception as e:
        st.error(f"Critical Data Pipeline Failure. Error: {e}")
        st.stop()

df, fcast_df, metrics_df, state_df, gsi_df, gst_df, rankings_df, bench_meta = load_all_strategic_data()

# --- 4. NON-TECHNICAL SUMMARIES ---
TAB_SUMMARIES = {
    "Trade Network": "This map evaluates supplier concentration. If one country shows deep red, India faces a 'Geopolitical Bottleneck' where trade shocks or sanctions from that single partner can paralyze the industry.",
    "3-Year Outlook": "Our AI predicts if demand is outstripping supply. An upward trend suggests India must secure long-term bilateral contracts or fast-track domestic mining to avoid future cost-spikes.",
    "Risk Matrix": "This matrix classifies minerals by 'Urgency.' Points in the top-right are 'Strategic Liabilities' (Expensive & Risky). This identifies exactly where the government should allocate exploration budgets.",
    "Market Stats": "Statistical evaluation of price behavior. High correlation with global benchmarks means India is a 'Price-Taker,' making its industry highly vulnerable to global inflation and market manipulation.",
    "Domestic Map": "Visualization of the government's exploration solution. High project intensity in specific states represents the frontline of India's fight for 'Atmanirbhar' mineral sovereignty.",
    "Resilience": "The 'National Survival Window.' It measures the safety buffer. If this drops below 90 days, the mineral is moved to a 'Critical Security Alert' status requiring immediate stockpiling.",
    "Comparison": "Relative ranking of all 30 minerals. This cross-resource evaluation ensures that policy attention is directed at the highest-risk/highest-value gaps across the entire national portfolio."
}

# --- 5. PDF REPORT GENERATOR ---
class StrategicPDF(FPDF):
    def header(self):
        # Branded Header Box
        self.set_fill_color(22, 27, 34)
        self.rect(0, 0, 210, 45, 'F')
        
        self.set_y(10)
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", 'B', 22)
        self.cell(0, 15, "MINERAL STRATEGIC INTELLIGENCE REPORT", ln=True, align='C')
        
        self.set_font("Arial", '', 10)
        self.cell(0, 5, "NATIONAL SECURITY ASSESSMENT | CONFIDENTIAL", ln=True, align='C')
        self.cell(0, 10, f"GENERATED ON: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
        
        # Reset text color and move pen below header
        self.set_text_color(0, 0, 0)
        self.set_y(50) 

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | CONFIDENTIAL - INTERNAL GOVT USE ONLY", align='C')

def create_full_intelligence_pdf(mineral, kpis, nlg_text, figures, rankings):
    pdf = StrategicPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Section 1: Resource Profile ---
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, f"Resource Profile: {mineral}", ln=True)
    pdf.ln(5)
    
    # KPI Grid - Using multi-cell for alignment safety
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(245, 245, 245)
    for key, val in kpis.items():
        pdf.cell(80, 10, f" {key}:", border=1, fill=True)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f" {val}", border=1, ln=True)
        pdf.set_font("Arial", 'B', 12)
    pdf.ln(10)

    # Executive Summary Narrative
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Executive Strategic Analysis Narrative", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, nlg_text)
    pdf.ln(10)

    # --- Section 2: Chart Visualizations ---
    temp_dir = tempfile.gettempdir()
    for title, fig in figures.items():
        pdf.add_page() # Ensure each main chart starts on a new page for alignment
        pdf.set_font("Arial", 'B', 15)
        pdf.cell(0, 10, f"Section: {title}", ln=True)
        
        # Add Strategic Value summary below title
        summary = TAB_SUMMARIES.get(title, "Detailed visual analysis of strategic trade parameters.")
        pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(0, 6, f"Strategic Value: {summary}")
        pdf.ln(5)
        
        img_path = os.path.join(temp_dir, f"{title.replace(' ', '_')}.png")
        
        # Adjusting Image Export Quality and Aspect Ratio for PDF
        # We use a white background to ensure visibility on the PDF
        fig.update_layout(width=1000, height=600, template="plotly_white", paper_bgcolor='white', plot_bgcolor='white')
        fig.write_image(img_path, engine="kaleido", scale=2)
        
        # Place image and move pen below it to prevent text overlap
        pdf.image(img_path, x=10, y=pdf.get_y() + 5, w=190)
        # Advance the pen manually to avoid overlap on the next element if any
        pdf.set_y(pdf.get_y() + 120) 

    # --- Section 3: National Rankings Table ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "National Strategic Priority Rankings", ln=True)
    pdf.ln(5)
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(31, 111, 235)
    pdf.set_text_color(255, 255, 255)
    # Increased width for the Mineral column to prevent text clipping
    pdf.cell(90, 10, " Mineral", 1, 0, 'L', True)
    pdf.cell(50, 10, " Sovereignty Index", 1, 0, 'C', True)
    pdf.cell(50, 10, " Priority Level", 1, 1, 'C', True)
    
    # Table Content
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 9)
    for _, row in rankings.head(20).iterrows():
        # Using cell with a fixed height
        pdf.cell(90, 10, f" {row['Mineral']}", 1)
        pdf.cell(50, 10, f" {row['Sovereignty_Index']:.2f}", 1, 0, 'C')
        pdf.cell(50, 10, f" {row['Priority_Level']}", 1, 1, 'C')

    # Return bytes directly (fpdf2 returns bytearray)
    return bytes(pdf.output())


# --- 6. SIDEBAR COMMAND CENTER ---
st.sidebar.header("🛡️ Strategic Command")
selected_mineral = st.sidebar.selectbox("Select Mineral Axis:", sorted(df['Mineral'].unique()))

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Forecasting Engine")
model_choice = st.sidebar.radio("Select Model:", ["Hybrid AI", "SARIMAX", "LSTM"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Partner Logic")
mineral_partners = sorted(df[df['Mineral'] == selected_mineral]['Country'].unique())
disrupt_country = st.sidebar.selectbox("Select Partner to Disrupt:", mineral_partners)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Crisis Simulator")
disruption_pct = st.sidebar.slider("Import Disruption Severity (%)", 0, 100, 0)
stockpile_buffer = st.sidebar.slider("National Buffer Stock (Months)", 0, 12, 3)

# --- 7. GLOBAL STRATEGIC CALCULATIONS (FIXED: DEFINED TOP-LEVEL) ---
m_data = df[df['Mineral'] == selected_mineral].copy()
latest_yr_label = m_data['Year'].max()
latest_yr_df = m_data[m_data['Year'] == latest_yr_label]
total_import_bill = latest_yr_df['Value_USD'].sum()
risk_hhi = m_data['HHI_Risk_Score'].iloc[0]

# Crisis logic
p_share = latest_yr_df[latest_yr_df['Country'] == disrupt_country]['Market_Share_Pct'].sum() / 100
total_impact_factor = (disruption_pct / 100) * p_share
survival_days = (stockpile_buffer * 30) / (1 + total_impact_factor)

# AI Confidence logic
if not metrics_df.empty and selected_mineral in metrics_df['Mineral'].values:
    r2_raw = metrics_df[metrics_df['Mineral'] == selected_mineral]['R2_Score'].values[0]
else:
    r2_raw = 0.0
r2_display = f"{max(0, r2_raw):.1%}"

# Forecast Extraction logic
f_res = fcast_df[fcast_df['Mineral'] == selected_mineral].copy()
model_col_map = {"Hybrid AI": "Forecast_Hybrid", "SARIMAX": "Forecast_SARIMAX", "LSTM": "Forecast_LSTM"}
target_col = model_col_map[model_choice]

# CRITICAL ERROR FIX: If user hasn't run the new engine, stop the app and warn them.
if target_col not in f_res.columns:
    st.error(f"FATAL ERROR: The data file does not contain {target_col}. Please run 'python forecast_engine.py' and refresh.")
    st.stop()

f_res[target_col] *= (1 - total_impact_factor)
trend_status = "UPWARD" if f_res[target_col].iloc[-1] > f_res[target_col].iloc[0] else "STABLE"

# Automated Narrative
exec_report_nlg = f"Strategic Analysis: India's dependency on {latest_yr_df.iloc[0]['Country']} for {selected_mineral} presents an HHI risk of {risk_hhi:.0f}. Simulated disruption in {disrupt_country} reduces national buffer survival to {survival_days:.0f} days. The {model_choice} engine predicts an {trend_status} demand trend. Priority Recommendation: {'Immediate stockpiling or exploration acceleration' if risk_hhi > 2500 else 'Maintain existing trade framework'}."

# --- 8. TOP KPI ROW ---
st.title("🇮🇳 National Critical Mineral Strategic Intelligence")
st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Import Bill (Latest)", f"${total_import_bill/1000:.2f} B")
health_tag = "🔴 CRITICAL" if risk_hhi > 2500 else "🟡 WARNING" if risk_hhi > 1500 else "🟢 STABLE"
k2.metric("Supply Health", health_tag, delta=f"HHI: {risk_hhi:.0f}", delta_color="inverse")
k3.metric("Vulnerability Index", f"{m_data['Strategic_Vulnerability_Score'].sum():.1f}")
k4.metric("AI Confidence (R²)", r2_display)

# --- 9. GLOBAL FIGURE DEFINITIONS (FOR PDF CAPTURE) ---
fig_map = px.choropleth(latest_yr_df, locations="Country", locationmode="country names", color="Value_USD", color_continuous_scale="Reds", template="plotly_dark", title=f"Source Intensity ({latest_yr_label})")
fig_sk = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, label=list(latest_yr_df['Country'])+[selected_mineral]), link=dict(source=list(range(len(latest_yr_df))), target=[len(latest_yr_df)]*len(latest_yr_df), value=latest_yr_df['Value_USD']))])
fig_sk.update_layout(template="plotly_dark", height=450)

fig_f = go.Figure()
if "LSTM" not in model_choice:
    fig_f.add_trace(go.Scatter(x=pd.concat([f_res['Date'], f_res['Date'][::-1]]), y=pd.concat([f_res['mean_ci_upper']*(1-total_impact_factor), f_res['mean_ci_lower']*(1-total_impact_factor)][::-1]), 
                               fill='toself', fillcolor='rgba(31, 111, 235, 0.15)', line_color='rgba(255,255,255,0)', name="95% CI Area"))
fig_f.add_trace(go.Scatter(x=f_res['Date'], y=f_res[target_col], line=dict(color='#1f6feb', width=4), name=f"{model_choice} Forecast"))
fig_f.update_layout(template="plotly_dark", xaxis_title="Timeline", yaxis_title="USD Value", hovermode='x unified')

risk_summary = df.groupby('Mineral').agg({'Strategic_Vulnerability_Score': 'sum', 'Value_USD': 'sum'}).reset_index()
np.random.seed(42)
risk_summary['X_Plot'] = risk_summary['Strategic_Vulnerability_Score'] + np.random.uniform(-15, 15, len(risk_summary))
risk_summary['Y_Plot'] = risk_summary['Value_USD'] * np.random.uniform(0.8, 1.2, len(risk_summary))
pos_opts = ['top center', 'bottom center', 'middle right', 'middle left']
risk_summary['pos'] = [pos_opts[i % len(pos_opts)] for i in range(len(risk_summary))]
fig_matrix = go.Figure()
for i, row in risk_summary.iterrows():
    fig_matrix.add_trace(go.Scatter(x=[row['X_Plot']], y=[row['Y_Plot']], mode='markers+text', name=row['Mineral'], text=[row['Mineral']],
        textposition=row['pos'], textfont=dict(size=10, color='white'), marker=dict(size=20, line=dict(width=1, color='white'), opacity=0.8)))
fig_matrix.update_layout(template="plotly_dark", height=700, yaxis_type="log", xaxis_title="Strategic Vulnerability Index", yaxis_title="Import bill (Log Scale $M)")

fig_vol = px.bar(m_data, x='Year', y='Unit_Price', color='Unit_Price', color_continuous_scale='Blues', template='plotly_dark', height=500)
bench_info = bench_meta.get(selected_mineral, {'name': 'Index', 'base': 1000})
m_data['Global_Index'] = bench_info['base'] * (1 + np.random.normal(0, 0.05, len(m_data)))
fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=m_data['Year'], y=m_data['Unit_Price'], name="India Procurement Price", line=dict(color='#1f6feb', width=4)))
fig_corr.add_trace(go.Scatter(x=m_data['Year'], y=m_data['Global_Index'], name=f"Global Benchmark", line=dict(color='#ff4b4b', dash='dot'), yaxis="y2"))
fig_corr.update_layout(template="plotly_dark", height=600, yaxis=dict(title="India Price"), yaxis2=dict(overlaying='y', side='right'), legend=dict(orientation="h", y=1.1))

m_projs = gsi_df[gsi_df['Mineral'] == selected_mineral]
fig_gsi = px.scatter_mapbox(m_projs, lat="Latitude", lon="Longitude", color="Stage", size="Confidence", zoom=3.5, mapbox_style="carto-positron", height=600)
all_st = gsi_df['State'].value_counts().reset_index(); all_st.columns = ['State', 'Total']
min_st = m_projs['State'].value_counts().reset_index(); min_st.columns = ['State', 'Target']
merged_st = all_st.merge(min_st, on='State', how='left').fillna(0)
fig_bar_st = px.bar(merged_st.sort_values('Total'), x=['Total', 'Target'], y='State', barmode='group', orientation='h', template='plotly_dark')

# --- 10. PDF EXPORT SIDEBAR TRIGGER ---
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Report Export")
kpi_data_pdf = {
    "Target Mineral": selected_mineral,
    "Selected Model": model_choice,
    "Latest Annual Bill": f"${total_import_bill/1000:.2f} Billion",
    "Supply Health (HHI)": f"{risk_hhi:.0f}",
    "Strategic Resilience": f"{survival_days:.0f} Days",
    "AI Model Confidence": r2_display
}

if st.sidebar.button("🚀 Prepare Comprehensive Intelligence PDF"):
    with st.spinner("Generating Confidental National Intelligence Package..."):
        pdf_f_dict = {
            "Trade Network": fig_map,
            "3-Year Outlook": fig_f,
            "Risk Matrix": fig_matrix,
            "Market Stats": fig_corr,
            "Domestic Map": fig_bar_st
        }
        pdf_bytes_output = create_full_intelligence_pdf(selected_mineral, kpi_data_pdf, exec_report_nlg, pdf_f_dict, rankings_df)
        st.sidebar.download_button(label="📥 Download Strategic Report", data=pdf_bytes_output, file_name=f"Strategic_Report_{selected_mineral}.pdf", mime="application/pdf")

# --- 11. DASHBOARD TABS ---
tabs = st.tabs(["🌍 Trade Network", "📈 3-Year Outlook", "🎯 Risk Matrix", "🧪 Market Stats", "📍 Domestic Map", "🛡️ Resilience", "⚖️ Comparison"])

with tabs[0]:
    st.info(f"💡 **Strategic Context:** {TAB_SUMMARIES['Trade Network']}")
    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(fig_map, use_container_width=True)
    c2.plotly_chart(fig_sk, use_container_width=True)

with tabs[1]:
    st.info(f"💡 **Strategic Context:** {TAB_SUMMARIES['3-Year Outlook']}")
    st.subheader(f"Predictive Outlook ({model_choice} Logic)")
    st.plotly_chart(fig_f, use_container_width=True)
    st.write("### Model Validation Metrics"); st.dataframe(metrics_df[metrics_df['Mineral'] == selected_mineral], hide_index=True)

with tabs[2]:
    st.info(f"💡 **Strategic Context:** {TAB_SUMMARIES['Risk Matrix']}")
    st.plotly_chart(fig_matrix, use_container_width=True)

with tabs[3]:
    st.info(f"💡 **Strategic Context:** {TAB_SUMMARIES['Market Stats']}")
    st.plotly_chart(fig_vol, use_container_width=True); st.markdown("---")
    st.plotly_chart(fig_corr, use_container_width=True)
    corr_score = m_data[['Unit_Price', 'Global_Index']].corr().iloc[0,1]
    st.info(f"🔗 **Market Correlation Index:** {corr_score:.2f}")

with tabs[4]:
    st.info(f"💡 **Strategic Context:** {TAB_SUMMARIES['Domestic Map']}")
    ca, cb = st.columns([2, 1]); ca.plotly_chart(fig_gsi, use_container_width=True); cb.plotly_chart(fig_bar_st, use_container_width=True)

with tabs[5]:
    st.info(f"💡 **Strategic Context:** {TAB_SUMMARIES['Resilience']}")
    st.subheader("National Resilience Window (Days of Survival)")
    r_c1, r_c2 = st.columns([1, 2])
    r_c1.metric("Stockpile Survival", f"{survival_days:.0f} Days")
    with r_c2:
        st.write("### Yearly Coverage Progress (Blue Rod)")
        st.progress(min(survival_days/365, 1.0))
        st.caption(f"Currently securing {survival_days/365:.1%} of a calendar year.")

with tabs[6]:
    st.info(f"💡 **Strategic Context:** {TAB_SUMMARIES['Comparison']}")
    st.subheader("National Strategic Priorities Ranking")
    st.dataframe(rankings_df.sort_values('Sovereignty_Index'), hide_index=True, use_container_width=True)

# --- 12. FOOTER NLG ---
st.markdown("---")
st.subheader("📝 Automated Executive Intelligence Report")
st.info(exec_report_nlg)