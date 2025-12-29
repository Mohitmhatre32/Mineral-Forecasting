# 🇮🇳 National Critical Mineral Strategic Intelligence  
### Advanced Decision Support System for India’s Mineral Security

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![Model](https://img.shields.io/badge/Model-Hybrid_LSTM_SARIMAX-gold.svg)

---

## 📖 Project Overview

As India accelerates its **green energy transition** and **high-tech manufacturing**, the security of **critical minerals** (Lithium, Cobalt, Copper, etc.) has become a matter of **national sovereignty**.

This project is a **Strategic Intelligence Dashboard** that goes beyond traditional data visualization into **policy-grade decision support**.  
It integrates:

- **Trade data** from DGCI&S  
- **Domestic exploration data** from the Geological Survey of India (GSI)  
- **Hybrid AI forecasting models**  

to predict **supply gaps**, **systemic risks**, and **geopolitical shock scenarios**.

---

## 🚀 Key Features

### 1. Hybrid AI Forecasting Engine
- **Multi-Model Architecture**
  - SARIMAX (statistical, linear trends)
  - LSTM (deep learning, non-linear patterns)
  - Hybrid model combining both
- **36-Month Forecast Horizon**
  - Medium-term projections with **95% confidence intervals**

---

### 2. Geopolitical Risk Intelligence
- **HHI Concentration Index**
  - Quantifies supplier monopoly risk
- **Strategic Vulnerability Score (SVS)**
  - Composite index combining:
    - Trade dependency
    - Geopolitical risk (normalized from *Fragile States Index 2024*)
- **Crisis Simulator**
  - Real-time *“What-If”* analysis  
  - Simulate a supply halt from any partner country and observe demand shock impact instantly

---

### 3. National Resilience Index (The Blue Rod)
- Computes the **Stockpile Survival Window**
- Indicates **number of days India can withstand a complete import disruption**
- Directly supports contingency planning and buffer stock policy

---

### 4. Geospatial Exploration Pipeline
- Maps **1,200+ active GSI exploration projects** using **Mapbox**
- **State-wise Exploration Intensity Chart**
  - Identifies domestic mineral sovereignty frontlines

---

### 5. Executive Intelligence Reporting
- **Automated NLG Reports**
  - Logic-driven summaries translating analytics into executive insights
- **One-Click PDF Export**
  - Multi-page strategic dossier
  - High-resolution charts and national priority rankings

---

## 🛠️ Technical Stack

| Layer | Tools |
|-----|------|
| Framework | Streamlit (Midnight Intelligence Theme) |
| Data Science | Pandas, NumPy, SciPy (ANOVA) |
| Machine Learning | TensorFlow (LSTM), Statsmodels (SARIMAX), Scikit-Learn (Isolation Forest) |
| Visualization | Plotly Graph Objects, Mapbox |
| Reporting | FPDF2, Kaleido |

---

## 📂 Project Structure

| File | Description |
|----|------------|
| `main.py` | Core strategic engine: risk indices, HHI, SVS, anomaly detection |
| `forecast_engine.py` | AI forecasting engine (LSTM + SARIMAX) for 30 minerals |
| `app.py` | Streamlit dashboard, simulators, PDF export |
| `all_minerals_merged.csv` | Master DGCI&S dataset (30 critical minerals) |
| `enriched_minerals.csv` | Output with computed risk & anomaly metrics |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Mohitmhatre32/Mineral-Forecasting.git
cd Mineral-Forecasting
git checkout data-Integration
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Data Pipelines (First Run Required)
```bash
python main.py
python forecast_engine.py
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🛡️ Strategic Impact

This system aligns directly with:

- **National Mineral Policy**
- **Atmanirbhar Bharat**
- **$5 Trillion Economy Vision**

By ranking minerals using a Sovereignty Index, policymakers can:

- **Prioritize P1 (Critical) minerals**
- **Optimize exploration budgets**
- **Preempt geopolitical supply shocks**
- **Strengthen long-term national resilience**

---

## 📌 Use Case

### Primary Users

- **Government policymakers**
- **Strategic planners**
- **Trade & energy analysts**
- **National security think tanks**

---

## 📄 License

For academic, research, and strategic demonstration purposes.