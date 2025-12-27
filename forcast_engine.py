# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import warnings
# import os

# warnings.filterwarnings("ignore")

# # ==========================================
# # 1. LOAD DATA
# # ==========================================
# if not os.path.exists('all_minerals_merged.csv'):
#     print("Error: all_minerals_merged.csv not found.")
#     exit()

# df = pd.read_csv('all_minerals_merged.csv')

# # 1.a Calculate Unit Price (Price Variation Analysis)
# # Avoid division by zero
# df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(1)
# df['Unit_Price'] = df['Value_USD'] / df['Volume']

# # Group Annual
# annual_df = df.groupby(['Mineral', 'Year'])[['Value_USD', 'Unit_Price']].sum().reset_index()

# def parse_year(y_str):
#     return int(str(y_str).split('-')[0])
# annual_df['Year_Start'] = annual_df['Year'].apply(parse_year)
# annual_df['Date'] = pd.to_datetime(annual_df['Year_Start'].astype(str) + '-04-01')

# # ==========================================
# # 2. EXPANSION (With Unit Price)
# # ==========================================
# def expand_data(mineral_name, data):
#     data = data.sort_values('Date')
#     start_date = data['Date'].min()
#     end_date = data['Date'].max() + pd.DateOffset(months=12)
    
#     full_range = pd.date_range(start=start_date, end=end_date, freq='MS')
#     ts_df = pd.DataFrame({'Date': full_range})
    
#     ts_df = pd.merge(ts_df, data[['Date', 'Value_USD', 'Unit_Price']], on='Date', how='left')
    
#     # Interpolate both Value and Price
#     cols = ['Value_USD', 'Unit_Price']
#     for c in cols:
#         ts_df[c] = ts_df[c].interpolate(method='linear').ffill().bfill()
#         if c == 'Value_USD':
#             ts_df[c] = ts_df[c] / 12 # Monthly average for value
#         # Unit price stays roughly same, so we don't divide by 12, just add noise
        
#         # Add Noise
#         np.random.seed(42)
#         noise = np.random.normal(0, ts_df[c] * 0.05, len(ts_df))
#         ts_df[c] = ts_df[c] + noise

#     ts_df['Mineral'] = mineral_name
#     return ts_df

# minerals = annual_df['Mineral'].unique()
# all_ts = []
# metrics_list = [] # Store accuracy scores
# forecast_results = []

# for m in minerals:
#     m_data = annual_df[annual_df['Mineral'] == m]
#     if len(m_data) > 0:
#         ts = expand_data(m, m_data)
#         all_ts.append(ts)

# full_df = pd.concat(all_ts)
# full_df.to_csv('monthly_expanded_data.csv', index=False) # Save for ANOVA

# # ==========================================
# # 3. FORECASTING & VALIDATION (The "Technical" Requirement)
# # ==========================================
# print("\n--- Training & Validating Models ---")

# for mineral in minerals:
#     subset = full_df[full_df['Mineral'] == mineral].set_index('Date')['Value_USD']
    
#     # TRAIN / TEST SPLIT (To calculate accuracy)
#     train_size = int(len(subset) * 0.85)
#     train, test = subset.iloc[:train_size], subset.iloc[train_size:]
    
#     try:
#         # Train Model
#         model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
        
#         # Test Prediction
#         predictions = model.forecast(len(test))
        
#         # Calculate Metrics (MAE, RMSE, MAPE)
#         mae = mean_absolute_error(test, predictions)
#         rmse = np.sqrt(mean_squared_error(test, predictions))
#         mape = np.mean(np.abs((test - predictions) / test)) * 100
        
#         # Save Metrics
#         metrics_list.append({
#             'Mineral': mineral,
#             'MAE': mae,
#             'RMSE': rmse,
#             'MAPE (%)': round(mape, 2),
#             'Model': 'Holt-Winters'
#         })
        
#         # Retrain on FULL data for future forecast
#         final_model = ExponentialSmoothing(subset, trend='add', seasonal='add', seasonal_periods=12).fit()
#         future_forecast = final_model.forecast(12)
        
#         f_df = pd.DataFrame({
#             'Date': future_forecast.index, 
#             'Forecast_Value': future_forecast.values, 
#             'Mineral': mineral
#         })
#         forecast_results.append(f_df)
#         print(f"-> {mineral}: MAPE = {mape:.2f}% (Accuracy Score)")
        
#     except Exception as e:
#         print(f"Failed {mineral}: {e}")

# # Save Outputs
# if metrics_list:
#     pd.DataFrame(metrics_list).to_csv('model_metrics.csv', index=False)
#     pd.concat(forecast_results).to_csv('final_forecast_values.csv', index=False)
#     print("\nSUCCESS: Saved 'model_metrics.csv' and 'final_forecast_values.csv'")

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

def run_predictive_modeling():
    print("--- Starting 3-Year SARIMAX Forecasting ---")
    try:
        df = pd.read_csv('all_minerals_merged.csv')
    except: return

    df['Date'] = pd.to_datetime(df['Year'].str.split('-').str[0] + '-04-01')
    forecast_results = []
    metrics_list = []

    for mineral in df['Mineral'].unique():
        m_data = df[df['Mineral'] == mineral].groupby('Date')['Value_USD'].sum()
        # Monthly expansion for Time Series
        ts = m_data.resample('MS').interpolate(method='linear')
        
        try:
            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            res = model.fit(disp=False)
            
            # Future Forecast (36 Months)
            fcast_obj = res.get_forecast(steps=36)
            f_df = fcast_obj.summary_frame(alpha=0.05)
            f_df['Mineral'] = mineral
            f_df = f_df.reset_index().rename(columns={'index': 'Date', 'mean': 'Forecast_Value'})
            forecast_results.append(f_df)
            
            # Validation
            pred = res.get_forecast(steps=len(ts)).predicted_mean
            metrics_list.append({'Mineral': mineral, 'R2_Score': round(r2_score(ts, pred), 3)})
            print(f"-> Forecast for {mineral} complete.")
        except: continue

    pd.concat(forecast_results).to_csv('final_forecast_values.csv', index=False)
    pd.DataFrame(metrics_list).to_csv('model_metrics.csv', index=False)
    print("SUCCESS: 36-Month forecast generated with R2 validation.")

if __name__ == "__main__":
    run_predictive_modeling()