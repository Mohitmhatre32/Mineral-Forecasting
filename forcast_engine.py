
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
import os

warnings.filterwarnings("ignore")

# --- HELPER: CREATE SEQUENCES FOR LSTM ---
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

def run_advanced_forecasting():
    print("--- Starting Hybrid AI Forecasting (SARIMAX + LSTM) ---")
    df = pd.read_csv('all_minerals_merged.csv')
    df['Date'] = pd.to_datetime(df['Year'].str.split('-').str[0] + '-04-01')
    
    forecast_output = []
    metrics_output = []

    for mineral in df['Mineral'].unique():
        data = df[df['Mineral'] == mineral].groupby('Date')['Value_USD'].sum()
        if len(data) < 3: continue
        
        # 1. Resample and Interpolate for Time Series
        ts = data.resample('MS').interpolate(method='linear')
        ts_values = ts.values.reshape(-1, 1)
        
        # 2. SARIMAX Model (Linear Component)
        try:
            sarimax_model = SARIMAX(ts, order=(1, 1, 1)).fit(disp=False)
            sarimax_pred = sarimax_model.fittedvalues
            sarimax_fcast = sarimax_model.get_forecast(steps=36).summary_frame(alpha=0.05)
            
            # 3. LSTM Model (Non-Linear Component)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(ts_values)
            
            seq_len = 6 # 6 months lookback
            if len(scaled_data) > seq_len:
                X, y = create_sequences(scaled_data, seq_len)
                
                lstm_model = Sequential([
                    LSTM(50, activation='relu', input_shape=(seq_len, 1), return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                lstm_model.compile(optimizer='adam', loss='mse')
                lstm_model.fit(X, y, epochs=20, verbose=0)
                
                # Predict Future with LSTM
                last_seq = scaled_data[-seq_len:].reshape(1, seq_len, 1)
                lstm_fcasts = []
                for _ in range(36):
                    next_val = lstm_model.predict(last_seq, verbose=0)
                    lstm_fcasts.append(next_val[0,0])
                    last_seq = np.append(last_seq[0,1:], next_val).reshape(1, seq_len, 1)
                
                lstm_final_fcast = scaler.inverse_transform(np.array(lstm_fcasts).reshape(-1, 1)).flatten()
            else:
                lstm_final_fcast = sarimax_fcast['mean'].values # Fallback

            # 4. HYBRID LOGIC: 60% SARIMAX + 40% LSTM
            hybrid_fcast = (sarimax_fcast['mean'].values * 0.6) + (lstm_final_fcast * 0.4)
            
            # Prepare Forecast DataFrame
            f_df = pd.DataFrame({
                'Date': sarimax_fcast.index,
                'Forecast_SARIMAX': sarimax_fcast['mean'].values,
                'Forecast_LSTM': lstm_final_fcast,
                'Forecast_Hybrid': hybrid_fcast,
                'mean_ci_upper': sarimax_fcast['mean_ci_upper'].values, # Using SARIMAX CI as proxy
                'mean_ci_lower': sarimax_fcast['mean_ci_lower'].values,
                'Mineral': mineral
            })
            forecast_output.append(f_df)
            
            # 5. VALIDATION METRICS
            r2 = r2_score(ts, sarimax_pred)
            metrics_output.append({
                'Mineral': mineral,
                'R2_Score': round(r2, 3),
                'MAE': round(mean_absolute_error(ts, sarimax_pred), 2),
                'Model_Type': 'Hybrid ARIMA-LSTM'
            })
            print(f"-> Optimized Hybrid Model for {mineral}")

        except Exception as e:
            print(f"Error in {mineral}: {e}")

    pd.concat(forecast_output).to_csv('final_forecast_values.csv', index=False)
    pd.DataFrame(metrics_output).to_csv('model_metrics.csv', index=False)
    print("SUCCESS: Multi-model forecasts generated.")

if __name__ == "__main__":
    run_advanced_forecasting()