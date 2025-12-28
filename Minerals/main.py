import pandas as pd
import glob
import os

# ==========================================
# 1. LOAD AND MERGE
# ==========================================
# We look for all files ending in '_master_data.csv'
# This picks up lithium_master_data.csv, copper_..., graphite_...
all_files = glob.glob('*_master_data.csv')

print(f"Found {len(all_files)} files: {all_files}")

df_list = []
for filename in all_files:
    df = pd.read_csv(filename)
    df_list.append(df)

if not df_list:
    print("Error: No csv files found. Make sure they are in the same folder.")
    exit()

# Combine into one Master DataFrame
master_df = pd.concat(df_list, ignore_index=True)

# ==========================================
# 2. DATA ENRICHMENT (The "Analytics" Part)
# ==========================================

# A. Clean Country Names (Remove "UNSPECIFIED" or garbage)
master_df = master_df[master_df['Country'] != 'UNSPECIFIED']

# B. Calculate "Market Share" (Risk Concentration)
# How much does India depend on a single country for a specific mineral?
# Formula: (Import Value from Country X / Total Import Value of Mineral) * 100
total_imports = master_df.groupby(['Year', 'Mineral'])['Value_USD'].transform('sum')
master_df['Market_Share_Pct'] = (master_df['Value_USD'] / total_imports) * 100

# ==========================================
# 3. GENERATE INSIGHTS (For your Report)
# ==========================================
print("\n--- FINAL DATASET SUMMARY ---")
print(master_df.groupby(['Mineral', 'Year'])['Value_USD'].sum())

print("\n--- CRITICAL VULNERABILITIES (High Dependency) ---")
# Show partners where we rely on them for > 40% of supply
high_risk = master_df[master_df['Market_Share_Pct'] > 40]
print(high_risk[['Year', 'Mineral', 'Country', 'Market_Share_Pct']].sort_values('Year', ascending=False))

# ==========================================
# 4. EXPORT FINAL DATASET
# ==========================================
master_df.to_csv('all_minerals_merged.csv', index=False)
print("\nSuccess! Saved 'all_minerals_merged.csv' - Ready for Dashboard.")