import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# We map each file to the specific column INDEX we need.
# Standard DGCI&S Comparison Table Layout:
# Col 0: SNo
# Col 1: Country / Region
# Col 2: Previous Year Value
# Col 3: CURRENT YEAR VALUE  <-- We want this
# Col 4: Growth Value
# Col 5: Previous Year Volume
# Col 6: CURRENT YEAR VOLUME <-- We want this
# Col 7: Growth Volume

files = [
    {'name': '18-19.xlsx', 'year': '2018-2019'},
    {'name': '19-20.xlsx', 'year': '2019-2020'},
    {'name': '20-21.xlsx', 'year': '2020-2021'},
    {'name': '21-22.xlsx', 'year': '2021-2022'},
    {'name': '22-23.xlsx', 'year': '2022-2023'},
    {'name': '23-24.xlsx', 'year': '2023-2024'},
    {'name': '24-25.xlsx', 'year': '2024-2025'}
]


combined_data = []

print("--- Starting Data Processing (Index Mode) ---")

for file_info in files:
    try:
        print(f"\nProcessing {file_info['name']}...")
        
        # Load file
        df = pd.read_excel(file_info['name'])
        
        # 1. FIND HEADER ROW
        # Look for the row containing 'Country / Region'
        header_row_idx = df[df.apply(lambda row: row.astype(str).str.contains('Country / Region', case=False).any(), axis=1)].index
        
        if not header_row_idx.empty:
            df.columns = df.iloc[header_row_idx[0]] # Set header
            df = df.iloc[header_row_idx[0]+1:]      # Drop rows above
        else:
            print(f"Skipping {file_info['name']}: Header not found.")
            continue

        # 2. RENAME COLUMNS MANUALLY (To fix duplicates)
        # We force standard names based on position to avoid "Duplicate Column" errors
        # We expect at least 7 columns.
        new_columns = ['SNo', 'Country', 'Prev_Value', 'Value_USD', 'Gr_Val', 'Prev_Vol', 'Volume', 'Gr_Vol']
        
        # Assign these names to the first 8 columns (if file has fewer, we slice the list)
        current_cols = df.columns.tolist()
        if len(current_cols) >= 8:
            df.columns = new_columns + current_cols[8:]
        else:
            # Fallback for shorter tables
            df.columns = new_columns[:len(current_cols)]

        # 3. FILTERING
        df = df[df['Country'].notna()]
        df = df[df['Country'] != 'Total']
        
        # 4. CLEANING & TYPES
        # Remove commas and convert to numbers
        for col in ['Value_USD', 'Volume']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. ENRICHMENT
        df['Year'] = file_info['year']
        df['Mineral'] = 'RareEarthMetal-Compounds'      # <--- Change name
        df['HS_Code'] = '28461010'    # <--- Change HS Code
        df['Flow'] = 'Import'
        
        # 6. SAVE
        cols_to_keep = ['Year', 'Mineral', 'HS_Code', 'Flow', 'Country', 'Value_USD', 'Volume']
        # Only keep columns that actually exist
        final_cols = [c for c in cols_to_keep if c in df.columns]
        
        combined_data.append(df[final_cols])
        print(f"-> Success: Extracted {len(df)} rows.")

    except Exception as e:
        print(f"ERROR in {file_info['name']}: {e}")

# ==========================================
# EXPORT
# ==========================================
if combined_data:
    master_df = pd.concat(combined_data, ignore_index=True)
    master_df.to_csv('RareEarthMetal-Compounds_master_data.csv', index=False)
    print("\nSUCCESS! Saved 'RareEarthMetal-Compounds_master_data.csv'")
    print(master_df.head())
else:
    print("\nNo data processed.")