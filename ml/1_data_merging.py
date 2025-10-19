import pandas as pd
import glob
import os
import csv

def merge_all_data():
    print("📁 Merging all 64 data files from solar_data folder...")
    
    # Updated path - solar_data is in the same folder
    raw_data_path = "solar_data/"
    all_files = glob.glob(raw_data_path + "*.csv")
    
    print(f"Found {len(all_files)} files")
    
    dataframes = []
    error_files = []
    
    for file in all_files:
        try:
            # Try different methods to read the problematic CSV files
            filename = os.path.basename(file)
            
            # METHOD 1: Try with error handling for bad lines
            try:
                df = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8')
            except:
                # METHOD 2: Try with different encoding
                try:
                    df = pd.read_csv(file, on_bad_lines='skip', encoding='latin-1')
                except:
                    # METHOD 3: Try with manual CSV reading
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            rows = []
                            for i, row in enumerate(reader):
                                if len(row) == 9:  # Expected number of columns based on error
                                    rows.append(row)
                                # Skip rows with wrong number of columns
                            if rows:
                                df = pd.DataFrame(rows[1:], columns=rows[0])  # First row as header
                            else:
                                raise ValueError("No valid rows found")
                    except Exception as e:
                        print(f"❌ All methods failed for {filename}: {e}")
                        error_files.append(filename)
                        continue
            
            # Extract info from filename
            parts = filename.split('_')
            
            if len(parts) >= 3:
                df['city'] = parts[0]  # jaipur, chennai, etc
                df['scenario_type'] = parts[2].replace('.csv', '')  # clean, noisy, etc
                df['quarter'] = parts[1]  # q1, q2, etc
                df['source_file'] = filename  # Keep original filename
            else:
                print(f"⚠️  Unexpected filename format: {filename}")
                continue
            
            dataframes.append(df)
            print(f"✅ Processed: {filename} - Shape: {df.shape}")
            
        except Exception as e:
            print(f"❌ Error with {filename}: {e}")
            error_files.append(filename)
    
    if not dataframes:
        print("❌ No dataframes to merge!")
        return None
    
    # Merge everything
    full_data = pd.concat(dataframes, ignore_index=True)
    
    # Save merged data - going up one level to project_ML
    output_path = "../data_processed/merged_complete_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_data.to_csv(output_path, index=False)
    
    print(f"🎉 Merged data shape: {full_data.shape}")
    print("📊 Columns:", full_data.columns.tolist())
    print(f"💾 Saved to: {output_path}")
    
    if error_files:
        print(f"\n⚠️  {len(error_files)} files had errors:")
        for f in error_files:
            print(f"   - {f}")
    
    return full_data

if __name__ == "__main__":
    data = merge_all_data()
    if data is not None:
        print("\n📋 Data Summary:")
        print(f"Total records: {len(data)}")
        print(f"Cities: {data['city'].unique()}")
        print(f"Scenarios: {data['scenario_type'].unique()}")
        print(f"Quarters: {data['quarter'].unique()}")