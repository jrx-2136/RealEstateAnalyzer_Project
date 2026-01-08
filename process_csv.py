import pandas as pd
import random

# Read the CSV file
csv_path = 'data/outputs/magicbricks_india_properties_cleaned.csv'
df = pd.read_csv(csv_path)

print(f"Original shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# Drop the specified columns
columns_to_drop = ['bedrooms', 'bathrooms', 'link', 'title']
df = df.drop(columns=columns_to_drop, errors='ignore')

print(f"\nAfter dropping columns: {df.shape}")

# Calculate missing price_per_sqft using price_total_inr and area_sqft
def calculate_price_per_sqft(row):
    """Calculate price_per_sqft if missing"""
    if pd.notna(row['price_per_sqft']):
        return row['price_per_sqft']
    elif pd.notna(row['price_total_inr']) and pd.notna(row['area_sqft']) and row['area_sqft'] > 0:
        return row['price_total_inr'] / row['area_sqft']
    else:
        return None

df['price_per_sqft'] = df.apply(calculate_price_per_sqft, axis=1)

# Add BHK column based on area_sqft
def assign_bhk(area):
    """Assign BHK based on area_sqft"""
    if pd.isna(area):
        return None
    elif area < 1000:
        return 1
    else:
        return random.randint(2, 5)

df['BHK'] = df['area_sqft'].apply(assign_bhk)

# Reorder columns for better readability
column_order = ['location', 'city', 'price_total_inr', 'price_per_sqft', 'area_sqft', 'BHK']
df = df[column_order]

# Save the processed CSV
output_path = 'data/outputs/magicbricks_india_final.csv'
df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"CSV PROCESSING SUMMARY")
print(f"{'='*60}")
print(f"Columns dropped: {columns_to_drop}")
print(f"New column 'BHK' added based on area_sqft:")
print(f"  - 1 BHK if area_sqft < 1000")
print(f"  - Random 2-5 BHK if area_sqft >= 1000")
print(f"\nCalculated missing price_per_sqft values")
print(f"Formula: price_per_sqft = price_total_inr / area_sqft")
print(f"\nFinal shape: {df.shape}")
print(f"Output saved to: {output_path}")
print(f"\n{'='*60}")
print(f"FIRST FEW ROWS:")
print(f"{'='*60}")
print(df.head(10))
print(f"\n{'='*60}")
print(f"DATA STATISTICS:")
print(f"{'='*60}")
print(f"\nBHK distribution:")
print(df['BHK'].value_counts().sort_index())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nPrice per sqft stats:")
print(df['price_per_sqft'].describe())
