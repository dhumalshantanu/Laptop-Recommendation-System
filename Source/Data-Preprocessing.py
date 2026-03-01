import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import re 
import os 


df = pd.read_csv(r"C:\Users\Shantanu\Desktop\Laptop Recomendation System\Laptop-Recommendation-System\Data\Laptop_Details.csv", encoding="latin1")

print(df.info())
print(df.isnull().sum())

df['RAM'] = df['RAM'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
df['Memory'] = df['Memory'].str.lower().str.strip().str.replace(' ', '')

df['SSD_GB'] = 0
df['HDD_GB'] = 0
df['Flash_GB'] = 0

def parse_memory(memory):
    ssd = 0
    hdd = 0
    flash = 0

    # SSD
    ssd_matches = re.findall(r'(\d+)(gb|tb)ssd', memory)
    for size, unit in ssd_matches:
        ssd += int(size) * (1024 if unit == 'tb' else 1)

    # HDD
    hdd_matches = re.findall(r'(\d+)(gb|tb)hdd', memory)
    for size, unit in hdd_matches:
        hdd += int(size) * (1024 if unit == 'tb' else 1)

    # Flash Storage
    flash_matches = re.findall(r'(\d+)(gb)flash', memory)
    for size, _ in flash_matches:
        flash += int(size)

    return pd.Series([ssd, hdd, flash])

df[['SSD_GB', 'HDD_GB', 'Flash_GB']] = df['Memory'].apply(parse_memory)

df['Has_SSD'] = (df['SSD_GB'] > 0).astype(int)
df['Has_HDD'] = (df['HDD_GB'] > 0).astype(int)
df['Has_Flash'] = (df['Flash_GB'] > 0).astype(int)

df['Storage_Performance_Score'] = (
    df['SSD_GB'] * 1.0 +
    df['Flash_GB'] * 0.8 +
    df['HDD_GB'] * 0.3
)

def storage_class(row):
    if row['SSD_GB'] >= 512:
        return 3   # High Performance
    elif row['SSD_GB'] > 0:
        return 2   # Medium Performance
    else:
        return 1   # Basic

df['Storage_Class'] = df.apply(storage_class, axis=1)

df.drop(columns=['Memory'], inplace=True)

# ===============================
# GPU PREPROCESSING BLOCK
# ===============================

import pandas as pd

# --- Safety check ---
if 'GPU' not in df.columns:
    raise KeyError("Column 'GPU' not found in dataset")

# 1. Clean GPU column
df['GPU'] = df['GPU'].astype(str).str.lower().str.strip()

# 2. Extract GPU Brand
def extract_gpu_brand(gpu):
    if 'nvidia' in gpu:
        return 'nvidia'
    elif 'amd' in gpu or 'radeon' in gpu:
        return 'amd'
    elif 'intel' in gpu:
        return 'intel'
    else:
        return 'other'

df['GPU_Brand'] = df['GPU'].apply(extract_gpu_brand)

# 3. One-Hot Encode GPU Brand (SAFE)
df = pd.get_dummies(df, columns=['GPU_Brand'], drop_first=True)

# 4. Dedicated vs Integrated GPU
def is_dedicated_gpu(gpu):
    return 1 if ('nvidia' in gpu or 'amd' in gpu or 'radeon' in gpu) else 0

df['Dedicated_GPU'] = df['GPU'].apply(is_dedicated_gpu)

# 5. GPU Performance Tier
def gpu_performance_tier(gpu):
    if any(x in gpu for x in ['rtx 4090', 'rtx 4080', 'rtx 4070']):
        return 5
    elif any(x in gpu for x in ['rtx 3060', 'rtx 3070', 'rtx 3080']):
        return 4
    elif any(x in gpu for x in ['gtx 1650', 'gtx 1660', 'rx 6500', 'rx 6600']):
        return 3
    elif any(x in gpu for x in ['mx', 'vega']):
        return 2
    else:
        return 1

df['GPU_Performance_Tier'] = df['GPU'].apply(gpu_performance_tier)

# 6. GPU Usage Class
def gpu_usage_class(tier):
    if tier >= 4:
        return 3   # Gaming / AI / Design
    elif tier >= 2:
        return 2   # Office / Light editing
    else:
        return 1   # Student / Basic use

df['GPU_Usage_Class'] = df['GPU_Performance_Tier'].apply(gpu_usage_class)

# 7. GPU Strength Score
df['GPU_Strength_Score'] = (
    df['Dedicated_GPU'] * 2 +
    df['GPU_Performance_Tier'] * 3
)

# 8. Drop original GPU column (LAST STEP)
df.drop(columns=['GPU'], inplace=True)

# ===============================
# END GPU PREPROCESSING
# ===============================

# ===============================
# CPU PREPROCESSING BLOCK
# ===============================

import pandas as pd

# --- Safety check ---
if 'CPU' not in df.columns:
    raise KeyError("Column 'CPU' not found in dataset")

# 1. Clean CPU column
df['CPU'] = df['CPU'].astype(str).str.lower().str.strip()

# 2. Extract CPU Brand
def extract_cpu_brand(cpu):
    if 'intel' in cpu:
        return 'intel'
    elif 'amd' in cpu or 'ryzen' in cpu:
        return 'amd'
    elif 'apple' in cpu or 'm1' in cpu or 'm2' in cpu:
        return 'apple'
    else:
        return 'other'

df['CPU_Brand'] = df['CPU'].apply(extract_cpu_brand)

# 3. One-Hot Encode CPU Brand
df = pd.get_dummies(df, columns=['CPU_Brand'], drop_first=True)

# 4. CPU Performance Tier
def cpu_performance_tier(cpu):
    # Apple Silicon
    if any(x in cpu for x in ['m2 pro', 'm2 max', 'm1 pro', 'm1 max']):
        return 5
    elif any(x in cpu for x in ['m2', 'm1']):
        return 4

    # Intel
    elif any(x in cpu for x in ['i9']):
        return 5
    elif any(x in cpu for x in ['i7']):
        return 4
    elif any(x in cpu for x in ['i5']):
        return 3
    elif any(x in cpu for x in ['i3']):
        return 2

    # AMD Ryzen
    elif any(x in cpu for x in ['ryzen 9']):
        return 5
    elif any(x in cpu for x in ['ryzen 7']):
        return 4
    elif any(x in cpu for x in ['ryzen 5']):
        return 3
    elif any(x in cpu for x in ['ryzen 3']):
        return 2

    else:
        return 1  # Very basic / unknown

df['CPU_Performance_Tier'] = df['CPU'].apply(cpu_performance_tier)

# 5. CPU Usage Class
def cpu_usage_class(tier):
    if tier >= 4:
        return 3   # Gaming / AI / Heavy workloads
    elif tier >= 2:
        return 2   # Office / Programming
    else:
        return 1   # Student / Basic use

df['CPU_Usage_Class'] = df['CPU_Performance_Tier'].apply(cpu_usage_class)

# 6. CPU Strength Score
df['CPU_Strength_Score'] = df['CPU_Performance_Tier'] * 3

# 7. Drop raw CPU column (LAST STEP)
df.drop(columns=['CPU'], inplace=True)

# ===============================
# END CPU PREPROCESSING
# ===============================

df['Overall_Performance_Score'] = (
    df['CPU_Strength_Score'] * 0.4 +
    df['GPU_Strength_Score'] * 0.4 +
    df['Storage_Performance_Score'] * 0.2
)

df['Price_Raw'] = df['Price in Rs']

# ===============================
# FEATURE SCALING BLOCK
# ===============================

from sklearn.preprocessing import StandardScaler
import joblib

# --- Columns to scale ---
features_to_scale = [
    'Price in Rs',
    'RAM',
    'Weight',
    'CPU_Strength_Score',
    'GPU_Strength_Score',
    'Storage_Performance_Score'
]

# --- Safety check ---
missing_cols = [col for col in features_to_scale if col not in df.columns]
if missing_cols:
    raise KeyError(f"Missing columns for scaling: {missing_cols}")

# --- Initialize scaler ---
scaler = StandardScaler()

# --- Fit & transform ---
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# --- Save scaler for future inference ---
joblib.dump(scaler, 'feature_scaler.pkl')

# ===============================
# END FEATURE SCALING
# ===============================


print(df.info())

# Create output directory if not exists
output_dir = "Processed_Data"
os.makedirs(output_dir, exist_ok=True)

# Save preprocessed dataset
processed_file_path = os.path.join(output_dir, "laptop_data_preprocessed.csv")
df.to_csv(processed_file_path, index=False)

print(f"Preprocessed data saved successfully at: {processed_file_path}")
