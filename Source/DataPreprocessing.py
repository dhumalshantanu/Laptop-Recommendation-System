import pandas as pd
import numpy as np

file_path = r"C:\Users\Shantanu\Desktop\Laptop Recomendation System\Laptop-Recommendation-System\Laptop_Details.xlsx"
df = pd.read_excel(file_path)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# Handle Missing Values
# Numerical columns → fill with median
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Categorical columns → fill with mode
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values handled successfully.")

# Convert RAM column (example: '8GB' → 8)
if "RAM" in df.columns:
    df["RAM"] = df["RAM"].astype(str).str.extract("(\d+)").astype(int)

# Convert Storage column (example: '512GB SSD' → 512)
if "Storage" in df.columns:
    df["Storage"] = df["Storage"].astype(str).str.extract("(\d+)").astype(int)

# Convert Price to numeric
if "Price in Rs" in df.columns:
    df["Price in Rs"] = df["Price in Rs"].astype(str).str.replace(",", "")
    df["Price in Rs"] = pd.to_numeric(df["Price in Rs"], errors="coerce")

# Fill any price NaN after conversion
df["Price in Rs"] = df["Price in Rs"].fillna(df["Price in Rs"].median())

# Standardize Text Columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("").str.lower().str.strip()


print("\nData type after cleaning:")
print(df.dtypes)

# Remove Duplicate Rows
df.drop_duplicates(inplace=True)

print("\nDuplicates removed.")
print("Final Shape:", df.shape)

# Save Cleaned Dataset
df.to_csv("cleaned_laptop_data.csv", index=False)

print("\nPhase 1 Completed Successfully!")
print("Cleaned data saved as 'cleaned_laptop_data.csv'")
