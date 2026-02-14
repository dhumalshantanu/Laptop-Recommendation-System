# =========================================
# LAPTOP RECOMMENDATION SYSTEM (FINAL)
# =========================================

import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------
# LOAD DATA & SCALER
# -----------------------------------------

DATA_PATH = r"C:\Users\Shantanu\Desktop\Laptop Recomendation System\Laptop-Recommendation-System\Processed_Data\laptop_data_preprocessed.csv"
SCALER_PATH = r"C:\Users\Shantanu\Desktop\Laptop Recomendation System\feature_scaler.pkl"

df = pd.read_csv(DATA_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------------------
# REQUIRED COLUMNS CHECK
# -----------------------------------------

required_cols = [
    'Price_Raw',
    'Price in Rs',
    'RAM',
    'Weight',
    'CPU_Strength_Score',
    'GPU_Strength_Score',
    'Storage_Performance_Score'
]

for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column missing: {col}")

# -----------------------------------------
# FEATURES USED FOR SIMILARITY (MUST MATCH SCALER)
# -----------------------------------------

SIMILARITY_FEATURES = [
    'Price in Rs',          # scaled
    'RAM',
    'Weight',
    'CPU_Strength_Score',
    'GPU_Strength_Score',
    'Storage_Performance_Score'
]

# -----------------------------------------
# USER TYPE PREFERENCE LOGIC
# -----------------------------------------

def user_type_preferences(user_type):
    user_type = user_type.lower()

    if user_type == "gaming":
        return {
            'CPU_Strength_Score': 5,
            'GPU_Strength_Score': 5,
            'RAM': 16,
            'Storage_Performance_Score': 600,
            'Weight': 2.6
        }

    elif user_type == "student":
        return {
            'CPU_Strength_Score': 2,
            'GPU_Strength_Score': 1,
            'RAM': 8,
            'Storage_Performance_Score': 256,
            'Weight': 1.6
        }

    elif user_type == "office":
        return {
            'CPU_Strength_Score': 3,
            'GPU_Strength_Score': 2,
            'RAM': 8,
            'Storage_Performance_Score': 512,
            'Weight': 2.0
        }

    else:
        raise ValueError("Invalid user type")

# -----------------------------------------
# BUILD USER FEATURE VECTOR (SCALED)
# -----------------------------------------

def build_user_vector(min_price, max_price, user_type):
    prefs = user_type_preferences(user_type)

    user_data = {
        'Price in Rs': (min_price + max_price) / 2,  # raw, will be scaled
        'RAM': prefs['RAM'],
        'Weight': prefs['Weight'],
        'CPU_Strength_Score': prefs['CPU_Strength_Score'],
        'GPU_Strength_Score': prefs['GPU_Strength_Score'],
        'Storage_Performance_Score': prefs['Storage_Performance_Score']
    }

    user_df = pd.DataFrame([user_data])

    # Apply SAME scaler used during preprocessing
    user_df[SIMILARITY_FEATURES] = scaler.transform(
        user_df[SIMILARITY_FEATURES]
    )

    return user_df

# -----------------------------------------
# RECOMMENDATION FUNCTION
# -----------------------------------------

def recommend_laptops(min_price, max_price, user_type, top_n=5):

    # 1. FILTER USING RAW PRICE (IMPORTANT)
    df_filtered = df[
        (df['Price_Raw'] >= min_price) &
        (df['Price_Raw'] <= max_price)
    ].copy()

    # Fallback if no laptops in exact range
    if df_filtered.empty:
        print("\n⚠️ No laptops found in this exact range.")
        print("Showing closest matches instead...\n")
        df_filtered = df.copy()

    # 2. Build user vector
    user_vector = build_user_vector(min_price, max_price, user_type)

    # 3. Compute cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        df_filtered[SIMILARITY_FEATURES]
    )[0]

    df_filtered['Similarity_Score'] = similarity_scores

    # 4. Rank recommendations
    recommendations = (
        df_filtered
        .sort_values(
            by=[
                'Similarity_Score',
                'CPU_Strength_Score',
                'GPU_Strength_Score'
            ],
            ascending=False
        )
        .head(top_n)
    )

    return recommendations

# -----------------------------------------
# USER INPUT (SAFE)
# -----------------------------------------

def take_user_input():
    print("\n💻 LAPTOP RECOMMENDATION SYSTEM")
    print("--------------------------------")

    # Budget input
    while True:
        try:
            min_price = int(input("Enter minimum budget (₹): ").strip())
            max_price = int(input("Enter maximum budget (₹): ").strip())
            if min_price > max_price:
                print("❌ Minimum budget cannot be greater than maximum budget.")
                continue
            break
        except ValueError:
            print("❌ Please enter valid numeric values.")

    # Usage input
    while True:
        print("\nSelect usage type:")
        print("1. Student")
        print("2. Office")
        print("3. Gaming")

        choice = input("Enter choice (1/2/3 or name): ").strip().lower()

        if choice in ['1', 'student']:
            user_type = "student"
            break
        elif choice in ['2', 'office']:
            user_type = "office"
            break
        elif choice in ['3', 'gaming']:
            user_type = "gaming"
            break
        else:
            print("❌ Invalid choice. Try again.")

    return min_price, max_price, user_type

# -----------------------------------------
# MAIN EXECUTION
# -----------------------------------------

if __name__ == "__main__":

    min_price, max_price, user_type = take_user_input()

    results = recommend_laptops(
        min_price=min_price,
        max_price=max_price,
        user_type=user_type,
        top_n=5
    )

    print("\n✅ TOP RECOMMENDED LAPTOPS:\n")
    print(
        results[
            [
                'Price_Raw',
                'RAM',
                'Weight',
                'CPU_Strength_Score',
                'GPU_Strength_Score',
                'Storage_Performance_Score',
                'Similarity_Score'
            ]
        ]
    )

# =========================================
# END OF FILE
# =========================================
