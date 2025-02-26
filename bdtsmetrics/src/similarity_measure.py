import pandas as pd
import numpy as np

def calculate_total_similarity(df1, df2):
    # Ensure the DataFrames have the same columns
    if not df1.columns.equals(df2.columns):
        raise ValueError("DataFrames must have identical columns to compare row-wise similarity")
    if df1.shape[0] != df2.shape[0]:
        # If lengths differ, align to the smaller length (alternatively, raise an error)
        min_len = min(len(df1), len(df2))
        df1 = df1.iloc[:min_len].copy()
        df2 = df2.iloc[:min_len].copy()
    
    # Convert boolean columns to numeric (1.0/0.0) for compatibility
    bool_cols = df1.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df1[bool_cols] = df1[bool_cols].astype(float)
        df2[bool_cols] = df2[bool_cols].astype(float)
    
    # Separate numeric and categorical columns
    numeric_cols = df1.select_dtypes(include=np.number).columns
    categorical_cols = df1.select_dtypes(include=['object', 'category']).columns
    
    # Fill missing values in numeric columns with 0
    df1_numeric = df1[numeric_cols].fillna(0.0)
    df2_numeric = df2[numeric_cols].fillna(0.0)
    
    # Encode categorical columns using one-hot (dummy) encoding
    if len(categorical_cols) > 0:
        # Concatenate categorical data from both DataFrames
        cat_combined = pd.concat([df1[categorical_cols], df2[categorical_cols]], ignore_index=True)
        # One-hot encode, treating NaN as a separate category
        cat_dummies = pd.get_dummies(cat_combined, dummy_na=True)
        # Split back into two sets of dummy-encoded data
        df1_cat_enc = cat_dummies.iloc[:df1.shape[0], :].reset_index(drop=True)
        df2_cat_enc = cat_dummies.iloc[df1.shape[0]:, :].reset_index(drop=True)
    else:
        # No categorical columns
        df1_cat_enc = pd.DataFrame(index=df1.index)
        df2_cat_enc = pd.DataFrame(index=df2.index)
    
    # Combine numeric and encoded categorical features for full feature vectors
    # Reset index to ensure proper alignment before concatenation
    df1_full = pd.concat([df1_numeric.reset_index(drop=True), df1_cat_enc.reset_index(drop=True)], axis=1)
    df2_full = pd.concat([df2_numeric.reset_index(drop=True), df2_cat_enc.reset_index(drop=True)], axis=1)
    
    # Convert DataFrames to NumPy arrays for fast computation
    X = df1_full.to_numpy(dtype=float)
    Y = df2_full.to_numpy(dtype=float)
    
    # Cosine similarity for each row: dot(x, y) / (||x|| * ||y||)
    dot_products = np.sum(X * Y, axis=1)
    norms_X = np.linalg.norm(X, axis=1)
    norms_Y = np.linalg.norm(Y, axis=1)
    # Avoid division by zero: if a row is all zeros in either vector, set similarity to 0
    cosine_sims = np.zeros_like(dot_products, dtype=float)
    valid_mask = (norms_X > 1e-12) & (norms_Y > 1e-12)   # tiny threshold to handle near-zero norms
    cosine_sims[valid_mask] = dot_products[valid_mask] / (norms_X[valid_mask] * norms_Y[valid_mask])
    
    # Euclidean distance for each row
    diff = X - Y
    distances = np.linalg.norm(diff, axis=1)
    # Convert distance to similarity: 1 / (1 + distance)
    dist_sims = 1.0 / (1.0 + distances)
    
    # Sum up the similarities across all rows
    total_cosine_similarity = cosine_sims.sum()
    total_distance_similarity = dist_sims.sum()
    average_cosine_similarity = total_cosine_similarity / len(cosine_sims)
    average_distance_similarity = total_distance_similarity / len(dist_sims)
    return average_cosine_similarity, average_distance_similarity

# Example usage (assuming df1 and df2 are defined DataFrames with identical schema):
# total_cos, total_dist = calculate_total_similarity(df1, df2)
# print("Total Cosine Similarity:", total_cos)
# print("Total Distance-Based Similarity:", total_dist)
