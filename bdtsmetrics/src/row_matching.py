import pandas as pd
import numpy as np


def match_dataframes_by_similarity(df1, df2,
                                   feature_columns=None,
                                   metric='euclidean'):
    """
    Matches rows from two DataFrames based on similarity computed from
    specified feature columns (default: first 4 columns). The number of
    matched rows equals the smaller DataFrame's number of rows. A greedy
    one-to-one matching is performed: once a row in the larger DataFrame
    is matched, it is removed from further consideration.

    This version handles a mix of numerical and categorical features by
    leaving numeric columns as-is and one-hot encoding categorical columns.

    Parameters:
      df1, df2          : Input DataFrames with at least 8 columns.
      feature_columns   : List (or Index) of column names to use. If None,
                          the first 4 columns from df1 are used.
      metric            : 'euclidean' or 'cosine'. For 'euclidean',
                          the row with the smallest L2 distance is chosen.
                          For 'cosine', the row with the highest cosine
                          similarity is chosen.

    Returns:
      matched_df1, matched_df2 : Two DataFrames (with reset index) where
                                 the i-th row in each DataFrame is a matched pair.
    """
    # Use the first column if feature_columns is not provided.
    if feature_columns is None:
        feature_columns = df1.columns[:1]

    # Check that df2 has these columns.
    if not all(col in df2.columns for col in feature_columns):
        raise ValueError("df2 does not contain all required feature columns.")

    # Decide which DataFrame is smaller.
    if len(df1) < len(df2):
        small_df, large_df = df1.copy(), df2.copy()
    else:
        small_df, large_df = df2.copy(), df1.copy()

    # Extract the feature subsets.
    features_small = small_df[feature_columns]
    features_large = large_df[feature_columns]

    # Identify numeric and categorical columns.
    numeric_cols = []
    categorical_cols = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(features_small[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Numeric features: use them as-is.
    if numeric_cols:
        numeric_small = features_small[numeric_cols].copy()
        numeric_large = features_large[numeric_cols].copy()
    else:
        numeric_small = pd.DataFrame(index=features_small.index)
        numeric_large = pd.DataFrame(index=features_large.index)

    # Categorical features: apply one-hot encoding.
    if categorical_cols:
        cat_small = pd.get_dummies(features_small[categorical_cols])
        cat_large = pd.get_dummies(features_large[categorical_cols])
        # Align the one-hot encoded DataFrames so they share the same columns.
        cat_small, cat_large = cat_small.align(cat_large, join='outer', axis=1, fill_value=0)
    else:
        cat_small = pd.DataFrame(index=features_small.index)
        cat_large = pd.DataFrame(index=features_large.index)

    # Combine numeric and categorical features.
    combined_small = pd.concat([numeric_small, cat_small], axis=1)
    combined_large = pd.concat([numeric_large, cat_large], axis=1)

    # Convert the combined features to NumPy arrays.
    arr_small = combined_small.to_numpy()
    arr_large = combined_large.to_numpy()


    n_large = arr_large.shape[0]
    # Create a boolean mask for available rows in the large DataFrame.
    available_mask = np.ones(n_large, dtype=bool)

    # Lists to store the original indices of the matched rows.
    match_small_indices = []
    match_large_indices = []

    # Loop over each row in the smaller DataFrame.
    for i, row in enumerate(arr_small):
        # Get the indices of the available rows in the large DataFrame.
        available_indices = np.nonzero(available_mask)[0]
        if len(available_indices) == 0:
            break  # No more rows available to match.

        # Compute the difference for available rows.
        candidates = arr_large[available_indices]
        if metric == 'euclidean':
            # Compute Euclidean distances.
            diff = candidates - row
            dists = np.linalg.norm(diff, axis=1)
            best_local_idx = np.argmin(dists)
        elif metric == 'cosine':
            # Compute cosine similarity.
            row_norm = np.linalg.norm(row)
            # To avoid division by zero add a tiny constant.
            candidates_norm = np.linalg.norm(candidates, axis=1) + 1e-10
            # Dot product between the row and each candidate.
            dots = np.dot(candidates, row)
            # Cosine similarity: higher is better.
            cos_sim = dots / (row_norm * candidates_norm)
            best_local_idx = np.argmax(cos_sim)
        else:
            raise ValueError("Unknown metric. Use 'euclidean' or 'cosine'.")

        # The best candidate's index in the full large array.
        best_global_idx = available_indices[best_local_idx]

        # Record the original indices.
        match_small_indices.append(small_df.index[i])
        match_large_indices.append(large_df.index[best_global_idx])

        # Mark the candidate as no longer available.
        available_mask[best_global_idx] = False

    # Build new DataFrames for the matched rows.
    matched_small_df = small_df.loc[match_small_indices].copy().reset_index(drop=True)
    matched_large_df = large_df.loc[match_large_indices].copy().reset_index(drop=True)

    return matched_small_df, matched_large_df

# ===========================
# Example usage:
# Assume you have two DataFrames, df1 and df2, each with at least 8 columns.
# They share the same column names, and you want to match rows based on the first 4 columns.

# df1 = pd.read_csv('data1.csv')
# df2 = pd.read_csv('data2.csv')

# Get the matched DataFrames (using Euclidean distance):
# matched_df1, matched_df2 = match_dataframes_by_similarity(df1, df2)

# Or, if you prefer cosine similarity:
# matched_df1, matched_df2 = match_dataframes_by_similarity(df1, df2, metric='cosine')

# Now, matched_df1 and matched_df2 each have min(len(df1), len(df2)) rows, where row i in each
# DataFrame forms a matched pair based on similarity of the feature columns.
# ===========================
