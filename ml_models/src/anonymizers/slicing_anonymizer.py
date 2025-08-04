# filepath: c:\Users\alise\OneDrive\Desktop\Bachelor Thesis\ml_models\src\anonymizers\slicing_anonymizer.py
import pandas as pd
import numpy as np
import os # For standalone testing
from typing import List

def run_slicing(df_input: pd.DataFrame, k: int, slice_definitions: List[List[str]], seed: int, sa_col: str | None = None) -> pd.DataFrame:
    """
    Applies Slicing anonymization to the provided DataFrame.
    Args:
        df_input (pd.DataFrame): The input DataFrame.
        k (int): The bucket size for horizontal slicing.
        slice_definitions (List[List[str]]): A list of lists, where each inner list defines a column slice.
                                             Example: [["colA", "colB"], ["colC", "colD", "SA_COL"]]
        seed (int): Random seed for shuffling.
        sa_col (str | None, optional): Name of the sensitive attribute column.
                                       If provided and present in a slice, it will be permuted with that slice.
    Returns:
        pd.DataFrame: The sliced (anonymized) DataFrame.
    """
    if df_input.empty:
        print("Error: Input DataFrame is empty for run_slicing.")
        return pd.DataFrame()
    if not slice_definitions:
        print("Error: No slice definitions provided for run_slicing.")
        return df_input.copy()
    if k <= 0:
        print("Error: k-value must be positive for run_slicing.")
        return df_input.copy()

    df_sliced = df_input.copy()
    
    # Validate slice definitions against DataFrame columns
    all_slice_cols = [col for slice_group in slice_definitions for col in slice_group]
    missing_cols = [col for col in all_slice_cols if col not in df_sliced.columns]
    if missing_cols:
        print(f"Warning: The following columns defined in slices are not in the DataFrame and will be ignored: {missing_cols}")
        # Filter out missing columns from slice_definitions
        valid_slice_definitions = []
        for slice_group in slice_definitions:
            valid_group = [col for col in slice_group if col in df_sliced.columns]
            if valid_group:
                valid_slice_definitions.append(valid_group)
        slice_definitions = valid_slice_definitions
        if not slice_definitions:
            print("Error: After filtering, no valid slice definitions remain.")
            return df_input.copy()

    # Shuffle the entire dataset first
    df_sliced = df_sliced.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Create an empty DataFrame with the same columns to store the final sliced data
    final_sliced_df = pd.DataFrame(columns=df_sliced.columns, index=df_sliced.index)

    num_rows = len(df_sliced)
    for i in range(0, num_rows, k):
        bucket_indices = df_sliced.index[i : min(i + k, num_rows)]
        
        if len(bucket_indices) == 0: continue # Should not happen with proper range

        # For each slice, permute its values within the current bucket
        for current_slice_cols in slice_definitions:
            # Ensure we only work with columns present in the current slice and the dataframe
            actual_slice_cols = [col for col in current_slice_cols if col in df_sliced.columns]
            if not actual_slice_cols: continue

            bucket_slice_data = df_sliced.loc[bucket_indices, actual_slice_cols].copy()
            
            # Permute the rows (records) within this slice's data for the current bucket
            permuted_bucket_slice_data = bucket_slice_data.sample(frac=1, random_state=seed + i + len(actual_slice_cols)).reset_index(drop=True)
            
            # Assign permuted data back to the corresponding slice columns in the final DataFrame for this bucket
            # Ensure indices align if final_sliced_df was pre-created with original index
            final_sliced_df.loc[bucket_indices, actual_slice_cols] = permuted_bucket_slice_data.values
    
    # Copy over any columns not in any slice definition from the original shuffled df
    # This ensures columns not part of slicing are still present (though shuffled globally)
    cols_not_in_slices = [col for col in df_sliced.columns if col not in all_slice_cols]
    if cols_not_in_slices:
        final_sliced_df[cols_not_in_slices] = df_sliced[cols_not_in_slices]

    return final_sliced_df


# --- Standalone testing block ---
if __name__ == "__main__":
    print(f"Running {__file__} as a standalone script for testing Slicing logic...")

    # Create a sample DataFrame for testing
    data = {
        'id': range(10),
        'age': [25, 30, 22, 45, 50, 35, 28, 60, 55, 40],
        'job': ['Dev', 'Dev', 'Analyst', 'Manager', 'Manager', 'Analyst', 'Dev', 'CEO', 'CEO', 'Manager'],
        'city': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
        'income': [50, 60, 45, 80, 90, 70, 55, 120, 110, 75] # Sensitive Attribute
    }
    sample_df = pd.DataFrame(data)

    print("\n--- Original Sample DataFrame ---")
    print(sample_df)

    test_k = 3
    test_seed = 42
    # Define slices: one with QIs, one with SA and a related QI
    test_slice_definitions = [
        ['age', 'job'],          # Slice 1
        ['city', 'income']       # Slice 2 (includes SA)
    ]
    # 'id' column is not in any slice

    print(f"\n--- Running Slicing with k={test_k}, seed={test_seed} ---")
    print(f"Slice Definitions: {test_slice_definitions}")

    sliced_df = run_slicing(sample_df.copy(), test_k, test_slice_definitions, test_seed, sa_col='income')

    print("\n--- Sliced Sample DataFrame ---")
    print(sliced_df)
    print("\nNote: 'id' column values should be shuffled globally but not within slice permutations.")
    print("Values within ['age', 'job'] should be permuted together within buckets.")
    print("Values within ['city', 'income'] should be permuted together within buckets, independently of the first slice.")

    # Test with adult dataset if available
    try:
        adult_csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "adult_train_for_arx.csv")
        if os.path.exists(adult_csv_path):
            print(f"\n--- Testing Slicing with Adult Dataset (path: {adult_csv_path}) ---")
            df_adult_raw = pd.read_csv(adult_csv_path, sep=';')
            
            adult_k = 10
            adult_seed = 42
            adult_slice_definitions = [
                ["age", "workclass", "education", "occupation"],
                ["maritalstatus", "relationship", "sex", "income"], # SA in a slice
                ["race", "nativecountry"]
            ]
            
            print(f"Running Slicing on Adult data with k={adult_k}, seed={adult_seed}...")
            adult_sliced = run_slicing(df_adult_raw.copy(), adult_k, adult_slice_definitions, adult_seed, sa_col="income")
            print("\n--- Sliced Adult DataFrame (head) ---")
            print(adult_sliced.head())
        else:
            print(f"\nAdult dataset not found at {adult_csv_path}, skipping Adult data test.")
    except Exception as e:
        print(f"Error during Adult Slicing dataset test: {e}")