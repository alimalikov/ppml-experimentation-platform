import streamlit as st
import pandas as pd
import numpy as np # Added for run_slicing
import json
from typing import List # Added for run_slicing
from ..base_anonymizer import Anonymizer

# --- Embedded Slicing Logic ---
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
        # Use st.warning or st.error for user feedback in a Streamlit context if appropriate
        # For now, keeping print for backend logic, but consider Streamlit feedback for user-facing errors.
        print("Error: Input DataFrame is empty for run_slicing.")
        return pd.DataFrame()
    if not slice_definitions:
        print("Error: No slice definitions provided for run_slicing.")
        return df_input.copy() # Return a copy as per original logic
    if k <= 0:
        print("Error: k-value must be positive for run_slicing.")
        return df_input.copy() # Return a copy

    df_sliced_internal = df_input.copy() # Use a different name to avoid confusion with outer scope if any
    
    # Validate slice definitions against DataFrame columns
    all_slice_cols_internal = [col for slice_group in slice_definitions for col in slice_group]
    missing_cols_internal = [col for col in all_slice_cols_internal if col not in df_sliced_internal.columns]
    
    if missing_cols_internal:
        # In a plugin context, st.warning might be better than print for user feedback
        st.warning(f"Slicing: The following columns defined in slices are not in the DataFrame and will be ignored: {missing_cols_internal}")
        # Filter out missing columns from slice_definitions
        valid_slice_definitions_internal = []
        for slice_group in slice_definitions:
            valid_group = [col for col in slice_group if col in df_sliced_internal.columns]
            if valid_group:
                valid_slice_definitions_internal.append(valid_group)
        slice_definitions = valid_slice_definitions_internal # Re-assign to the corrected list
        if not slice_definitions:
            st.error("Slicing Error: After filtering, no valid slice definitions remain.")
            return df_input.copy() # Return original data copy

    # Shuffle the entire dataset first
    df_sliced_internal = df_sliced_internal.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Create an empty DataFrame with the same columns to store the final sliced data
    final_sliced_df_internal = pd.DataFrame(columns=df_sliced_internal.columns, index=df_sliced_internal.index)

    num_rows = len(df_sliced_internal)
    for i in range(0, num_rows, k):
        bucket_indices = df_sliced_internal.index[i : min(i + k, num_rows)]
        
        if len(bucket_indices) == 0: continue

        for current_slice_cols_group in slice_definitions: # Iterate through the corrected slice_definitions
            actual_cols_in_group = [col for col in current_slice_cols_group if col in df_sliced_internal.columns]
            if not actual_cols_in_group: continue

            bucket_slice_data = df_sliced_internal.loc[bucket_indices, actual_cols_in_group].copy()
            
            permuted_bucket_slice_data = bucket_slice_data.sample(frac=1, random_state=seed + i + len(actual_cols_in_group)).reset_index(drop=True)
            
            final_sliced_df_internal.loc[bucket_indices, actual_cols_in_group] = permuted_bucket_slice_data.values
    
    # Copy over any columns not in any slice definition from the original shuffled df
    cols_not_in_any_slice = [col for col in df_sliced_internal.columns if col not in all_slice_cols_internal]
    if cols_not_in_any_slice:
        final_sliced_df_internal[cols_not_in_any_slice] = df_sliced_internal[cols_not_in_any_slice]

    return final_sliced_df_internal
# --- End of Embedded Slicing Logic ---


class SlicingPlugin(Anonymizer):
    """
    Plugin for Slicing based on defined column groups and bucket sizes.
    """

    def __init__(self):
        self._name = "Slicing (Column Groups)"

    def get_name(self) -> str:
        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Suppression & Generalization"

    def get_sidebar_ui(self, all_cols: list, sa_col_to_pass: str | None, df_raw: pd.DataFrame, unique_key_prefix: str) -> dict:
        st.sidebar.header(f"{self.get_name()} Configuration")

        k_key = f"{unique_key_prefix}_k_bucket_size"
        seed_key = f"{unique_key_prefix}_seed"
        slice_defs_key = f"{unique_key_prefix}_slice_definitions_str"

        min_k = 1
        max_k_val = min_k
        if df_raw is not None and not df_raw.empty:
            max_k_val = max(min_k, len(df_raw))
        
        default_k = st.session_state.get(k_key, max(min_k, 3))
        clamped_default_k = max(min_k, min(default_k, max_k_val))
        k_bucket_size = st.sidebar.slider(
            "Select Bucket Size (k):",
            min_value=min_k,
            max_value=max_k_val,
            value=clamped_default_k,
            step=1,
            key=k_key,
            help="Number of rows grouped into a bucket before permuting column slices."
        )

        default_seed = st.session_state.get(seed_key, 42)
        seed = st.sidebar.number_input(
            "Random Seed:",
            min_value=0,
            value=default_seed,
            step=1,
            key=seed_key,
            help="Seed for reproducible shuffling and permutations."
        )

        default_slice_defs_str = st.session_state.get(slice_defs_key, '') # Default to empty string
        # Auto-generate a simple default if empty and columns exist, and it's the first time (or empty)
        if not default_slice_defs_str and all_cols:
            qis = [col for col in all_cols if col != sa_col_to_pass]
            temp_defs = []
            if qis:
                temp_defs.append(qis)
            if sa_col_to_pass and sa_col_to_pass in all_cols: # Ensure SA is valid
                # Check if SA is already part of a QI group (less likely with this simple default)
                sa_in_qis = any(sa_col_to_pass in group for group in temp_defs)
                if not sa_in_qis:
                    temp_defs.append([sa_col_to_pass])
            
            # Only update if temp_defs actually has content
            if temp_defs:
                 default_slice_defs_str = json.dumps(temp_defs)
            else: # If no QIs and no SA, default to empty list string
                 default_slice_defs_str = '[]'
            st.session_state[slice_defs_key] = default_slice_defs_str # Store the generated default

        slice_definitions_str = st.sidebar.text_area(
            "Slice Definitions (JSON list of lists):",
            value=default_slice_defs_str, # Use the potentially auto-generated default
            key=slice_defs_key,
            height=150,
            help='Define column groups. E.g., [["colA", "colB"], ["colC", "SA_COL"]]. Use double quotes for names.'
        )
        st.sidebar.caption("Example: `[[\"age\", \"job\"], [\"city\", \"income\"]]`")

        parsed_slice_defs = []
        error_message = ""
        try:
            parsed_slice_defs = json.loads(slice_definitions_str)
            if not isinstance(parsed_slice_defs, list):
                raise ValueError("Slice definitions must be a list.")
            for item in parsed_slice_defs:
                if not isinstance(item, list):
                    raise ValueError("Each slice group must be a list of column names.")
                for col_name in item:
                    if not isinstance(col_name, str):
                        raise ValueError("Column names in slices must be strings.")
        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON for Slice Definitions: {e}"
        except ValueError as e:
            error_message = f"Invalid format for Slice Definitions: {e}"
        
        if error_message:
            st.sidebar.error(error_message)

        return {
            "k_bucket_size": k_bucket_size,
            "seed": seed,
            "slice_definitions_str": slice_definitions_str,
        }

    def anonymize(self, df_input: pd.DataFrame, parameters: dict, sa_col: str | None) -> pd.DataFrame:
        k_val = parameters.get("k_bucket_size", 3)
        seed_val = parameters.get("seed", 42)
        slice_definitions_str = parameters.get("slice_definitions_str", "[]")

        slice_definitions = []
        try:
            slice_definitions = json.loads(slice_definitions_str)
            if not isinstance(slice_definitions, list):
                st.error("Slicing Error: Slice definitions must be a list of lists (e.g., [[\"colA\"], [\"colB\"]]).")
                return pd.DataFrame()
            for item in slice_definitions:
                if not isinstance(item, list):
                     st.error("Slicing Error: Each slice group within definitions must be a list of column names.")
                     return pd.DataFrame()
                for col_name in item:
                    if not isinstance(col_name, str):
                        st.error("Slicing Error: Column names within slice groups must be strings.")
                        return pd.DataFrame()
        except json.JSONDecodeError as e:
            st.error(f"Slicing Error: Could not parse Slice Definitions string: {e}. Please provide valid JSON.")
            return pd.DataFrame()
        
        if not slice_definitions and not df_input.empty:
            st.warning("Slicing Warning: No slice definitions provided or parsed correctly. The Slicing algorithm might return the original data structure or an error if it requires definitions.")

        # Call the embedded run_slicing function
        return run_slicing(
            df_input=df_input.copy(),
            k=k_val,
            slice_definitions=slice_definitions,
            seed=seed_val,
            sa_col=sa_col
        )

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> dict:
        slice_defs_str = st.session_state.get(f"{unique_key_prefix}_slice_definitions_str", "[]")
        return {
            "k_bucket_size": st.session_state.get(f"{unique_key_prefix}_k_bucket_size", 3),
            "seed": st.session_state.get(f"{unique_key_prefix}_seed", 42),
            "slice_definitions_str": slice_defs_str,
        }

    def apply_config_import(self, config_params: dict, all_cols: list, unique_key_prefix: str):
        st.session_state[f"{unique_key_prefix}_k_bucket_size"] = config_params.get("k_bucket_size", 3)
        st.session_state[f"{unique_key_prefix}_seed"] = config_params.get("seed", 42)
        
        slice_defs_str = config_params.get("slice_definitions_str", "[]")
        st.session_state[f"{unique_key_prefix}_slice_definitions_str"] = slice_defs_str
        
        try:
            json.loads(slice_defs_str)
        except json.JSONDecodeError:
            st.warning(f"Imported slice definitions for Slicing are not valid JSON: {slice_defs_str[:100]}...")

def get_plugin():
    return SlicingPlugin()