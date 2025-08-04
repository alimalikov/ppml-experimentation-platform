import pandas as pd # Ensure pandas is imported
import numpy as np
import os # For standalone testing

def apply_dynamic_rules_to_df(df: pd.DataFrame, cols_to_suppress: list, generalization_rules: dict) -> pd.DataFrame:
    """
    Applies user-defined suppression, perturbation, and generalization rules to a given DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        cols_to_suppress (list): A list of column names to suppress (drop).
        generalization_rules (dict): A dictionary where keys are column names
                                     and values are rule definitions.
                                     Example:
                                     {
                                         'age': { # Example with perturbation and then binning
                                             'perturb_noise': {'method': 'gaussian', 'std_dev_percent': 0.05, 'mean_percent': 0.01},
                                             'bins': [0, 30, 60, np.inf], 
                                             'labels': ['Young', 'Mid', 'Old']
                                         },
                                         'salary': { # Example with only perturbation
                                             'perturb_noise': {'method': 'uniform', 'low_abs': -500, 'high_abs': 500}
                                         },
                                         'education': {'map': {'HS-grad': 'HighSchool', 'Bachelors': 'College'}}
                                     }
    Returns:
        pd.DataFrame: The DataFrame with rules applied.
    """
    df_mod = df.copy()
    # print(f"  Applying dynamic rules: Suppress={cols_to_suppress}, Rules for columns={list(generalization_rules.keys())}")

    # 1. Suppression
    actual_cols_to_drop = [col for col in cols_to_suppress if col in df_mod.columns]
    if actual_cols_to_drop:
        df_mod = df_mod.drop(columns=actual_cols_to_drop)
        # print(f"    > Suppressed columns: {actual_cols_to_drop}")

    # 2. Generalization and Perturbation (per column)
    # print("    > Starting column-specific rules (perturbation/generalization)...")
    for col, rules in generalization_rules.items():
        if col in df_mod.columns: # Check if column still exists
            # print(f"    Processing column: {col} with rules: {rules.keys()}")
            initial_original_nan_mask = df_mod[col].isnull() # NaNs before any operation on this column

            # --- Stage 1: Apply Perturbation (Noise Addition) if defined ---
            if 'perturb_noise' in rules:
                # print(f"      Attempting perturbation for '{col}'")
                # Ensure column is numeric for perturbation
                if not pd.api.types.is_numeric_dtype(df_mod[col]):
                    df_mod[col] = pd.to_numeric(df_mod[col], errors='coerce')
                
                # Proceed only if the column is now numeric (or partially numeric)
                if pd.api.types.is_numeric_dtype(df_mod[col]) and not df_mod[col].isnull().all():
                    noise_params = rules['perturb_noise']
                    method = noise_params.get('method')
                    
                    non_nan_indices = df_mod[col].notna()
                    if non_nan_indices.any():
                        original_values_for_perturb = df_mod.loc[non_nan_indices, col].copy()

                        if method == 'gaussian':
                            mean_percent = noise_params.get('mean_percent', 0.0)
                            std_dev_percent = noise_params.get('std_dev_percent', 0.05) # Default 5%

                            if std_dev_percent < 0:
                                # print(f"      - Warning: std_dev_percent for Gaussian noise on column '{col}' is negative. Using absolute value.")
                                std_dev_percent = abs(std_dev_percent)

                            noise_mean_component = original_values_for_perturb * mean_percent
                            noise_std_dev_component = np.abs(original_values_for_perturb) * std_dev_percent
                            
                            # Ensure scale is not zero for all entries if original values can be zero
                            # A small epsilon can be added to noise_std_dev_component if values are often 0
                            # However, np.random.normal handles scale=0 by returning loc.
                            
                            gaussian_noise = np.random.normal(loc=noise_mean_component, scale=noise_std_dev_component)
                            df_mod.loc[non_nan_indices, col] = original_values_for_perturb + gaussian_noise
                            # print(f"        Applied Gaussian noise to '{col}'.")

                        elif method == 'uniform':
                            low_abs = noise_params.get('low_abs', 0.0)
                            high_abs = noise_params.get('high_abs', 0.0)

                            if low_abs > high_abs:
                                # print(f"      - Warning: Uniform noise low_abs ({low_abs}) > high_abs ({high_abs}) for column '{col}'. Swapping them.")
                                low_abs, high_abs = high_abs, low_abs
                            
                            if low_abs == high_abs: # No noise if range is zero
                                uniform_noise = np.full(len(original_values_for_perturb), low_abs)
                            else:
                                uniform_noise = np.random.uniform(low=low_abs, high=high_abs, size=len(original_values_for_perturb))
                            df_mod.loc[non_nan_indices, col] = original_values_for_perturb + uniform_noise
                            # print(f"        Applied Uniform noise to '{col}'.")
                        else:
                            print(f"      - Unknown perturbation method '{method}' for column '{col}'. Skipping perturbation.")
                # else:
                    # print(f"      - Column '{col}' not suitable for numerical perturbation after coercion attempt.")
            
            # --- Stage 2: Apply Generalization (Binning or Mapping) if defined ---
            # This operates on potentially perturbed data.
            
            # Handling NaNs for generalization: use a mask that includes initial NaNs AND any NaNs created by coercion for perturbation
            current_col_nan_mask_for_gen = initial_original_nan_mask | df_mod[col].isnull()

            if 'bins' in rules and 'labels' in rules:
                # print(f"      Attempting binning for '{col}'")
                try:
                    # Ensure column is numeric before binning (might have been perturbed)
                    if not pd.api.types.is_numeric_dtype(df_mod[col]):
                         df_mod[col] = pd.to_numeric(df_mod[col], errors='coerce')
                    
                    # Update NaN mask again if coercion happened here
                    current_col_nan_mask_for_gen = current_col_nan_mask_for_gen | df_mod[col].isnull()

                    binned_values = pd.cut(df_mod[col], bins=rules['bins'], labels=rules['labels'], right=True, include_lowest=True)
                    
                    if 'numerical_map' in rules:
                        df_mod[col] = binned_values.map(rules['numerical_map'])
                        df_mod[col] = pd.to_numeric(df_mod[col], errors='coerce')
                    else:
                        df_mod[col] = binned_values.astype(str)
                    
                    df_mod.loc[current_col_nan_mask_for_gen, col] = np.nan
                    # print(f"        Binned column '{col}'.")
                except Exception as e:
                    print(f"      - Error generalizing column '{col}' with bins: {e}.")
                    df_mod.loc[initial_original_nan_mask, col] = np.nan # Fallback

            elif 'map' in rules: # Categorical generalization
                # print(f"      Attempting mapping for '{col}'")
                try:
                    df_mod[col] = df_mod[col].astype(str).str.strip() # Ensure string type for mapping
                    # Update NaN mask for strings that represent NaNs
                    current_col_nan_mask_for_gen = current_col_nan_mask_for_gen | df_mod[col].isin(['nan', 'None', '<NA>', '', 'NaN', 'NaT'])
                    
                    df_mod[col] = df_mod[col].map(rules['map']).fillna('Other_UserDefinedMap')
                    df_mod.loc[current_col_nan_mask_for_gen, col] = np.nan
                    # print(f"        Mapped column '{col}'.")
                except Exception as e:
                    print(f"      - Error generalizing column '{col}' with map: {e}.")
                    df_mod.loc[initial_original_nan_mask, col] = np.nan # Fallback
            
            # Ensure original NaNs are preserved if no other rule handled them for this column
            df_mod.loc[initial_original_nan_mask, col] = np.nan
        # else:
            # print(f"    > Warning: Column '{col}' for rules not found in DataFrame.")
    # print("    > Column-specific rules finished.")
    return df_mod

def run_dynamic_manual_rules(df_input: pd.DataFrame, 
                             suppression_list: list, 
                             generalization_rules_dict: dict, 
                             sa_col: str | None = None) -> pd.DataFrame:
    """
    Applies user-defined manual suppression and generalization rules to the DataFrame.
    Args:
        df_input (pd.DataFrame): The input DataFrame.
        suppression_list (list): A list of column names to suppress.
        generalization_rules_dict (dict): A dictionary defining generalization rules
                                          for specific columns. See apply_dynamic_rules_to_df
                                          docstring for format.
        sa_col (str | None, optional): Name of the sensitive attribute column.
                                       This column will be preserved if not part of rules.
    Returns:
        pd.DataFrame: The DataFrame with dynamic manual rules applied.
    """
    if df_input.empty:
        print("Error: Input DataFrame is empty for run_dynamic_manual_rules.")
        return pd.DataFrame()

    df_processed = df_input.copy()
    sensitive_attribute_data = None
    sa_was_part_of_rules = False

    # Separate SA column if specified and exists, to preserve it,
    # unless it's explicitly targeted by the dynamic rules.
    if sa_col and sa_col in df_processed.columns:
        if sa_col in suppression_list or sa_col in generalization_rules_dict:
            sa_was_part_of_rules = True
            # SA will be processed by apply_dynamic_rules_to_df
            df_features = df_processed 
        else:
            # SA is not in rules, separate it
            sensitive_attribute_data = df_processed[sa_col].copy()
            df_features = df_processed.drop(columns=[sa_col])
    else:
        df_features = df_processed # No SA or SA not in df, process all columns

    # Apply the dynamic rules
    df_features_anonymized = apply_dynamic_rules_to_df(df_features, suppression_list, generalization_rules_dict)

    # Reconstruct the DataFrame
    if sensitive_attribute_data is not None and not sa_was_part_of_rules:
        # SA was separated and not part of rules, add it back
        final_df = df_features_anonymized.copy() # Use copy to avoid modifying returned df_features_anonymized
        final_df[sa_col] = sensitive_attribute_data
        
        # Attempt to restore original column order, including the SA column
        original_cols_order = list(df_input.columns)
        current_cols = list(final_df.columns)
        ordered_cols = [col for col in original_cols_order if col in current_cols]
        ordered_cols.extend([col for col in current_cols if col not in ordered_cols]) # Add any new (shouldn't be)
        final_df = final_df[ordered_cols]
    else:
        # SA was part of rules, or no SA to handle separately
        final_df = df_features_anonymized

    return final_df

# --- Standalone testing block ---
if __name__ == "__main__":
    print(f"Running {__file__} as a standalone script for testing Dynamic Manual Rules logic...")

    data = {
        'id': [1,2,3,4,5,6,7],
        'age': [25, 30, 22, 45, 50, np.nan, 35], 
        'city': ['NY', 'LA', 'NY', 'SF', 'LA', 'NY', 'SF'],
        'occupation': ['Dev', 'Dev', 'Analyst', 'Manager', 'Manager', 'Dev', 'Analyst'],
        'salary': [50000, 60000, 45000, 80000, 90000, 55000, 70000],
        'score': [100, 200, 150, 300, 250, np.nan, 180], # New numeric column for perturbation test
        'education': ['Bachelors', 'Masters', 'Bachelors', 'PhD', 'Masters', 'Bachelors', 'PhD']
    }
    sample_df = pd.DataFrame(data)

    print("\n--- Original Sample DataFrame ---")
    print(sample_df)

    # Define dynamic rules for testing
    test_suppression = ['id']
    test_generalization_perturb = {
        'age': { # Perturbation then Binning
            'perturb_noise': {'method': 'gaussian', 'std_dev_percent': 0.1, 'mean_percent': 0.0}, # 10% std dev noise
            'bins': [0, 29, 49, np.inf], 'labels': ['Young', 'Adult', 'Senior']
        },
        'city': {'map': {'NY': 'EastCoast', 'LA': 'WestCoast', 'SF': 'WestCoast'}},
        'salary': { # Only Binning
            'bins': [0, 49999, 79999, np.inf], 'labels': ['Low', 'Medium', 'High']
        },
        'score': { # Only Perturbation (Uniform)
             'perturb_noise': {'method': 'uniform', 'low_abs': -10, 'high_abs': 10}
        }
    }
    test_sa = 'salary' 

    print(f"\n--- Running Dynamic Manual Rules (Test 1) ---")
    print(f"Suppression: {test_suppression}")
    print(f"Rules: {test_generalization_perturb}")
    print(f"Sensitive Attribute (SA): {test_sa}")

    anonymized_df = run_dynamic_manual_rules(sample_df.copy(), 
                                             test_suppression, 
                                             test_generalization_perturb, 
                                             sa_col=test_sa)

    print("\n--- Anonymized Sample DataFrame (Test 1) ---")
    print(anonymized_df)
    print("\nInfo:")
    anonymized_df.info()
    print("\nOriginal 'age' vs Anonymized 'age' (example of perturb then bin):")
    print(pd.concat([sample_df['age'].rename('Original Age'), anonymized_df['age'].rename('Anonymized Age')], axis=1))
    print("\nOriginal 'score' vs Anonymized 'score' (example of uniform perturb):")
    print(pd.concat([sample_df['score'].rename('Original Score'), anonymized_df['score'].rename('Anonymized Score')], axis=1))


    # Test case: SA not part of rules
    test_sa_not_in_rules = 'occupation'
    test_suppression_2 = ['id']
    test_generalization_2 = {
        'age': {'bins': [0, 35, np.inf], 'labels': ['Younger', 'Older']},
        'salary': {'bins': [0, 60000, np.inf], 'labels': ['UpTo60k', 'Over60k']}
    }
    print(f"\n--- Running Dynamic Manual Rules (SA '{test_sa_not_in_rules}' not in rules) ---")
    anonymized_df_2 = run_dynamic_manual_rules(sample_df.copy(),
                                               test_suppression_2,
                                               test_generalization_2,
                                               sa_col=test_sa_not_in_rules)
    print("\n--- Anonymized Sample DataFrame (SA not in rules) ---")
    print(anonymized_df_2)
    print("\nInfo:")
    anonymized_df_2.info()


    # Test with Adult dataset if available
    try:
        adult_csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "adult_train_for_arx.csv")
        if os.path.exists(adult_csv_path):
            print(f"\n--- Testing Dynamic Manual Rules with Adult Dataset (path: {adult_csv_path}) ---")
            df_adult_raw = pd.read_csv(adult_csv_path, sep=';')
            df_adult_raw.dropna(inplace=True) # Simple cleaning for test

            adult_suppress = ['fnlwgt', 'education_num']
            adult_generalize = {
                'age': {'bins': [0, 24, 34, 44, 54, 64, np.inf], 'labels': ['0-24', '25-34', '35-44', '45-54', '55-64', '65+']},
                'hours_per_week': {'bins': [0, 39, 40, np.inf], 'labels': ['Part-time', 'Full-time-40', 'Over-time']},
                'education': {'map': {'Preschool': 'Elem-School', '1st-4th': 'Elem-School', '5th-6th': 'Elem-School', 
                                      '7th-8th': 'Mid-School', '9th': 'HS-Level', '10th': 'HS-Level', '11th': 'HS-Level', '12th': 'HS-Level',
                                      'HS-grad': 'HS-Grad', 'Some-college': 'Some-College', 
                                      'Assoc-acdm': 'Associate', 'Assoc-voc': 'Associate', 
                                      'Bachelors': 'Bachelors', 'Masters': 'Post-Grad', 'Prof-school': 'Post-Grad', 'Doctorate': 'Post-Grad'}}
            }
            adult_sa = 'income'
            
            print("Running Dynamic Manual Rules on Adult data...")
            adult_anonymized = run_dynamic_manual_rules(df_adult_raw.copy(), adult_suppress, adult_generalize, sa_col=adult_sa)
            print("\n--- Anonymized Adult DataFrame (Dynamic Rules) (head) ---")
            print(adult_anonymized.head())
            print("\nInfo:")
            adult_anonymized.info()
        else:
            print(f"\nAdult dataset not found at {adult_csv_path}, skipping Adult data test.")
    except Exception as e:
        print(f"Error during Adult dataset test for Dynamic Manual Rules: {e}")
