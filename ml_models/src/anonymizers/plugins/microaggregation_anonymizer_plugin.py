import streamlit as st

import pandas as pd

from typing import List, Dict, Any, Set

import json

import numpy as np # For checking numeric types



# For use in the plugin editor / test environment

# When saved to the file system, your app's save logic should convert this to:

# from ..base_anonymizer import Anonymizer

from ..base_anonymizer import Anonymizer



class MicroaggregationAnonymizer(Anonymizer):

    """

    Performs microaggregation on selected numerical quasi-identifier columns.

    For each selected column, data is sorted, grouped into sets of at least 'k' records,

    and original values are replaced by the group mean.

    """



    def __init__(self):

        self._name: str = "Microaggregation"

        self._description: str = (

            "Anonymizes data by grouping records and replacing values in selected "

            "numerical columns with the mean of their group. (K-value defines minimum group size)"

        )

        self.default_k_value: int = 3



    def get_name(self) -> str:

        """Returns the display name of the anonymization technique."""

        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Suppression & Generalization"

    def get_description(self) -> str:

        """Returns a brief description of what the plugin does."""

        return self._description



    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, df_raw: pd.DataFrame | None, unique_key_prefix: str) -> Dict[str, Any]:

        """Renders UI elements in the Streamlit sidebar for this plugin."""

        st.sidebar.subheader(f"{self.get_name()} Configuration")



        numeric_cols: List[str] = []

        if df_raw is not None and not df_raw.empty:

            numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()

            if sa_col_to_pass and sa_col_to_pass in numeric_cols: # Exclude SA col from QI options

                numeric_cols.remove(sa_col_to_pass)

        

        if not all_cols:

            st.sidebar.warning("No data columns available to configure.")

        elif not numeric_cols:

            st.sidebar.warning("No numerical columns available for microaggregation (or SA column is the only numeric one).")



        # Retrieve current session state values or defaults

        default_selected_qis = st.session_state.get(f"{unique_key_prefix}_microagg_qi_cols", [])

        # Filter default_selected_qis to only include available numeric_cols

        valid_default_qis = [col for col in default_selected_qis if col in numeric_cols]





        selected_qi_columns = st.sidebar.multiselect(

            "Select Numerical QI Columns for Microaggregation:",

            options=numeric_cols,

            default=valid_default_qis,

            key=f"{unique_key_prefix}_microagg_qi_cols",

            help="Choose one or more numerical columns to apply microaggregation."

        )



        k_value = st.sidebar.number_input(

            "K-value (Minimum Group Size):",

            min_value=2,

            value=st.session_state.get(f"{unique_key_prefix}_microagg_k_value", self.default_k_value),

            step=1,

            format="%d",

            key=f"{unique_key_prefix}_microagg_k_value",

            help="Each group will have at least K records. Recommended K >= 3."

        )

        

        return {

            "selected_qi_columns": selected_qi_columns,

            "k_value": k_value

        }



    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:

        """Applies microaggregation to the selected columns."""

        if df_input.empty:

            st.warning(f"{self.get_name()}: Input DataFrame is empty. Returning as is.")

            return df_input



        df_anonymized = df_input.copy()

        

        selected_qis: List[str] = parameters.get("selected_qi_columns", [])

        k_value: int = parameters.get("k_value", self.default_k_value)



        if not selected_qis:

            st.warning(f"{self.get_name()}: No QI columns selected for microaggregation. Returning original data.")

            return df_anonymized



        if k_value < 2:

            st.error(f"{self.get_name()}: K-value must be at least 2. Using default K={self.default_k_value} instead.")

            k_value = self.default_k_value

        

        processed_cols_count = 0

        for qi_col in selected_qis:

            if qi_col not in df_anonymized.columns:

                st.warning(f"{self.get_name()}: Column '{qi_col}' not found in the DataFrame. Skipping.")

                continue



            if not pd.api.types.is_numeric_dtype(df_anonymized[qi_col]):

                st.warning(f"{self.get_name()}: Column '{qi_col}' is not numeric. Skipping microaggregation for this column.")

                continue



            # Handle NaN values: fill with mean or median before sorting, or they might affect sorting/grouping.

            # For simplicity here, we'll proceed, but NaNs might cluster. A more robust approach might impute them.

            if df_anonymized[qi_col].isnull().any():

                st.info(f"{self.get_name()}: Column '{qi_col}' contains NaN values. These may affect grouping and mean calculation.")



            # Create a series to store new microaggregated values

            # Initialize with original values to handle any records not covered by loop (should not happen with correct loop)

            new_values = df_anonymized[qi_col].copy() 

            

            # Get original indices sorted by the values in the current QI column

            # Drop NaNs for sorting to avoid issues, their original values will be preserved by new_values init

            # or handled by mean calculation if they fall into a group.

            valid_series = df_anonymized[qi_col].dropna()

            if valid_series.empty:

                st.warning(f"{self.get_name()}: Column '{qi_col}' has no valid numeric data after dropping NaNs. Skipping.")

                continue

            

            sorted_indices = valid_series.sort_values().index

            

            num_records_for_col = len(sorted_indices)

            if num_records_for_col == 0: # Should be caught by valid_series.empty

                continue



            i = 0

            while i < num_records_for_col:

                end_slice = i + k_value

                

                # If the remaining records after this group would be less than k_value,

                # and this is not the only group, merge them into the current group.

                if (num_records_for_col - end_slice) < k_value and (num_records_for_col - end_slice) > 0:

                    end_slice = num_records_for_col 

                    

                current_group_indices = sorted_indices[i:end_slice]

                

                if len(current_group_indices) > 0:

                    group_actual_values = df_anonymized.loc[current_group_indices, qi_col]

                    mean_val = group_actual_values.mean() # Mean of the original values in the group

                    

                    # Update the temporary series with the mean for these original indices

                    new_values.loc[current_group_indices] = mean_val

                

                i = end_slice

            

            # Assign the microaggregated values back to the DataFrame

            df_anonymized[qi_col] = new_values

            processed_cols_count +=1



        if processed_cols_count > 0:

            st.success(f"{self.get_name()}: Microaggregation applied to {processed_cols_count} column(s) with K={k_value}.")

        else:

            st.info(f"{self.get_name()}: No columns were processed with microaggregation.")

            

        return df_anonymized



    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:

        """Builds the configuration dictionary for export."""

        return {

            "selected_qi_columns": st.session_state.get(f"{unique_key_prefix}_microagg_qi_cols", []),

            "k_value": st.session_state.get(f"{unique_key_prefix}_microagg_k_value", self.default_k_value)

        }



    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):

        """Applies imported configuration to the UI elements."""

        st.session_state[f"{unique_key_prefix}_microagg_qi_cols"] = config_params.get("selected_qi_columns", [])

        st.session_state[f"{unique_key_prefix}_microagg_k_value"] = config_params.get("k_value", self.default_k_value)



    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):

        """Renders the export button for this plugin's configuration."""

        json_string = json.dumps(config_to_export, indent=4)

        st.sidebar.download_button(

            label=f"Export {self.get_name()} Config",

            data=json_string,

            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json",

            mime="application/json",

            key=f"{unique_key_prefix}_export_button_microagg" # Ensure unique key

        )



    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:

        """Renders the anonymize button for this plugin."""

        return st.button(f"Anonymize with {self.get_name()}", key=f"{unique_key_prefix}_anonymize_button_microagg") # Ensure unique key



# Factory function that the main application will call

def get_plugin():

    return MicroaggregationAnonymizer()