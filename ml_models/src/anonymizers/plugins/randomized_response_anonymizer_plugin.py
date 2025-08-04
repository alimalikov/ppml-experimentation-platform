import streamlit as st

import pandas as pd

import numpy as np # For checking numeric types and random

from typing import List, Dict, Any, Set, Optional

import json

import random # For generating random numbers



# For use in the plugin editor / test environment

# When saved to the file system, your app's save logic should convert this to:

# from ..base_anonymizer import Anonymizer

from ..base_anonymizer import Anonymizer



class RandomizedResponseAnonymizer(Anonymizer):

    """

    Applies Randomized Response to a selected sensitive attribute column.

    Users define positive/negative outcomes and probabilities for truthful

    and randomized answers.

    """



    def __init__(self):

        self._name: str = "Randomized Response"

        self._description: str = (

            "Masks sensitive data by perturbing responses based on user-defined probabilities. "

            "Each individual either tells the truth or provides a randomized answer."

        )

        self.default_prob_truth: float = 0.7

        self.default_prob_positive_if_randomizing: float = 0.5



    def get_name(self) -> str:

        """Returns the display name of the anonymization technique."""

        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Perturbation Methods"

    def get_description(self) -> str:

        """Returns a brief description of what the plugin does."""

        return self._description



    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, df_raw: Optional[pd.DataFrame], unique_key_prefix: str) -> Dict[str, Any]:

        """Renders UI elements in the Streamlit sidebar for this plugin."""

        st.sidebar.subheader(f"{self.get_name()} Configuration")



        if not all_cols:

            st.sidebar.warning("No data columns available to configure.")

            return { # Return defaults or empty if no columns

                "selected_column": None,

                "positive_outcome_value": None,

                "negative_outcome_value": None,

                "prob_truth": self.default_prob_truth,

                "prob_positive_if_randomizing": self.default_prob_positive_if_randomizing

            }



        # --- Column Selection ---

        selected_column = st.sidebar.selectbox(

            "Select Sensitive Column for Randomized Response:",

            options=all_cols,

            index=0, # Default to first column if available

            key=f"{unique_key_prefix}_rr_column",

            help="Choose the column containing the sensitive attribute."

        )



        unique_values_in_col: List[Any] = []

        if df_raw is not None and not df_raw.empty and selected_column and selected_column in df_raw.columns:

            try:

                unique_values_in_col = df_raw[selected_column].dropna().unique().tolist()

                if not unique_values_in_col:

                     st.sidebar.warning(f"Column '{selected_column}' has no unique non-NaN values.")

                elif len(unique_values_in_col) == 1:

                     st.sidebar.info(f"Column '{selected_column}' has only one unique value: '{unique_values_in_col[0]}'. Randomized Response may not be very effective.")

            except Exception as e:

                st.sidebar.error(f"Could not get unique values from '{selected_column}': {e}")

        

        # --- Outcome Value Definitions ---

        # Retrieve current session state values or defaults for outcome values

        # These will be strings from text_input, conversion might be needed if original col is numeric

        

        # For simplicity, we'll use text inputs. Dropdowns from unique_values could be complex if many unique values.

        # A more advanced UI might offer dropdowns if len(unique_values_in_col) is small.

        

        positive_outcome_value_str = st.sidebar.text_input(

            "Define 'Positive' Outcome Value for Output:",

            value=str(st.session_state.get(f"{unique_key_prefix}_rr_pos_val", unique_values_in_col[0] if len(unique_values_in_col) > 0 else "Yes")),

            key=f"{unique_key_prefix}_rr_pos_val",

            help="The value representing a 'positive' or 'sensitive' outcome in the randomized output (e.g., 'Yes', '1', 'HasCondition')."

        )



        negative_outcome_value_str = st.sidebar.text_input(

            "Define 'Negative' Outcome Value for Output:",

            value=str(st.session_state.get(f"{unique_key_prefix}_rr_neg_val", unique_values_in_col[1] if len(unique_values_in_col) > 1 else "No")),

            key=f"{unique_key_prefix}_rr_neg_val",

            help="The value representing a 'negative' or 'non-sensitive' outcome in the randomized output (e.g., 'No', '0', 'DoesNotHaveCondition')."

        )

        

        if positive_outcome_value_str == negative_outcome_value_str:

            st.sidebar.error("Positive and Negative outcome values must be different.")



        # --- Probabilities ---

        prob_truth = st.sidebar.slider(

            "P(Tell Truth): Probability of keeping original response",

            min_value=0.0, max_value=1.0,

            value=float(st.session_state.get(f"{unique_key_prefix}_rr_prob_truth", self.default_prob_truth)),

            step=0.01,

            key=f"{unique_key_prefix}_rr_prob_truth",

            help="Probability (0.0 to 1.0) that the original value is reported."

        )



        prob_positive_if_randomizing = st.sidebar.slider(

            "P(Report Positive | Randomizing): Probability of reporting 'Positive Outcome'",

            min_value=0.0, max_value=1.0,

            value=float(st.session_state.get(f"{unique_key_prefix}_rr_prob_pos_rand", self.default_prob_positive_if_randomizing)),

            step=0.01,

            key=f"{unique_key_prefix}_rr_prob_pos_rand",

            help="If randomizing (i.e., not telling truth), this is the probability of reporting the 'Positive Outcome Value'."

        )

        

        return {

            "selected_column": selected_column,

            "positive_outcome_value_str": positive_outcome_value_str,

            "negative_outcome_value_str": negative_outcome_value_str,

            "prob_truth": prob_truth,

            "prob_positive_if_randomizing": prob_positive_if_randomizing

        }



    def _try_cast_value(self, value_str: str, target_type_example: Any) -> Any:

        """Attempts to cast string value to the type of target_type_example."""

        if isinstance(target_type_example, (np.bool_, bool)):

            if value_str.lower() in ["true", "1", "yes"]: return True

            if value_str.lower() in ["false", "0", "no"]: return False

        elif isinstance(target_type_example, (np.integer, int)):

            try: return int(value_str)

            except ValueError: pass

        elif isinstance(target_type_example, (np.floating, float)):

            try: return float(value_str)

            except ValueError: pass

        return value_str # Default to string if casting fails or not a recognized type



    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: Optional[str]) -> pd.DataFrame:

        """Applies Randomized Response to the selected column."""

        if df_input.empty:

            st.warning(f"{self.get_name()}: Input DataFrame is empty. Returning as is.")

            return df_input



        df_anonymized = df_input.copy()

        

        selected_col: Optional[str] = parameters.get("selected_column")

        positive_outcome_str: Optional[str] = parameters.get("positive_outcome_value_str")

        negative_outcome_str: Optional[str] = parameters.get("negative_outcome_value_str")

        prob_truth: float = parameters.get("prob_truth", self.default_prob_truth)

        prob_pos_rand: float = parameters.get("prob_positive_if_randomizing", self.default_prob_positive_if_randomizing)



        if not selected_col or selected_col not in df_anonymized.columns:

            st.error(f"{self.get_name()}: Selected column '{selected_col}' not found or not specified. Aborting.")

            return df_input

        

        if positive_outcome_str is None or negative_outcome_str is None:

            st.error(f"{self.get_name()}: Positive or Negative outcome values not defined. Aborting.")

            return df_input



        if positive_outcome_str == negative_outcome_str:

            st.error(f"{self.get_name()}: Positive and Negative outcome values must be different. Aborting.")

            return df_input



        # Attempt to cast outcome values to the original column's dtype for consistency

        # Get an example non-NaN value from the original column to infer type

        original_col_series = df_anonymized[selected_col].dropna()

        target_type_example = original_col_series.iloc[0] if not original_col_series.empty else "string_fallback"



        positive_outcome = self._try_cast_value(positive_outcome_str, target_type_example)

        negative_outcome = self._try_cast_value(negative_outcome_str, target_type_example)



        # Apply Randomized Response row by row

        new_column_values = []

        for original_value in df_anonymized[selected_col]:

            if pd.isna(original_value): # Preserve NaNs

                new_column_values.append(original_value)

                continue



            if random.random() <= prob_truth: # Tell the truth

                new_column_values.append(original_value)

            else: # Randomize response

                if random.random() <= prob_pos_rand:

                    new_column_values.append(positive_outcome)

                else:

                    new_column_values.append(negative_outcome)

        

        df_anonymized[selected_col] = new_column_values

        # Ensure the column dtype is consistent if possible, especially if all values became the same type

        try:

            df_anonymized[selected_col] = pd.Series(new_column_values, index=df_anonymized.index, dtype=type(target_type_example) if target_type_example != "string_fallback" else None)

        except Exception: # Fallback if type conversion is problematic

            df_anonymized[selected_col] = pd.Series(new_column_values, index=df_anonymized.index)





        st.success(f"{self.get_name()}: Randomized Response applied to column '{selected_col}'.")

        return df_anonymized



    def build_config_export(self, unique_key_prefix: str, sa_col: Optional[str]) -> Dict[str, Any]:

        """Builds the configuration dictionary for export."""

        return {

            "selected_column": st.session_state.get(f"{unique_key_prefix}_rr_column"),

            "positive_outcome_value_str": st.session_state.get(f"{unique_key_prefix}_rr_pos_val"),

            "negative_outcome_value_str": st.session_state.get(f"{unique_key_prefix}_rr_neg_val"),

            "prob_truth": st.session_state.get(f"{unique_key_prefix}_rr_prob_truth", self.default_prob_truth),

            "prob_positive_if_randomizing": st.session_state.get(f"{unique_key_prefix}_rr_prob_pos_rand", self.default_prob_positive_if_randomizing)

        }



    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):

        """Applies imported configuration to the UI elements."""

        st.session_state[f"{unique_key_prefix}_rr_column"] = config_params.get("selected_column")

        st.session_state[f"{unique_key_prefix}_rr_pos_val"] = config_params.get("positive_outcome_value_str")

        st.session_state[f"{unique_key_prefix}_rr_neg_val"] = config_params.get("negative_outcome_value_str")

        st.session_state[f"{unique_key_prefix}_rr_prob_truth"] = config_params.get("prob_truth", self.default_prob_truth)

        st.session_state[f"{unique_key_prefix}_rr_prob_pos_rand"] = config_params.get("prob_positive_if_randomizing", self.default_prob_positive_if_randomizing)



    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):

        """Renders the export button for this plugin's configuration."""

        json_string = json.dumps(config_to_export, indent=4)

        st.sidebar.download_button(

            label=f"Export {self.get_name()} Config",

            data=json_string,

            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json",

            mime="application/json",

            key=f"{unique_key_prefix}_export_button_rr" # Ensure unique key

        )



    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:

        """Renders the anonymize button for this plugin."""

        return st.button(f"Anonymize with {self.get_name()}", key=f"{unique_key_prefix}_anonymize_button_rr") # Ensure unique key



# Factory function that the main application will call

def get_plugin():

    return RandomizedResponseAnonymizer()