import streamlit as st
import pandas as pd
from ..base_anonymizer import Anonymizer # Relative import to get Anonymizer from parent directory
from ..mondrian_anonymizer import run_mondrian # Relative import for the core Mondrian logic

class MondrianPlugin(Anonymizer):
    """
    Plugin for Mondrian k-Anonymity.
    """

    def __init__(self):
        """Initializes the plugin, setting its name."""
        self._name = "Mondrian k-Anonymity"

    def get_name(self) -> str:
        """Returns the display name of the anonymization technique."""
        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Suppression & Generalization"

    def get_sidebar_ui(self, all_cols: list, sa_col_to_pass: str | None, df_raw: pd.DataFrame, unique_key_prefix: str) -> dict:
        """
        Renders the Mondrian-specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"{self.get_name()} Configuration") # Use self.get_name() for consistency

        # Define session state keys using the unique_key_prefix
        qi_key = f"{unique_key_prefix}_qi_cols"
        numeric_qi_key = f"{unique_key_prefix}_numeric_qi_cols"
        k_key = f"{unique_key_prefix}_k_value"

        # --- QI Columns Multiselect ---
        # Get default from session state if it exists, otherwise empty list
        default_qi_cols = st.session_state.get(qi_key, [])
        # Filter default_qi_cols to ensure they are present in the current all_cols
        valid_default_qi_cols = [col for col in default_qi_cols if col in all_cols]

        selected_qi_cols = st.sidebar.multiselect(
            "Select Quasi-Identifier (QI) columns:",
            options=all_cols,
            default=valid_default_qi_cols,
            key=qi_key # Streamlit uses this key to manage the widget's state
        )
        # No need to explicitly set st.session_state[qi_key] = selected_qi_cols here,
        # Streamlit handles it because of the 'key' argument.

        # --- Numeric QI Columns Multiselect (derived from selected QIs) ---
        options_for_numeric_qi = selected_qi_cols # Numeric QIs must be a subset of selected QIs
        default_numeric_qis = st.session_state.get(numeric_qi_key, [])
        # Filter default_numeric_qis to ensure they are valid options
        valid_default_numeric_qis = [col for col in default_numeric_qis if col in options_for_numeric_qi]

        selected_numeric_qis = st.sidebar.multiselect(
            "Select NUMERIC QI columns (from chosen QIs):",
            options=options_for_numeric_qi,
            default=valid_default_numeric_qis,
            key=numeric_qi_key
        )

        # --- k-value Slider ---
        min_k = 2
        # Determine a sensible max_k based on DataFrame size
        max_k_val = min_k
        if df_raw is not None and not df_raw.empty:
            max_k_val = max(min_k, min(50, len(df_raw) // 2 if len(df_raw) > 3 else min_k)) # Ensure len(df_raw)//2 is at least min_k

        default_k = st.session_state.get(k_key, min_k)
        # Clamp default_k to be within the current min_k and max_k_val
        clamped_default_k = max(min_k, min(default_k, max_k_val))

        k_value = st.sidebar.slider(
            "Select k-value:",
            min_value=min_k,
            max_value=max_k_val,
            value=clamped_default_k,
            step=1,
            key=k_key,
            help=f"k must be between {min_k} and {max_k_val} for the current dataset size."
        )

        # Return the collected parameters
        return {
            "qi_cols": selected_qi_cols,       # These are read directly from widget return values
            "numeric_qis": selected_numeric_qis, # which reflect st.session_state due to 'key'
            "k": k_value,
        }

    def anonymize(self, df_input: pd.DataFrame, parameters: dict, sa_col: str | None) -> pd.DataFrame:
        """
        Performs Mondrian k-anonymization.
        """
        qi_cols = parameters.get("qi_cols", [])
        numeric_qis = parameters.get("numeric_qis", [])
        k_val = parameters.get("k", 2) # Default k to 2 if not provided

        if not qi_cols:
            st.error("Mondrian Error: Please select at least one Quasi-Identifier (QI) column.")
            return pd.DataFrame() # Return empty DataFrame on configuration error

        if k_val < 2:
            st.error("Mondrian Error: k-value must be at least 2.")
            return pd.DataFrame()

        if k_val > len(df_input):
            st.error(f"Mondrian Error: k-value ({k_val}) cannot be greater than the number of rows in the dataset ({len(df_input)}).")
            return pd.DataFrame()

        # Call the existing run_mondrian function
        return run_mondrian(
            df_input=df_input.copy(), # Pass a copy to avoid modifying the original df_raw
            qi_cols=qi_cols,
            numeric_qis=numeric_qis,
            k=k_val,
            sa_col=sa_col
        )

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> dict:
        """
        Builds the Mondrian-specific part of the configuration for export.
        It reads values from st.session_state using the keys defined in get_sidebar_ui.
        """
        return {
            "qi_cols": st.session_state.get(f"{unique_key_prefix}_qi_cols", []),
            "numeric_qis": st.session_state.get(f"{unique_key_prefix}_numeric_qi_cols", []),
            "k": st.session_state.get(f"{unique_key_prefix}_k_value", 2), # Default to 2 if not found
        }

    def apply_config_import(self, config_params: dict, all_cols: list, unique_key_prefix: str):
        """
        Applies imported Mondrian configuration parameters to the Streamlit session state.
        """
        qi_key = f"{unique_key_prefix}_qi_cols"
        numeric_qi_key = f"{unique_key_prefix}_numeric_qi_cols"
        k_key = f"{unique_key_prefix}_k_value"

        # Set QI columns, ensuring they exist in the current dataset's all_cols
        imported_qi_cols = config_params.get("qi_cols", [])
        st.session_state[qi_key] = [col for col in imported_qi_cols if col in all_cols]

        # Set Numeric QI columns, ensuring they are a subset of the (potentially filtered) qi_cols
        # and also exist in all_cols (redundant check but safe)
        imported_numeric_qis = config_params.get("numeric_qis", [])
        st.session_state[numeric_qi_key] = [
            col for col in imported_numeric_qis
            if col in st.session_state[qi_key] and col in all_cols
        ]

        st.session_state[k_key] = config_params.get("k", 2) # Default to 2 if not found

# Factory function for the plugin loader
def get_plugin():
    """Returns an instance of the MondrianPlugin."""
    return MondrianPlugin()