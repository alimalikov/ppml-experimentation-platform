import streamlit as st
import pandas as pd
from abc import ABC, abstractmethod
import json

class Anonymizer(ABC):
    """
    Abstract Base Class for anonymization plugins.
    Each plugin should inherit from this class and implement its abstract methods.
    It also provides common helper methods for UI elements like buttons.
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the display name of the anonymization technique.
        This name will be used in the UI, like in the technique selection dropdown.
        Example: "Mondrian k-Anonymity"
        """
        pass

    def get_category(self) -> str:
        """
        Returns the category of the anonymization technique.
        This is used to group techniques in the UI for better organization.
        
        Standard categories include:
        - "Privacy Models" (k-anonymity, l-diversity, t-closeness, etc.)
        - "Differential Privacy" (various DP mechanisms)
        - "Generative Models" (GANs, VAEs, synthetic data generation)
        - "Perturbation Methods" (noise addition, randomization)
        - "Suppression & Generalization" (data hiding, generalization)
        - "Utility Preserving" (format-preserving encryption, etc.)
        - "Advanced Techniques" (federated learning, homomorphic encryption)
        - "Custom/Experimental" (user-defined or experimental methods)
        
        Default implementation returns "Other" if not overridden.
        """
        return "Other"

    @abstractmethod
    def get_sidebar_ui(self, all_cols: list, sa_col_to_pass: str | None, df_raw: pd.DataFrame, unique_key_prefix: str) -> dict:
        """
        Renders the technique-specific UI elements (widgets) in the Streamlit sidebar.
        This method is responsible for creating sliders, multiselects, text inputs, etc.,
        that are needed to configure this specific anonymization technique.

        It should use `st.session_state` with keys prefixed by `unique_key_prefix`
        to manage its widget states. This ensures that widget states persist across reruns
        and that keys do not clash between different plugins or other parts of the app.

        Args:
            all_cols (list): List of all column names from the uploaded DataFrame.
                             Used to populate dropdowns for column selection.
            sa_col_to_pass (str | None): The currently selected sensitive attribute column.
                                         Plugins might use this for context or validation.
            df_raw (pd.DataFrame): The raw uploaded DataFrame. Plugins might use this
                                   for context like data types, number of rows, etc.,
                                   to set sensible defaults or ranges for their widgets.
            unique_key_prefix (str): A unique string prefix for this plugin (e.g., "mondrian_plugin").
                                     All st.session_state keys and widget keys created by this
                                     plugin should use this prefix (e.g., f"{unique_key_prefix}_k_value").

        Returns:
            dict: A dictionary of parameters collected from the UI. These parameters
                  will be passed to the `anonymize` method.
                  Example: {"k": 5, "qi_cols": ["age", "zipcode"]}
        """
        pass

    @abstractmethod
    def anonymize(self, df_input: pd.DataFrame, parameters: dict, sa_col: str | None) -> pd.DataFrame:
        """
        Performs the actual anonymization logic using the provided DataFrame and parameters.

        Args:
            df_input (pd.DataFrame): The DataFrame to be anonymized. It's recommended
                                     to work on a copy (e.g., `df_input.copy()`) if modifications
                                     are made in place.
            parameters (dict): A dictionary of parameters collected from `get_sidebar_ui`
                               (the return value of that method).
            sa_col (str | None): The selected sensitive attribute column, if any.

        Returns:
            pd.DataFrame: The anonymized DataFrame. If anonymization fails or is not
                          possible with the given parameters, it might return an empty
                          DataFrame or raise an exception (which should be handled by the caller).
        """
        pass

    @abstractmethod
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> dict:
        """
        Builds the technique-specific part of the configuration for export to a JSON file.
        This method should read the current values of its parameters, typically from
        `st.session_state` using the `unique_key_prefix` for its keys.

        Args:
            unique_key_prefix (str): The unique prefix used for this plugin's session state keys.
            sa_col (str | None): The currently selected sensitive attribute. (Note: The main app
                                 will store this at the top level of the config; this is passed
                                 for context if the plugin needs it to decide what to export).

        Returns:
            dict: A dictionary containing the parameters specific to this anonymization
                  technique that should be saved in the configuration file. This dictionary
                  will typically be stored under a "parameters" key in the main config JSON.
                  Example: {"k": 5, "qi_cols": ["age", "zipcode"]}
        """
        pass

    @abstractmethod
    def apply_config_import(self, config_params: dict, all_cols: list, unique_key_prefix: str):
        """
        Applies imported configuration parameters to the Streamlit session state.
        This ensures that when a configuration file is loaded, the plugin's UI widgets
        are updated to reflect the imported settings.

        Args:
            config_params (dict): The 'parameters' part of the imported configuration JSON
                                  that is specific to this technique.
            all_cols (list): List of all column names from the currently loaded DataFrame.
                             Used for validating imported column names (e.g., ensuring
                             an imported QI column still exists in the current dataset).
            unique_key_prefix (str): The unique prefix for this plugin's session state keys.
                                     The method should write to `st.session_state` using
                                     keys like `st.session_state[f"{unique_key_prefix}_k_value"]`.
        """
        pass

    # --- Helper methods (non-abstract, can be used directly by plugins or overridden) ---

    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        """
        Renders a standard 'Anonymize with [Technique Name]' button in the sidebar.
        Plugins can call this method from their `get_sidebar_ui` or the main app can call it.

        Args:
            unique_key_prefix (str): The unique prefix for this plugin.

        Returns:
            bool: True if the button was clicked in this Streamlit rerun, False otherwise.
        """
        return st.sidebar.button(
            f"Anonymize with {self.get_name()}",
            type="primary",
            key=f"{unique_key_prefix}_anonymize_button"
        )

    def get_export_button_ui(self, full_config_to_export: dict, unique_key_prefix: str):
        """
        Renders a standard 'Export [Technique Name] Config' button.
        The main application will typically call this after the plugin has
        contributed its parameters via `build_config_export`.

        Args:
            full_config_to_export (dict): The complete configuration dictionary
                                          (including 'technique', 'sa_col', and the
                                          plugin-specific 'parameters') to be saved.
            unique_key_prefix (str): The unique prefix for this plugin.
        """
        file_name_safe_technique_name = self.get_name().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config",
            data=json.dumps(full_config_to_export, indent=2).encode("utf-8"),
            file_name=f"{file_name_safe_technique_name}_config.json",
            mime="application/json",
            key=f"{unique_key_prefix}_export_button",
        )