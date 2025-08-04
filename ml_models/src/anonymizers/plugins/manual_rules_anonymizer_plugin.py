import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import json
import uuid
import traceback
import logging

from ..base_anonymizer import Anonymizer

class ManualRulesAnonymizer(Anonymizer):
    """
    Applies user-defined manual rules for suppression and generalization.
    Rules can be configured for specific columns.
    """

    def __init__(self):
        self._name: str = "Manual Rules Engine"
        self._description: str = (
            "Apply custom rules for data suppression, generalization (mapping, numeric ranges), or perturbation. "
            "Rules are applied sequentially as defined."
        )
        self.rule_types: List[str] = [
            "Suppress",
            "Generalize by Mapping",
            "Generalize by Range (Numeric)",
            "Perturb Numeric"
        ]
        self.perturbation_methods: List[str] = ["Gaussian", "Uniform"] # Keep this

    def get_name(self) -> str:
        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Suppression & Generalization"

    def get_description(self) -> str:
        return self._description

    def _get_rules_state(self, unique_key_prefix: str) -> List[Dict[str, Any]]:
        """Safely retrieves the list of rules from session state."""
        state_key = f"{unique_key_prefix}_manual_rules_list"
        if state_key not in st.session_state:
            st.session_state[state_key] = []
        return st.session_state[state_key]

    def _update_rules_state(self, unique_key_prefix: str, rules: List[Dict[str, Any]]):
        """Updates the list of rules in session state."""
        st.session_state[f"{unique_key_prefix}_manual_rules_list"] = rules

    def _add_rule(self, unique_key_prefix: str, all_cols: List[str]):
        rules = self._get_rules_state(unique_key_prefix)
        new_rule_id = str(uuid.uuid4())
        default_col = all_cols[0] if all_cols else None
        rules.append(
            {
                "id": new_rule_id,
                "column": default_col,
                "rule_type": self.rule_types[0],  # Default to "Suppress"
                "params": self._get_default_params(self.rule_types[0]),
            }
        )
        self._update_rules_state(unique_key_prefix, rules)

    def _remove_rule(self, unique_key_prefix: str, rule_id_to_remove: str):
        rules = self._get_rules_state(unique_key_prefix)
        rules = [rule for rule in rules if rule.get("id") != rule_id_to_remove]
        self._update_rules_state(unique_key_prefix, rules)

    def _get_default_params(self, rule_type: str) -> Dict[str, Any]:
        if rule_type == "Suppress":
            return {"placeholder": "*"}
        elif rule_type == "Generalize by Mapping":
            return {
                "mappings": pd.DataFrame([{"Original Value": "", "New Value": ""}]),
                "default_action": "Keep Original",
                "default_value_or_placeholder": ""
            }
        elif rule_type == "Generalize by Range (Numeric)":
            return {
                "ranges": pd.DataFrame([{"Min": None, "Max": None, "Label": ""}]),
                "default_action": "Keep Original",
                "default_value_or_placeholder": ""
            }
        elif rule_type == "Perturb Numeric":
            return {
                "perturbation_method": self.perturbation_methods[0], # Default to Gaussian
                "gaussian_std_dev": 1.0,
                "uniform_low": -0.5,
                "uniform_high": 0.5
            }
        return {}

    def _handle_rule_type_change_callback(self, unique_key_prefix: str, rule_id: str):
        rules = self._get_rules_state(unique_key_prefix)
        rule_to_update = next((r for r in rules if r.get("id") == rule_id), None)
        if rule_to_update:
            new_type_widget_key = f"{unique_key_prefix}_rule_{rule_id}_type"
            newly_selected_rule_type = st.session_state.get(new_type_widget_key)
            if newly_selected_rule_type and rule_to_update.get("rule_type") != newly_selected_rule_type:
                rule_to_update["rule_type"] = newly_selected_rule_type
                rule_to_update["params"] = self._get_default_params(newly_selected_rule_type)
                self._update_rules_state(unique_key_prefix, rules)
                st.rerun()

    # --- NEW CALLBACK for Perturbation Method Change ---
    def _handle_perturb_method_change_callback(self, unique_key_prefix: str, rule_id: str):
        """Callback to update perturbation method and rerun."""
        rules = self._get_rules_state(unique_key_prefix)
        rule_to_update = next((r for r in rules if r.get("id") == rule_id), None)
        if rule_to_update and rule_to_update.get("rule_type") == "Perturb Numeric":
            method_widget_key = f"{unique_key_prefix}_rule_{rule_id}_perturb_method"
            newly_selected_method = st.session_state.get(method_widget_key)
            # Ensure params dictionary exists
            if "params" not in rule_to_update:
                rule_to_update["params"] = self._get_default_params("Perturb Numeric")
            if newly_selected_method and rule_to_update["params"].get("perturbation_method") != newly_selected_method:
                rule_to_update["params"]["perturbation_method"] = newly_selected_method
                # We don't need to reset other perturb params (e.g., std_dev or low/high)
                # as the user might want to switch back and forth preserving those.
                # The UI will conditionally show the relevant ones.
                self._update_rules_state(unique_key_prefix, rules)
                st.rerun()

    def get_sidebar_ui(
        self,
        all_cols: List[str],
        sa_col_to_pass: Optional[str],
        df_raw: Optional[pd.DataFrame],
        unique_key_prefix: str,
    ) -> Dict[str, Any]:
        st.sidebar.subheader(f"{self.get_name()} Configuration")
        rules = self._get_rules_state(unique_key_prefix)
        if st.sidebar.button("Add New Rule", key=f"{unique_key_prefix}_add_rule_btn"):
            if not all_cols:
                st.sidebar.warning("Cannot add rule: No data columns available.")
            else:
                self._add_rule(unique_key_prefix, all_cols)
                st.rerun()
        st.sidebar.markdown("---")
        if not rules:
            st.sidebar.info("No rules defined. Click 'Add New Rule' to start.")
        rules_to_render = list(rules)
        for i, rule_config_iter in enumerate(rules_to_render): # Use a different name to avoid confusion
            rule_id = rule_config_iter.get("id", str(i))
            # Get the definitive rule_config from session state for modification
            # This ensures callbacks modify the actual session state object
            rule_config = next((r for r in rules if r.get("id") == rule_id), None)
            if not rule_config: # Should not happen if rules_to_render is from rules
                continue
            with st.sidebar.container():
                st.markdown(f"**Rule {i+1}** (ID: `{rule_id[:4]}...`)")
                current_column = rule_config.get("column")
                selected_col_for_rule = st.selectbox(
                    "Column to Apply Rule:", options=all_cols,
                    index=all_cols.index(current_column) if current_column and current_column in all_cols else 0,
                    key=f"{unique_key_prefix}_rule_{rule_id}_column"
                )
                rule_config["column"] = selected_col_for_rule # Directly update the item from 'rules' list
                current_rule_type = rule_config.get("rule_type")
                st.selectbox(
                    label="Rule Type:", options=self.rule_types,
                    index=self.rule_types.index(current_rule_type) if current_rule_type in self.rule_types else 0,
                    key=f"{unique_key_prefix}_rule_{rule_id}_type",
                    on_change=self._handle_rule_type_change_callback,
                    args=(unique_key_prefix, rule_id)
                )
                # After the selectbox, current_rule_type in rule_config might have been updated by the callback
                # So, we re-fetch it for the conditional UI rendering for *this* pass
                current_rule_type = rule_config.get("rule_type") # Re-access after potential callback
                params = rule_config.get("params", self._get_default_params(current_rule_type))
                if "params" not in rule_config: # Ensure params dict exists if rule was just changed
                    rule_config["params"] = params
                if current_rule_type == "Suppress":
                    params["placeholder"] = st.text_input(
                        "Suppression Placeholder:", value=params.get("placeholder", "*"),
                        key=f"{unique_key_prefix}_rule_{rule_id}_suppress_placeholder"
                    )
                elif current_rule_type == "Generalize by Mapping":
                    st.markdown("Define Mappings (Original Value -> New Value):")
                    default_mappings_df = pd.DataFrame([{"Original Value": "", "New Value": ""}])
                    mappings_data = params.get("mappings", default_mappings_df)
                    if not isinstance(mappings_data, pd.DataFrame) or list(mappings_data.columns) != ["Original Value", "New Value"]:
                        mappings_data = default_mappings_df
                    edited_mappings_df = st.data_editor(
                        mappings_data, num_rows="dynamic",
                        key=f"{unique_key_prefix}_rule_{rule_id}_map_editor", use_container_width=True,
                        column_config={
                            "Original Value": st.column_config.TextColumn(required=True),
                            "New Value": st.column_config.TextColumn(required=True)
                        }
                    )
                    params["mappings"] = edited_mappings_df
                    params["default_action"] = st.selectbox(
                        "For Unmapped Values:", options=["Keep Original", "Suppress", "Assign Default Label"],
                        index=["Keep Original", "Suppress", "Assign Default Label"].index(params.get("default_action", "Keep Original")),
                        key=f"{unique_key_prefix}_rule_{rule_id}_map_default_action"
                    )
                    if params["default_action"] in ["Suppress", "Assign Default Label"]:
                        params["default_value_or_placeholder"] = st.text_input(
                            f"{'Placeholder' if params['default_action'] == 'Suppress' else 'Default Label'}:",
                            value=params.get("default_value_or_placeholder", ""),
                            key=f"{unique_key_prefix}_rule_{rule_id}_map_default_value"
                        )
                elif current_rule_type == "Generalize by Range (Numeric)":
                    st.markdown("Define Numeric Ranges (Min, Max -> Label):")
                    default_ranges_df = pd.DataFrame([{"Min": None, "Max": None, "Label": ""}])
                    ranges_data = params.get("ranges", default_ranges_df)
                    if not isinstance(ranges_data, pd.DataFrame) or list(ranges_data.columns) != ["Min", "Max", "Label"]:
                        ranges_data = default_ranges_df
                    edited_ranges_df = st.data_editor(
                        ranges_data, num_rows="dynamic",
                        key=f"{unique_key_prefix}_rule_{rule_id}_range_editor", use_container_width=True,
                        column_config={
                            "Min": st.column_config.NumberColumn(format="%.2f", required=False),
                            "Max": st.column_config.NumberColumn(format="%.2f", required=False),
                            "Label": st.column_config.TextColumn(required=True)
                        }
                    )
                    params["ranges"] = edited_ranges_df
                    params["default_action"] = st.selectbox(
                        "For Values Outside Ranges:", options=["Keep Original", "Suppress", "Assign Default Label"],
                        index=["Keep Original", "Suppress", "Assign Default Label"].index(params.get("default_action", "Keep Original")),
                        key=f"{unique_key_prefix}_rule_{rule_id}_range_default_action"
                    )
                    if params["default_action"] in ["Suppress", "Assign Default Label"]:
                        params["default_value_or_placeholder"] = st.text_input(
                            f"{'Placeholder' if params['default_action'] == 'Suppress' else 'Default Label'}:",
                            value=params.get("default_value_or_placeholder", ""),
                            key=f"{unique_key_prefix}_rule_{rule_id}_range_default_value"
                        )
                elif current_rule_type == "Perturb Numeric":
                    # Ensure params for perturbation exist if rule type was just changed
                    if "perturbation_method" not in params: 
                        default_perturb_params = self._get_default_params("Perturb Numeric")
                        params.update(default_perturb_params)
                    st.selectbox( # Perturbation method selectbox
                        "Perturbation Method (Additive Noise):", options=self.perturbation_methods,
                        index=self.perturbation_methods.index(params.get("perturbation_method", self.perturbation_methods[0])),
                        key=f"{unique_key_prefix}_rule_{rule_id}_perturb_method",
                        on_change=self._handle_perturb_method_change_callback, # MODIFIED
                        args=(unique_key_prefix, rule_id) # MODIFIED
                    )
                    # Re-access perturbation_method from params after potential callback update
                    current_perturb_method = params.get("perturbation_method")
                    if current_perturb_method == "Gaussian":
                        params["gaussian_std_dev"] = st.number_input(
                            "Gaussian Noise Standard Deviation (σ):", value=float(params.get("gaussian_std_dev", 1.0)),
                            min_value=0.0, step=0.01, format="%.3f", # Increased precision for std dev
                            key=f"{unique_key_prefix}_rule_{rule_id}_perturb_gauss_std",
                            help="Standard deviation of the zero-mean Gaussian noise to be added."
                        )
                    elif current_perturb_method == "Uniform":
                        params["uniform_low"] = st.number_input(
                            "Uniform Noise Lower Bound:", value=float(params.get("uniform_low", -0.5)),
                            step=0.01, format="%.3f", # Increased precision
                            key=f"{unique_key_prefix}_rule_{rule_id}_perturb_uni_low",
                            help="Lower bound of the uniform noise to be added."
                        )
                        params["uniform_high"] = st.number_input(
                            "Uniform Noise Upper Bound:", value=float(params.get("uniform_high", 0.5)),
                            step=0.01, format="%.3f", # Increased precision
                            key=f"{unique_key_prefix}_rule_{rule_id}_perturb_uni_high",
                            help="Upper bound of the uniform noise to be added."
                        )
                        if params.get("uniform_low", 0) >= params.get("uniform_high", 0): # Check current values
                            st.warning("Uniform lower bound should be strictly less than the upper bound.")
                rule_config["params"] = params # Ensure params are set back to the main rule_config from session
                if st.button("Remove Rule", key=f"{unique_key_prefix}_remove_rule_{rule_id}_btn"):
                    self._remove_rule(unique_key_prefix, rule_id)
                    st.rerun()
                st.markdown("---")
        self._update_rules_state(unique_key_prefix, rules) # Ensure any direct modifications to rule_config items are saved
        return {"rules_config": self._get_rules_state(unique_key_prefix)}

    def _apply_suppression(self, series: pd.Series, placeholder: str) -> pd.Series:
        return pd.Series([placeholder] * len(series), index=series.index, dtype=object)

    def _apply_mapping(
        self,
        series: pd.Series,
        mappings_df: pd.DataFrame,
        default_action: str,
        default_value: Any,
    ) -> pd.Series:
        # Convert mappings to a dictionary for faster lookup
        mapping_dict = {
            str(row["Original Value"]): row["New Value"]
            for _, row in mappings_df.iterrows()
            if pd.notna(row["Original Value"])
        }
        new_series = series.copy().astype(object)  # Work with object type for flexibility
        for idx, val in series.items():
            str_val = str(val)
            if pd.isna(val):
                new_series[idx] = val
            elif str_val in mapping_dict:
                new_series[idx] = mapping_dict[str_val]
            else:
                if default_action == "Suppress": new_series[idx] = default_value
                elif default_action == "Assign Default Label": new_series[idx] = default_value
                else: new_series[idx] = val # Keep Original
        return new_series

    def _apply_range_generalization(
        self,
        series: pd.Series,
        ranges_df: pd.DataFrame,
        default_action: str,
        default_value: Any,
    ) -> pd.Series:
        # Ensure series is numeric
        try:
            numeric_series = pd.to_numeric(series, errors="coerce")
        except Exception:
            st.warning(
                f"Column '{series.name}' could not be converted to numeric for range generalization. Skipping rule for this column."
            )
            return series
        new_series = series.copy().astype(object)
        for idx, val in numeric_series.items():
            if pd.isna(val):
                new_series[idx] = val
                continue
            original_val_for_default = series[idx]
            applied_range_rule = False
            for _, rule_range in ranges_df.iterrows():
                min_val = pd.to_numeric(rule_range.get("Min"), errors="coerce")
                max_val = pd.to_numeric(rule_range.get("Max"), errors="coerce")
                label = rule_range.get("Label")
                min_ok = pd.isna(min_val) or val >= min_val
                max_ok = pd.isna(max_val) or val <= max_val
                if min_ok and max_ok:
                    new_series[idx] = label
                    applied_range_rule = True
                    break
            if not applied_range_rule:
                if default_action == "Suppress": new_series[idx] = default_value
                elif default_action == "Assign Default Label": new_series[idx] = default_value
                else: new_series[idx] = original_val_for_default
        return new_series

    def _apply_perturbation(self, series: pd.Series, method: str, gauss_std_dev: float, uni_low: float, uni_high: float) -> pd.Series:
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.isnull().all(): # All values are NaN or unconvertible
                st.info(f"Column '{series.name}' contains no valid numeric data for perturbation. Skipping.")
                return series
        except Exception as e:
            st.warning(f"Column '{series.name}' could not be converted to numeric for perturbation. Error: {e}. Skipping.")
            return series
        perturbed_series = numeric_series.copy() # Work with the numeric version
        non_na_indices = numeric_series.notna() # Get indices of valid numbers
        if not non_na_indices.any(): # No valid numbers to perturb
             st.info(f"Column '{series.name}' has no non-NaN numeric values to perturb. Skipping.")
             return series
        if method == "Gaussian":
            if gauss_std_dev < 0:
                st.error(f"Gaussian standard deviation ({gauss_std_dev}) cannot be negative for column '{series.name}'. Using 0.")
                gauss_std_dev = 0
            noise = np.random.normal(0, gauss_std_dev, size=non_na_indices.sum())
            perturbed_series.loc[non_na_indices] = numeric_series.loc[non_na_indices] + noise
        elif method == "Uniform":
            if uni_low >= uni_high:
                st.error(f"Uniform noise lower bound ({uni_low}) must be less than upper bound ({uni_high}) for column '{series.name}'. Skipping perturbation for this rule.")
                return series # Return original numeric series if params invalid
            noise = np.random.uniform(uni_low, uni_high, size=non_na_indices.sum())
            perturbed_series.loc[non_na_indices] = numeric_series.loc[non_na_indices] + noise
        else:
            st.warning(f"Unknown perturbation method: {method}. Skipping perturbation for column '{series.name}'.")
            return series # Return original numeric series
        # Create a new series with original NaNs and perturbed values
        final_series = series.copy().astype(object) # Start with original to preserve NaNs and non-numeric types
        final_series.loc[non_na_indices] = perturbed_series.loc[non_na_indices]
        # Attempt to cast back to original series's dtype if it was numeric, else keep as float
        if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_integer_dtype(series.dtype):
            try:
                final_series = final_series.astype(series.dtype)
            except ValueError: # If conversion fails (e.g. float to int after adding noise)
                 pass # Keep as float (perturbed_series dtype)
        elif pd.api.types.is_integer_dtype(series.dtype):
            # If original was integer, noise makes it float.
            # We could offer rounding, but for now, it becomes float.
            pass
        return final_series

    def anonymize(
        self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: Optional[str]
    ) -> pd.DataFrame:
        if df_input.empty:
            st.warning(f"{self.get_name()}: Input DataFrame is empty.")
            return df_input
        rules_config: List[Dict[str, Any]] = parameters.get("rules_config", [])
        if not rules_config:
            st.info(f"{self.get_name()}: No rules defined or applied.")
            return df_input
        df_anonymized = df_input.copy()
        for i, rule in enumerate(rules_config):
            col_name = rule.get("column")
            rule_type = rule.get("rule_type")
            params = rule.get("params", {})
            if not col_name or col_name not in df_anonymized.columns:
                st.warning(f"Rule {i+1}: Column '{col_name}' not found. Skipping rule.")
                continue
            # st.info(f"Applying Rule {i+1} ({rule_type}) to column '{col_name}'...") # Can be noisy
            target_series = df_anonymized[col_name]
            try:
                if rule_type == "Suppress":
                    df_anonymized[col_name] = self._apply_suppression(
                        target_series, params.get("placeholder", "*")
                    )
                elif rule_type == "Generalize by Mapping":
                    mappings_df = params.get("mappings")
                    if isinstance(mappings_df, pd.DataFrame) and not mappings_df.empty:
                        valid_mappings_df = mappings_df.dropna(subset=["Original Value"])
                        valid_mappings_df = valid_mappings_df[
                            valid_mappings_df["Original Value"].astype(str).str.strip() != ""
                        ]
                        if not valid_mappings_df.empty:
                            df_anonymized[col_name] = self._apply_mapping(
                                target_series, valid_mappings_df,
                                params.get("default_action", "Keep Original"),
                                params.get("default_value_or_placeholder", ""),
                            )
                        else:
                            st.warning(f"Rule {i + 1} (Mapping): No valid mappings after filtering. Skipping.")
                    else:
                        st.warning(f"Rule {i + 1} (Mapping): No mappings defined. Skipping.")
                elif rule_type == "Generalize by Range (Numeric)":
                    ranges_df = params.get("ranges")
                    if isinstance(ranges_df, pd.DataFrame) and not ranges_df.empty:
                        valid_ranges_df = ranges_df.dropna(subset=["Label"])
                        valid_ranges_df = valid_ranges_df[
                            valid_ranges_df["Label"].astype(str).str.strip() != ""
                        ]
                        # Further validation: Min <= Max if both are numbers
                        indices_to_drop = []
                        for idx_r, r in valid_ranges_df.iterrows():
                            min_r, max_r = pd.to_numeric(r.get("Min"), errors='coerce'), pd.to_numeric(r.get("Max"), errors='coerce')
                            if pd.notna(min_r) and pd.notna(max_r) and min_r > max_r:
                                st.warning(
                                    f"Rule {i + 1} (Range): Invalid range Min ({min_r}) > Max ({max_r}) for label '{r.get('Label')}'. This range will be ignored."
                                )
                                indices_to_drop.append(idx_r)
                        if indices_to_drop:
                            valid_ranges_df = valid_ranges_df.drop(indices_to_drop)
                        if not valid_ranges_df.empty:
                            df_anonymized[col_name] = self._apply_range_generalization(
                                target_series, valid_ranges_df,
                                params.get("default_action", "Keep Original"),
                                params.get("default_value_or_placeholder", ""),
                            )
                        else:
                            st.warning(f"Rule {i + 1} (Range): No valid ranges after filtering. Skipping.")
                    else:
                        st.warning(f"Rule {i + 1} (Range): No ranges defined. Skipping.")
                elif rule_type == "Perturb Numeric": # New Rule processing
                    df_anonymized[col_name] = self._apply_perturbation(
                        target_series,
                        params.get("perturbation_method", self.perturbation_methods[0]),
                        float(params.get("gaussian_std_dev", 1.0)),
                        float(params.get("uniform_low", -0.5)),
                        float(params.get("uniform_high", 0.5))
                    )
            except Exception as e:
                st.error(f"Error applying Rule {i+1} ({rule_type}) to column '{col_name}': {e}")
                traceback.print_exc() # For more detailed debugging in console
        st.success(f"{self.get_name()}: All defined rules processed.")
        return df_anonymized

    def build_config_export(self, unique_key_prefix: str, sa_col: Optional[str]) -> Dict[str, Any]:
        rules = self._get_rules_state(unique_key_prefix)
        exportable_rules = []
        for rule_config in rules:
            exp_rule = rule_config.copy()
            if "params" in exp_rule and isinstance(exp_rule["params"], dict):
                exp_rule_params = {}
                for k, v in exp_rule["params"].items():
                    if isinstance(v, pd.DataFrame):
                        exp_rule_params[k] = v.to_dict(orient="records")
                    else:
                        exp_rule_params[k] = v 
                exp_rule["params"] = exp_rule_params
            exportable_rules.append(exp_rule)
        return {"rules_config": exportable_rules}

    def apply_config_import(
        self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str
    ):
        imported_rules_raw = config_params.get("rules_config", [])
        deserialized_rules = []
        for rule_dict_raw in imported_rules_raw:
            des_rule = rule_dict_raw.copy()
            if "params" in des_rule and isinstance(des_rule["params"], dict):
                # Deep copy params for modification
                des_rule["params"] = des_rule["params"].copy()
                # Convert list of dicts back to DataFrames if they exist
                if "mappings" in des_rule["params"] and isinstance(des_rule["params"]["mappings"], list):
                    des_rule["params"]["mappings"] = pd.DataFrame(des_rule["params"]["mappings"])
                if "ranges" in des_rule["params"] and isinstance(des_rule["params"]["ranges"], list):
                    des_rule["params"]["ranges"] = pd.DataFrame(des_rule["params"]["ranges"])
            deserialized_rules.append(des_rule)
        self._update_rules_state(unique_key_prefix, deserialized_rules)

    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        try:
            json_string = json.dumps(config_to_export, indent=4)
        except TypeError as e:
            st.error(f"Error serializing config for export: {e}. Using string conversion as fallback.")
            json_string = json.dumps(config_to_export, indent=4, default=str) # Fallback
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config", data=json_string,
            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json",
            mime="application/json", key=f"{unique_key_prefix}_export_manual_rules"
        )

    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        return st.button(
            f"Anonymize with {self.get_name()}",
            key=f"{unique_key_prefix}_anonymize_manual_rules",
        )

def get_plugin():
    return ManualRulesAnonymizer()

