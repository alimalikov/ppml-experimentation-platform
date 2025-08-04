"""
Professional p-sensitive k-anonymity plugin for the anonymization tool.
Provides comprehensive p-sensitive k-anonymity implementation for probabilistic privacy protection.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import numpy as np
from ..base_anonymizer import Anonymizer
from ..p_sensitive_core import apply_p_sensitive_k_anonymity

class PSensitivePlugin(Anonymizer):
    """
    Professional p-sensitive k-anonymity plugin for probabilistic privacy protection.
    """

    def __init__(self):
        """Initialize the p-sensitive k-anonymity plugin."""
        self._name = "p-Sensitive k-Anonymity"
        self._description = ("Specialized privacy model that extends k-anonymity to protect specific "
                           "sensitive values by ensuring the probability of inferring these values "
                           "from any equivalence class does not exceed a threshold p.")

    def get_name(self) -> str:
        """Returns the display name of the anonymization technique."""
        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Privacy Models"

    def get_description(self) -> str:
        """Returns detailed description of the technique."""
        return self._description

    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the p-sensitive k-anonymity specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🛡️ {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About p-Sensitive k-Anonymity"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Probabilistic privacy protection for specific sensitive values
            - Automatic sensitive value detection
            - Flexible threshold-based protection
            - Protection against inference attacks
            
            **Best for:** Datasets with specific values requiring protection (e.g., rare diseases, 
            high salaries, sensitive demographics)
            """)

        # Define session state keys
        qi_key = f"{unique_key_prefix}_qi_cols"
        k_key = f"{unique_key_prefix}_k_value"
        p_key = f"{unique_key_prefix}_p_value"
        sa_key = f"{unique_key_prefix}_sensitive_col"
        sensitive_values_key = f"{unique_key_prefix}_sensitive_values"
        auto_detect_key = f"{unique_key_prefix}_auto_detect"
        strategy_key = f"{unique_key_prefix}_generalization_strategy"
        show_metrics_key = f"{unique_key_prefix}_show_metrics"

        # Sensitive Attribute Selection
        st.sidebar.subheader("🎯 Sensitive Attribute")
        
        if sa_col_to_pass and sa_col_to_pass in all_cols:
            default_sa = sa_col_to_pass
            st.sidebar.info(f"Using pre-selected sensitive attribute: **{sa_col_to_pass}**")
        else:
            default_sa = st.session_state.get(sa_key, all_cols[0] if all_cols else None)
        
        sa_col_options = all_cols if all_cols else ["No columns available"]
        sa_index = sa_col_options.index(default_sa) if default_sa in sa_col_options else 0
        
        selected_sa_col = st.sidebar.selectbox(
            "Sensitive Attribute Column:",
            options=sa_col_options,
            index=sa_index,
            key=sa_key,
            help="Column containing sensitive information with specific values to protect"
        )

        # Analyze sensitive attribute
        sa_analysis = {}
        if df_raw is not None and selected_sa_col in df_raw.columns:
            sa_values = df_raw[selected_sa_col].dropna()
            value_counts = sa_values.value_counts()
            value_frequencies = sa_values.value_counts(normalize=True)
            
            sa_analysis = {
                'total_records': len(sa_values),
                'unique_values': sa_values.nunique(),
                'value_counts': value_counts,
                'value_frequencies': value_frequencies,
                'rare_values': value_frequencies[value_frequencies < 0.1].index.tolist(),
                'common_values': value_frequencies[value_frequencies >= 0.1].index.tolist()
            }
            
            with st.sidebar.expander(f"📊 Sensitive Attribute Analysis: {selected_sa_col}"):
                st.metric("Total Records", sa_analysis['total_records'])
                st.metric("Unique Values", sa_analysis['unique_values'])
                st.metric("Rare Values (<10%)", len(sa_analysis['rare_values']))
                st.metric("Common Values (≥10%)", len(sa_analysis['common_values']))
                
                # Show value distribution
                st.markdown("**Value Distribution:**")
                for value, freq in value_frequencies.head(5).items():
                    rarity = "🔴 Rare" if freq < 0.1 else "🟡 Uncommon" if freq < 0.3 else "🟢 Common"
                    st.write(f"• {value}: {freq:.1%} {rarity}")

        # Quasi-Identifier Columns Selection
        st.sidebar.subheader("📊 Quasi-Identifier Selection")
        default_qi_cols = st.session_state.get(qi_key, [])
        valid_default_qi_cols = [col for col in default_qi_cols if col in all_cols and col != selected_sa_col]

        qi_col_options = [col for col in all_cols if col != selected_sa_col]
        
        selected_qi_cols = st.sidebar.multiselect(
            "Select Quasi-Identifier (QI) columns:",
            options=qi_col_options,
            default=valid_default_qi_cols,
            key=qi_key,
            help="Columns for grouping records (excluding sensitive attribute)"
        )

        # Privacy Parameters
        st.sidebar.subheader("🔒 Privacy Parameters")
        
        # K-value configuration
        min_k = 2
        if df_raw is not None and not df_raw.empty:
            max_k_val = min(50, len(df_raw) // 2)
            default_k = min(5, max_k_val)
        else:
            max_k_val = 20
            default_k = 5

        k_value = st.sidebar.slider(
            "k-value (minimum group size):",
            min_value=min_k,
            max_value=max_k_val,
            value=st.session_state.get(k_key, default_k),
            step=1,
            key=k_key,
            help="Minimum group size for k-anonymity"
        )

        # P-value configuration
        st.sidebar.subheader("🎲 p-Sensitivity Configuration")
        
        min_p = 0.01
        max_p = 1.0
        default_p = 0.5
        
        p_value = st.sidebar.slider(
            "p-value (maximum inference probability):",
            min_value=min_p,
            max_value=max_p,
            value=st.session_state.get(p_key, default_p),
            step=0.01,
            key=p_key,
            help="Maximum probability of inferring sensitive values (lower = stronger protection)"
        )

        # Privacy level assessment
        if p_value <= 0.2:
            privacy_level = "🔵 Maximum Privacy"
            privacy_desc = "Very strict inference protection"
        elif p_value <= 0.4:
            privacy_level = "🟢 High Privacy"
            privacy_desc = "Strong inference protection"
        elif p_value <= 0.6:
            privacy_level = "🟡 Medium Privacy"
            privacy_desc = "Balanced privacy and utility"
        else:
            privacy_level = "🔴 Low Privacy"
            privacy_desc = "Relaxed inference protection"

        st.sidebar.info(f"**{privacy_level}**\n{privacy_desc}")

        # Sensitive Values Selection
        st.sidebar.subheader("🔍 Sensitive Values Protection")
        
        auto_detect = st.sidebar.checkbox(
            "Auto-detect sensitive values",
            value=st.session_state.get(auto_detect_key, True),
            key=auto_detect_key,
            help="Automatically identify sensitive values based on frequency (values with <30% frequency)"
        )

        sensitive_values_to_protect = []
        
        if auto_detect:
            if sa_analysis.get('rare_values'):
                sensitive_values_to_protect = sa_analysis['rare_values']
                st.sidebar.success(f"Auto-detected {len(sensitive_values_to_protect)} sensitive values")
                
                with st.sidebar.expander("🔍 Auto-detected Sensitive Values"):
                    for value in sensitive_values_to_protect[:10]:  # Show first 10
                        freq = sa_analysis['value_frequencies'].get(value, 0)
                        st.write(f"• {value} ({freq:.1%})")
                    if len(sensitive_values_to_protect) > 10:
                        st.write(f"... and {len(sensitive_values_to_protect) - 10} more")
            else:
                st.sidebar.warning("No rare values auto-detected. All values have ≥30% frequency.")
                # Fall back to least frequent value
                if sa_analysis.get('value_frequencies') is not None and len(sa_analysis['value_frequencies']) > 0:
                    least_frequent = sa_analysis['value_frequencies'].idxmin()
                    sensitive_values_to_protect = [least_frequent]
                    st.sidebar.info(f"Using least frequent value: {least_frequent}")
        else:
            # Manual selection
            if sa_analysis.get('unique_values'):
                all_values = sa_analysis['value_counts'].index.tolist()
                
                default_manual = st.session_state.get(sensitive_values_key, [])
                valid_default_manual = [v for v in default_manual if v in all_values]
                
                selected_sensitive_values = st.sidebar.multiselect(
                    "Select sensitive values to protect:",
                    options=all_values,
                    default=valid_default_manual,
                    key=sensitive_values_key,
                    help="Choose specific values that need probabilistic protection"
                )
                sensitive_values_to_protect = selected_sensitive_values
            else:
                st.sidebar.warning("No data available for manual selection")

        # Show protection analysis
        if sensitive_values_to_protect and sa_analysis.get('value_frequencies') is not None:
            st.sidebar.markdown("**Protection Analysis:**")
            total_protected_records = 0
            for value in sensitive_values_to_protect[:3]:  # Show first 3
                count = sa_analysis['value_counts'].get(value, 0)
                freq = sa_analysis['value_frequencies'].get(value, 0)
                total_protected_records += count
                risk_level = "🔴 High Risk" if freq < 0.05 else "🟡 Medium Risk" if freq < 0.2 else "🟢 Low Risk"
                st.sidebar.write(f"• {value}: {count} records ({freq:.1%}) {risk_level}")
            
            if len(sensitive_values_to_protect) > 3:
                remaining_count = sum(sa_analysis['value_counts'].get(v, 0) for v in sensitive_values_to_protect[3:])
                total_protected_records += remaining_count
                st.sidebar.write(f"• +{len(sensitive_values_to_protect) - 3} more values: {remaining_count} records")
            
            protection_ratio = total_protected_records / sa_analysis['total_records'] if sa_analysis['total_records'] > 0 else 0
            st.sidebar.metric("Protected Records", f"{total_protected_records} ({protection_ratio:.1%})")

        # Generalization Strategy
        st.sidebar.subheader("⚙️ Algorithm Configuration")
        
        strategy_options = {
            "optimal": "Optimal - Best privacy/utility balance",
            "greedy": "Greedy - Fast for large datasets",
            "binary": "Binary Search - Balanced approach"
        }
        
        current_strategy = st.session_state.get(strategy_key, "optimal")
        selected_strategy_display = st.sidebar.selectbox(
            "Generalization Strategy:",
            options=list(strategy_options.values()),
            index=list(strategy_options.keys()).index(current_strategy),
            key=f"{strategy_key}_display"
        )
        
        selected_strategy = [k for k, v in strategy_options.items() if v == selected_strategy_display][0]
        st.session_state[strategy_key] = selected_strategy

        # Advanced Options
        with st.sidebar.expander("🔧 Advanced Options"):
            show_detailed_metrics = st.checkbox(
                "Show detailed privacy metrics",
                value=st.session_state.get(show_metrics_key, True),
                key=show_metrics_key
            )
            
            # Feasibility analysis
            if sa_analysis.get('total_records') and sensitive_values_to_protect:
                max_group_violations = 0
                for value in sensitive_values_to_protect:
                    value_count = sa_analysis['value_counts'].get(value, 0)
                    if value_count / k_value > p_value:
                        max_group_violations += 1
                
                if max_group_violations == 0:
                    st.success("✅ Configuration appears feasible")
                elif max_group_violations <= len(sensitive_values_to_protect) * 0.3:
                    st.warning("⚠️ Some sensitive values may be difficult to protect")
                else:
                    st.error("❌ Configuration may be too strict - consider higher p-value or lower k-value")

        # Validation and Warnings
        validation_passed = True
        
        if not selected_qi_cols:
            st.sidebar.error("⚠️ Please select at least one Quasi-Identifier column")
            validation_passed = False
        
        if not selected_sa_col or selected_sa_col == "No columns available":
            st.sidebar.error("⚠️ Please select a valid Sensitive Attribute column")
            validation_passed = False
        
        if not sensitive_values_to_protect:
            st.sidebar.error("⚠️ No sensitive values selected for protection")
            validation_passed = False
        
        if p_value >= 1.0:
            st.sidebar.warning("⚠️ p-value of 1.0 provides no protection")

        if validation_passed:
            st.sidebar.success("✅ Configuration is valid")

        return {
            "qi_cols": selected_qi_cols,
            "k": k_value,
            "p": p_value,
            "sensitive_column": selected_sa_col,
            "sensitive_values": sensitive_values_to_protect if not auto_detect else None,  # None means auto-detect
            "auto_detect_sensitive": auto_detect,
            "generalization_strategy": selected_strategy,
            "show_detailed_metrics": show_detailed_metrics
        }

    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Performs p-sensitive k-anonymity anonymization.
        """
        qi_cols = parameters.get("qi_cols", [])
        k_val = parameters.get("k", 2)
        p_val = parameters.get("p", 0.5)
        sensitive_col = parameters.get("sensitive_column")
        sensitive_values = parameters.get("sensitive_values")  # None means auto-detect
        auto_detect = parameters.get("auto_detect_sensitive", True)
        strategy = parameters.get("generalization_strategy", "optimal")
        show_metrics = parameters.get("show_detailed_metrics", True)

        # Validation
        if not qi_cols:
            st.error("❌ p-Sensitive Error: Please select at least one Quasi-Identifier column.")
            return pd.DataFrame()

        if not sensitive_col or sensitive_col not in df_input.columns:
            st.error("❌ p-Sensitive Error: Please select a valid Sensitive Attribute column.")
            return pd.DataFrame()

        if k_val < 2:
            st.error("❌ p-Sensitive Error: k-value must be at least 2.")
            return pd.DataFrame()
        
        if not (0 < p_val <= 1):
            st.error("❌ p-Sensitive Error: p-value must be between 0 and 1.")
            return pd.DataFrame()

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing p-sensitive k-anonymity...")
            progress_bar.progress(15)
            
            if auto_detect or sensitive_values is None:
                status_text.text("🔄 Auto-detecting sensitive values...")
                progress_bar.progress(25)
            
            status_text.text("🔄 Analyzing sensitive value distributions...")
            progress_bar.progress(35)
            
            status_text.text("🔄 Applying k-anonymity preprocessing...")
            progress_bar.progress(50)
            
            status_text.text("🔄 Enforcing p-sensitivity constraints...")
            progress_bar.progress(70)
            
            # Apply p-sensitive k-anonymity
            anonymized_df, metrics = apply_p_sensitive_k_anonymity(
                df=df_input.copy(),
                k=k_val,
                p=p_val,
                qi_columns=qi_cols,
                sensitive_column=sensitive_col,
                sensitive_values=sensitive_values,  # None triggers auto-detection
                generalization_strategy=strategy
            )
            
            progress_bar.progress(85)
            status_text.text("🔄 Calculating privacy metrics...")
            
            if anonymized_df.empty:
                st.error("❌ p-Sensitive k-anonymity failed. Try higher p-value or lower k-value.")
                return pd.DataFrame()
            
            progress_bar.progress(100)
            status_text.text("✅ p-Sensitive k-anonymity completed successfully!")
            
            # Display results
            st.success(f"✅ **p-Sensitive k-Anonymity Applied Successfully**")
            
            protected_values = metrics.get('protected_sensitive_values', [])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("k-value", k_val)
            with col2:
                st.metric("p-value", f"{p_val:.3f}")
            with col3:
                st.metric("Protected Values", len(protected_values))
            with col4:
                compliance = metrics.get('p_sensitivity_compliance', 0)
                st.metric("p-Sensitive Compliance", f"{compliance:.1%}")
            
            # Show protected values
            if protected_values:
                with st.expander("🛡️ Protected Sensitive Values", expanded=False):
                    st.markdown("**Values under probabilistic protection:**")
                    for i, value in enumerate(protected_values):
                        if i < 10:  # Show first 10
                            original_count = (df_input[sensitive_col] == value).sum()
                            anonymized_count = (anonymized_df[sensitive_col] == value).sum()
                            original_freq = original_count / len(df_input) if len(df_input) > 0 else 0
                            anonymized_freq = anonymized_count / len(anonymized_df) if len(anonymized_df) > 0 else 0
                            st.write(f"• **{value}**: {original_count}→{anonymized_count} records ({original_freq:.1%}→{anonymized_freq:.1%})")
                        elif i == 10:
                            st.write(f"... and {len(protected_values) - 10} more values")
                            break
            
            # Show detailed metrics
            if show_metrics and metrics:
                with st.expander("📊 Detailed Privacy & Utility Metrics", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔒 Privacy Protection**")
                        st.metric("p-Sensitive Groups", f"{metrics.get('p_sensitive_groups', 'N/A')}")
                        st.metric("Total Groups", f"{metrics.get('total_groups', 'N/A')}")
                        st.metric("Min Group Size", metrics.get('min_group_size', 'N/A'))
                        st.metric("Max Sensitive Probability", f"{metrics.get('max_sensitive_probability', 0):.3f}")
                        
                    with col2:
                        st.markdown("**📈 Utility Metrics**")
                        st.metric("Information Loss", f"{metrics.get('information_loss', 0):.1f}%")
                        st.metric("Suppression Ratio", f"{metrics.get('suppression_ratio', 0):.1f}%")
                        st.metric("Avg Group Size", f"{metrics.get('avg_group_size', 0):.1f}")
                        st.metric("Min Sensitive Probability", f"{metrics.get('min_sensitive_probability', 0):.3f}")
                    
                    # Probability analysis
                    st.markdown("**🎲 Probability Analysis**")
                    prob_analysis = {
                        'Metric': [
                            'p-value Threshold', 
                            'Maximum Probability', 
                            'Average Probability',
                            'Probability Violations'
                        ],
                        'Value': [
                            f"{p_val:.3f}",
                            f"{metrics.get('max_sensitive_probability', 0):.3f}",
                            f"{metrics.get('avg_sensitive_probability', 0):.3f}",
                            f"{metrics.get('total_groups', 0) - metrics.get('p_sensitive_groups', 0)}"
                        ],
                        'Status': [
                            "Threshold",
                            "✅ Within Limit" if metrics.get('max_sensitive_probability', 0) <= p_val else "❌ Violation",
                            "Information",
                            "✅ No Violations" if metrics.get('p_sensitivity_compliance', 0) >= 1.0 else "⚠️ Has Violations"
                        ]
                    }
                    st.dataframe(pd.DataFrame(prob_analysis), use_container_width=True)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return anonymized_df

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ p-Sensitive k-Anonymity Error: {str(e)}")
            st.exception(e)
            return pd.DataFrame()

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """
        Builds the p-sensitive k-anonymity specific configuration for export.
        """
        return {
            "qi_cols": st.session_state.get(f"{unique_key_prefix}_qi_cols", []),
            "k": st.session_state.get(f"{unique_key_prefix}_k_value", 2),
            "p": st.session_state.get(f"{unique_key_prefix}_p_value", 0.5),
            "sensitive_column": st.session_state.get(f"{unique_key_prefix}_sensitive_col"),
            "sensitive_values": st.session_state.get(f"{unique_key_prefix}_sensitive_values"),
            "auto_detect_sensitive": st.session_state.get(f"{unique_key_prefix}_auto_detect", True),
            "generalization_strategy": st.session_state.get(f"{unique_key_prefix}_generalization_strategy", "optimal"),
            "show_detailed_metrics": st.session_state.get(f"{unique_key_prefix}_show_metrics", True)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """
        Applies imported p-sensitive k-anonymity configuration parameters to the session state.
        """
        # Set QI columns
        imported_qi_cols = config_params.get("qi_cols", [])
        st.session_state[f"{unique_key_prefix}_qi_cols"] = [col for col in imported_qi_cols if col in all_cols]
        
        # Set parameters
        st.session_state[f"{unique_key_prefix}_k_value"] = config_params.get("k", 2)
        st.session_state[f"{unique_key_prefix}_p_value"] = config_params.get("p", 0.5)
        
        # Set sensitive column
        sensitive_col = config_params.get("sensitive_column")
        if sensitive_col in all_cols:
            st.session_state[f"{unique_key_prefix}_sensitive_col"] = sensitive_col
        
        # Set sensitive values and auto-detect flag
        st.session_state[f"{unique_key_prefix}_sensitive_values"] = config_params.get("sensitive_values", [])
        st.session_state[f"{unique_key_prefix}_auto_detect"] = config_params.get("auto_detect_sensitive", True)
        
        # Set strategy
        strategy = config_params.get("generalization_strategy", "optimal")
        if strategy in ["optimal", "greedy", "binary"]:
            st.session_state[f"{unique_key_prefix}_generalization_strategy"] = strategy
        
        # Set metrics preference
        st.session_state[f"{unique_key_prefix}_show_metrics"] = config_params.get("show_detailed_metrics", True)


# Factory function for the plugin loader
def get_plugin():
    """Returns an instance of the PSensitivePlugin."""
    return PSensitivePlugin()
