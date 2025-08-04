"""
Professional k-anonymity plugin for the anonymization tool.
Provides comprehensive k-anonymity implementation with multiple generalization strategies.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import json
from ..base_anonymizer import Anonymizer
from ..k_anonymity_simple import apply_k_anonymity

class KAnonymityPlugin(Anonymizer):
    """
    Professional k-anonymity plugin with multiple generalization strategies.
    """

    def __init__(self):
        """Initialize the k-anonymity plugin."""
        self._name = "k-Anonymity"
        self._description = ("Professional k-anonymity implementation ensuring each record is "
                           "indistinguishable from at least k-1 other records based on "
                           "quasi-identifier attributes. Supports multiple generalization strategies.")

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
        Renders the k-anonymity specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔒 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About k-Anonymity"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Multiple generalization strategies (Optimal, Greedy, Binary Search)
            - Automatic hierarchy generation for numeric and categorical data
            - Comprehensive privacy and utility metrics
            - Flexible quasi-identifier selection
            
            **Best for:** Basic privacy protection, compliance requirements, datasets with clear QI columns
            """)

        # Define session state keys
        qi_key = f"{unique_key_prefix}_qi_cols"
        k_key = f"{unique_key_prefix}_k_value"
        strategy_key = f"{unique_key_prefix}_generalization_strategy"
        show_metrics_key = f"{unique_key_prefix}_show_metrics"

        # Quasi-Identifier Columns Selection
        st.sidebar.subheader("📊 Quasi-Identifier Selection")
        default_qi_cols = st.session_state.get(qi_key, [])
        valid_default_qi_cols = [col for col in default_qi_cols if col in all_cols]

        # Provide helpful suggestions for QI columns
        suggested_qi_patterns = ['age', 'zip', 'postal', 'date', 'birth', 'gender', 'race', 
                               'education', 'occupation', 'income', 'salary', 'location', 'address']
        suggested_cols = []
        for col in all_cols:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in suggested_qi_patterns):
                suggested_cols.append(col)

        if suggested_cols:
            st.sidebar.info(f"💡 Suggested QI columns: {', '.join(suggested_cols[:3])}{'...' if len(suggested_cols) > 3 else ''}")

        selected_qi_cols = st.sidebar.multiselect(
            "Select Quasi-Identifier (QI) columns:",
            options=all_cols,
            default=valid_default_qi_cols,
            key=qi_key,
            help="Choose columns that could be used to identify individuals when combined. "
                 "These will be generalized to achieve k-anonymity."
        )

        # K-value Configuration
        st.sidebar.subheader("🎯 Privacy Parameters")
        
        # Determine sensible k-value range
        min_k = 2
        if df_raw is not None and not df_raw.empty:
            max_k_val = min(100, len(df_raw) // 2)
            default_k = min(5, max_k_val)
        else:
            max_k_val = 20
            default_k = 5

        # Get current k value, ensuring it's within bounds
        current_k = st.session_state.get(k_key, default_k)
        clamped_k = max(min_k, min(current_k, max_k_val))

        k_value = st.sidebar.slider(
            "k-value (minimum group size):",
            min_value=min_k,
            max_value=max_k_val,
            value=clamped_k,
            step=1,
            key=k_key,
            help=f"Each record must be indistinguishable from at least k-1 others. "
                 f"Higher values = better privacy but more generalization. Range: {min_k}-{max_k_val}"
        )

        # Display privacy level indicator
        if k_value <= 3:
            privacy_level = "🔴 Low Privacy"
            privacy_desc = "Minimal protection, suitable for low-risk data"
        elif k_value <= 7:
            privacy_level = "🟡 Medium Privacy"
            privacy_desc = "Balanced protection for most use cases"
        elif k_value <= 15:
            privacy_level = "🟢 High Privacy"
            privacy_desc = "Strong protection for sensitive data"
        else:
            privacy_level = "🔵 Maximum Privacy"
            privacy_desc = "Highest protection, may significantly reduce utility"

        st.sidebar.info(f"**{privacy_level}**\n{privacy_desc}")

        # Generalization Strategy
        st.sidebar.subheader("⚙️ Algorithm Configuration")
        
        strategy_options = {
            "optimal": "Optimal - Dynamic programming approach (best privacy/utility balance)",
            "greedy": "Greedy - Fast algorithm, good for large datasets",
            "binary": "Binary Search - Balanced approach with controlled generalization"
        }
        
        current_strategy = st.session_state.get(strategy_key, "optimal")
        selected_strategy_display = st.sidebar.selectbox(
            "Generalization Strategy:",
            options=list(strategy_options.values()),
            index=list(strategy_options.keys()).index(current_strategy),
            key=f"{strategy_key}_display",
            help="Choose the algorithm for determining how to generalize data"
        )
        
        # Map back to strategy key
        selected_strategy = [k for k, v in strategy_options.items() if v == selected_strategy_display][0]
        st.session_state[strategy_key] = selected_strategy

        # Advanced Options
        with st.sidebar.expander("🔧 Advanced Options"):
            show_detailed_metrics = st.checkbox(
                "Show detailed privacy metrics",
                value=st.session_state.get(show_metrics_key, True),
                key=show_metrics_key,
                help="Display comprehensive privacy and utility analysis after anonymization"
            )
            
            if df_raw is not None and selected_qi_cols:
                st.markdown("**Data Analysis:**")
                total_records = len(df_raw)
                qi_combinations = len(df_raw[selected_qi_cols].drop_duplicates()) if selected_qi_cols else total_records
                
                st.metric("Total Records", total_records)
                st.metric("Unique QI Combinations", qi_combinations)
                
                if qi_combinations > 0:
                    estimated_groups = max(1, qi_combinations // k_value)
                    st.metric("Estimated Groups After Anonymization", estimated_groups)
                    
                    avg_group_size = total_records / estimated_groups if estimated_groups > 0 else 0
                    st.metric("Estimated Avg Group Size", f"{avg_group_size:.1f}")

        # Validation and Warnings
        if not selected_qi_cols:
            st.sidebar.error("⚠️ Please select at least one Quasi-Identifier column")
        elif df_raw is not None:
            # Check for potential issues
            if len(selected_qi_cols) > 5:
                st.sidebar.warning("⚠️ Many QI columns selected - this may result in over-generalization")
            
            # Check data types
            categorical_cols = []
            numeric_cols = []
            for col in selected_qi_cols:
                if col in df_raw.columns:
                    if pd.api.types.is_numeric_dtype(df_raw[col]):
                        numeric_cols.append(col)
                    else:
                        categorical_cols.append(col)
            
            if categorical_cols or numeric_cols:
                st.sidebar.success(f"✅ QI Analysis: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

        return {
            "qi_cols": selected_qi_cols,
            "k": k_value,
            "generalization_strategy": selected_strategy,
            "show_detailed_metrics": show_detailed_metrics
        }

    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Performs k-anonymity anonymization.
        """
        qi_cols = parameters.get("qi_cols", [])
        k_val = parameters.get("k", 2)
        strategy = parameters.get("generalization_strategy", "optimal")
        show_metrics = parameters.get("show_detailed_metrics", True)

        # Validation
        if not qi_cols:
            st.error("❌ k-Anonymity Error: Please select at least one Quasi-Identifier (QI) column.")
            return pd.DataFrame()

        if k_val < 2:
            st.error("❌ k-Anonymity Error: k-value must be at least 2.")
            return pd.DataFrame()

        if k_val > len(df_input):
            st.error(f"❌ k-Anonymity Error: k-value ({k_val}) cannot be greater than dataset size ({len(df_input)}).")
            return pd.DataFrame()

        # Check if QI columns exist
        missing_cols = [col for col in qi_cols if col not in df_input.columns]
        if missing_cols:
            st.error(f"❌ k-Anonymity Error: QI columns not found: {missing_cols}")
            return pd.DataFrame()

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing k-anonymity algorithm...")
            progress_bar.progress(20)
            
            status_text.text("🔄 Analyzing data structure and building generalization hierarchies...")
            progress_bar.progress(40)
            
            status_text.text(f"🔄 Applying {strategy} generalization strategy...")
            progress_bar.progress(60)
            
            # Apply k-anonymity
            anonymized_df, metrics = apply_k_anonymity(
                df=df_input.copy(),
                k=k_val,
                qi_columns=qi_cols,
                generalization_strategy=strategy
            )
            
            progress_bar.progress(80)
            status_text.text("🔄 Calculating privacy metrics...")
            
            # Display results
            if anonymized_df.empty:
                st.error("❌ k-Anonymity failed to produce anonymized data. Try reducing k-value or selecting fewer QI columns.")
                return pd.DataFrame()
            
            progress_bar.progress(100)
            status_text.text("✅ k-Anonymity completed successfully!")
            
            # Display summary metrics
            st.success(f"✅ **k-Anonymity Applied Successfully**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("k-value", k_val)
            with col2:
                st.metric("QI Columns", len(qi_cols))
            with col3:
                st.metric("Equivalence Classes", metrics.get('anonymized_equivalence_classes', 'N/A'))
            with col4:
                st.metric("Min Group Size", metrics.get('min_group_size', 'N/A'))
            
            # Show detailed metrics if requested
            if show_metrics and metrics:
                with st.expander("📊 Detailed Privacy & Utility Metrics", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔒 Privacy Metrics**")
                        st.metric("K-Anonymity Satisfied", "✅ Yes" if metrics.get('min_group_size', 0) >= k_val else "❌ No")
                        st.metric("Average Group Size", f"{metrics.get('avg_group_size', 0):.1f}")
                        st.metric("Largest Group Size", metrics.get('max_group_size', 'N/A'))
                        
                    with col2:
                        st.markdown("**📈 Utility Metrics**")
                        st.metric("Information Loss", f"{metrics.get('information_loss', 0):.1f}%")
                        st.metric("Suppression Ratio", f"{metrics.get('suppression_ratio', 0):.1f}%")
                        st.metric("Generalization Strategy", strategy.title())
                    
                    # Show QI column analysis
                    st.markdown("**🎯 Quasi-Identifier Analysis**")
                    qi_analysis_df = pd.DataFrame({
                        'QI Column': qi_cols,
                        'Data Type': [str(df_input[col].dtype) for col in qi_cols],
                        'Unique Values (Original)': [df_input[col].nunique() for col in qi_cols],
                        'Unique Values (Anonymized)': [anonymized_df[col].nunique() if col in anonymized_df.columns else 0 for col in qi_cols]
                    })
                    st.dataframe(qi_analysis_df, use_container_width=True)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return anonymized_df

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ k-Anonymity Error: {str(e)}")
            st.exception(e)
            return pd.DataFrame()

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """
        Builds the k-anonymity specific configuration for export.
        """
        return {
            "qi_cols": st.session_state.get(f"{unique_key_prefix}_qi_cols", []),
            "k": st.session_state.get(f"{unique_key_prefix}_k_value", 2),
            "generalization_strategy": st.session_state.get(f"{unique_key_prefix}_generalization_strategy", "optimal"),
            "show_detailed_metrics": st.session_state.get(f"{unique_key_prefix}_show_metrics", True)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """
        Applies imported k-anonymity configuration parameters to the session state.
        """
        # Set QI columns (filter to ensure they exist in current dataset)
        imported_qi_cols = config_params.get("qi_cols", [])
        st.session_state[f"{unique_key_prefix}_qi_cols"] = [col for col in imported_qi_cols if col in all_cols]
        
        # Set k-value
        st.session_state[f"{unique_key_prefix}_k_value"] = config_params.get("k", 2)
        
        # Set generalization strategy
        strategy = config_params.get("generalization_strategy", "optimal")
        if strategy in ["optimal", "greedy", "binary"]:
            st.session_state[f"{unique_key_prefix}_generalization_strategy"] = strategy
        
        # Set metrics display preference
        st.session_state[f"{unique_key_prefix}_show_metrics"] = config_params.get("show_detailed_metrics", True)


# Factory function for the plugin loader
def get_plugin():
    """Returns an instance of the KAnonymityPlugin."""
    return KAnonymityPlugin()
