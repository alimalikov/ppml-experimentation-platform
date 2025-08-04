"""
Professional l-diversity plugin for the anonymization tool.
Provides comprehensive l-diversity implementation building on k-anonymity.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import json
from ..base_anonymizer import Anonymizer
from ..l_diversity_core import apply_l_diversity

class LDiversityPlugin(Anonymizer):
    """
    Professional l-diversity plugin with multiple diversity measures.
    """

    def __init__(self):
        """Initialize the l-diversity plugin."""
        self._name = "l-Diversity"
        self._description = ("Advanced privacy model that builds on k-anonymity by ensuring "
                           "each equivalence class has at least l well-represented values "
                           "for sensitive attributes, protecting against homogeneity attacks.")

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
        Renders the l-diversity specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🌈 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About l-Diversity"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Builds on k-anonymity for stronger protection
            - Multiple diversity measures (Distinct, Entropy, Recursive)
            - Protection against homogeneity and background knowledge attacks
            - Advanced privacy metrics and analysis
            
            **Best for:** Sensitive datasets, medical records, datasets with clear sensitive attributes
            """)

        # Define session state keys
        qi_key = f"{unique_key_prefix}_qi_cols"
        k_key = f"{unique_key_prefix}_k_value"
        l_key = f"{unique_key_prefix}_l_value"
        sa_key = f"{unique_key_prefix}_sensitive_col"
        diversity_key = f"{unique_key_prefix}_diversity_type"
        strategy_key = f"{unique_key_prefix}_generalization_strategy"
        show_metrics_key = f"{unique_key_prefix}_show_metrics"

        # Sensitive Attribute Selection
        st.sidebar.subheader("🎯 Sensitive Attribute")
        
        # Use passed SA column or let user select
        if sa_col_to_pass and sa_col_to_pass in all_cols:
            default_sa = sa_col_to_pass
            st.sidebar.info(f"Using pre-selected sensitive attribute: **{sa_col_to_pass}**")
        else:
            default_sa = st.session_state.get(sa_key, all_cols[0] if all_cols else None)
        
        # Suggest potential sensitive attributes
        sensitive_patterns = ['diagnosis', 'disease', 'condition', 'salary', 'income', 'race', 
                            'religion', 'political', 'medical', 'health', 'treatment', 'outcome']
        suggested_sa_cols = []
        for col in all_cols:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in sensitive_patterns):
                suggested_sa_cols.append(col)

        if suggested_sa_cols:
            st.sidebar.info(f"💡 Suggested sensitive columns: {', '.join(suggested_sa_cols[:2])}")

        sa_col_options = all_cols if all_cols else ["No columns available"]
        sa_index = sa_col_options.index(default_sa) if default_sa in sa_col_options else 0
        
        selected_sa_col = st.sidebar.selectbox(
            "Sensitive Attribute Column:",
            options=sa_col_options,
            index=sa_index,
            key=sa_key,
            help="Column containing sensitive information that needs diversity protection"
        )

        # Analyze sensitive attribute if data is available
        sa_analysis = {}
        if df_raw is not None and selected_sa_col in df_raw.columns:
            sa_values = df_raw[selected_sa_col].dropna()
            sa_analysis = {
                'total_records': len(sa_values),
                'unique_values': sa_values.nunique(),
                'value_counts': sa_values.value_counts(),
                'most_common': sa_values.mode().iloc[0] if len(sa_values) > 0 else None
            }
            
            with st.sidebar.expander(f"📊 Sensitive Attribute Analysis: {selected_sa_col}"):
                st.metric("Total Records", sa_analysis['total_records'])
                st.metric("Unique Values", sa_analysis['unique_values'])
                
                if len(sa_analysis['value_counts']) > 0:
                    st.markdown("**Top Values:**")
                    top_values = sa_analysis['value_counts'].head(3)
                    for value, count in top_values.items():
                        percentage = (count / sa_analysis['total_records']) * 100
                        st.write(f"• {value}: {count} ({percentage:.1f}%)")

        # Quasi-Identifier Columns Selection
        st.sidebar.subheader("📊 Quasi-Identifier Selection")
        default_qi_cols = st.session_state.get(qi_key, [])
        valid_default_qi_cols = [col for col in default_qi_cols if col in all_cols and col != selected_sa_col]

        # Filter out sensitive column from QI options
        qi_col_options = [col for col in all_cols if col != selected_sa_col]
        
        selected_qi_cols = st.sidebar.multiselect(
            "Select Quasi-Identifier (QI) columns:",
            options=qi_col_options,
            default=valid_default_qi_cols,
            key=qi_key,
            help="Columns that could identify individuals when combined (excluding sensitive attribute)"
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

        current_k = st.session_state.get(k_key, default_k)
        clamped_k = max(min_k, min(current_k, max_k_val))

        k_value = st.sidebar.slider(
            "k-value (k-anonymity requirement):",
            min_value=min_k,
            max_value=max_k_val,
            value=clamped_k,
            step=1,
            key=k_key,
            help="Minimum group size for k-anonymity (l-diversity builds on this)"
        )

        # L-value configuration
        min_l = 2
        if sa_analysis.get('unique_values'):
            max_l_val = min(sa_analysis['unique_values'], 10)
            default_l = min(3, max_l_val)
        else:
            max_l_val = 5
            default_l = 2

        current_l = st.session_state.get(l_key, default_l)
        clamped_l = max(min_l, min(current_l, max_l_val))

        l_value = st.sidebar.slider(
            "l-value (diversity requirement):",
            min_value=min_l,
            max_value=max_l_val,
            value=clamped_l,
            step=1,
            key=l_key,
            help="Minimum number of well-represented sensitive values in each group"
        )

        # Diversity Type Selection
        st.sidebar.subheader("🔧 Diversity Configuration")
        
        diversity_options = {
            "distinct": "Distinct l-Diversity - Simple count of different values",
            "entropy": "Entropy l-Diversity - Information-theoretic measure",
            "recursive": "Recursive (c,l)-Diversity - Protection against skewed distributions"
        }
        
        current_diversity = st.session_state.get(diversity_key, "distinct")
        selected_diversity_display = st.sidebar.selectbox(
            "Diversity Measure:",
            options=list(diversity_options.values()),
            index=list(diversity_options.keys()).index(current_diversity),
            key=f"{diversity_key}_display",
            help="Method for measuring and ensuring diversity in sensitive values"
        )
        
        # Map back to diversity key
        selected_diversity = [k for k, v in diversity_options.items() if v == selected_diversity_display][0]
        st.session_state[diversity_key] = selected_diversity

        # Show diversity explanation
        diversity_explanations = {
            "distinct": "Each group must contain at least l different sensitive values.",
            "entropy": "Each group's entropy must be at least log₂(l).",
            "recursive": "Most frequent value must be less than c times the sum of other l-1 values."
        }
        st.sidebar.info(f"**{selected_diversity.title()}:** {diversity_explanations[selected_diversity]}")

        # Generalization Strategy
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
            
            # Privacy level assessment
            if sa_analysis.get('unique_values'):
                diversity_ratio = l_value / sa_analysis['unique_values']
                if diversity_ratio <= 0.3:
                    privacy_level = "🟢 Good diversity requirement"
                elif diversity_ratio <= 0.6:
                    privacy_level = "🟡 Moderate diversity requirement"
                else:
                    privacy_level = "🔴 High diversity requirement"
                
                st.markdown(f"**Diversity Assessment:** {privacy_level}")
                st.metric("Diversity Ratio", f"{diversity_ratio:.2f}")

        # Validation and Warnings
        validation_passed = True
        
        if not selected_qi_cols:
            st.sidebar.error("⚠️ Please select at least one Quasi-Identifier column")
            validation_passed = False
        
        if not selected_sa_col or selected_sa_col == "No columns available":
            st.sidebar.error("⚠️ Please select a valid Sensitive Attribute column")
            validation_passed = False
        
        if l_value > k_value and k_value > 0:
            st.sidebar.warning("⚠️ l-value is greater than k-value. This is acceptable but may be harder to achieve.")
        
        if sa_analysis.get('unique_values') and l_value > sa_analysis['unique_values']:
            st.sidebar.error(f"⚠️ l-value ({l_value}) cannot exceed unique sensitive values ({sa_analysis['unique_values']})")
            validation_passed = False

        if validation_passed:
            st.sidebar.success("✅ Configuration is valid")

        return {
            "qi_cols": selected_qi_cols,
            "k": k_value,
            "l": l_value,
            "sensitive_column": selected_sa_col,
            "diversity_type": selected_diversity,
            "generalization_strategy": selected_strategy,
            "show_detailed_metrics": show_detailed_metrics
        }

    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Performs l-diversity anonymization.
        """
        qi_cols = parameters.get("qi_cols", [])
        k_val = parameters.get("k", 2)
        l_val = parameters.get("l", 2)
        sensitive_col = parameters.get("sensitive_column")
        diversity_type = parameters.get("diversity_type", "distinct")
        strategy = parameters.get("generalization_strategy", "optimal")
        show_metrics = parameters.get("show_detailed_metrics", True)

        # Validation
        if not qi_cols:
            st.error("❌ l-Diversity Error: Please select at least one Quasi-Identifier column.")
            return pd.DataFrame()

        if not sensitive_col or sensitive_col not in df_input.columns:
            st.error("❌ l-Diversity Error: Please select a valid Sensitive Attribute column.")
            return pd.DataFrame()

        if k_val < 2:
            st.error("❌ l-Diversity Error: k-value must be at least 2.")
            return pd.DataFrame()
        
        if l_val < 2:
            st.error("❌ l-Diversity Error: l-value must be at least 2.")
            return pd.DataFrame()

        # Check sensitive attribute diversity
        unique_sensitive_values = df_input[sensitive_col].nunique()
        if l_val > unique_sensitive_values:
            st.error(f"❌ l-Diversity Error: l-value ({l_val}) cannot exceed unique sensitive values ({unique_sensitive_values}).")
            return pd.DataFrame()

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing l-diversity algorithm...")
            progress_bar.progress(15)
            
            status_text.text("🔄 Analyzing sensitive attribute distribution...")
            progress_bar.progress(30)
            
            status_text.text("🔄 Applying k-anonymity preprocessing...")
            progress_bar.progress(45)
            
            status_text.text(f"🔄 Enforcing {diversity_type} l-diversity...")
            progress_bar.progress(70)
            
            # Apply l-diversity
            anonymized_df, metrics = apply_l_diversity(
                df=df_input.copy(),
                k=k_val,
                l=l_val,
                qi_columns=qi_cols,
                sensitive_column=sensitive_col,
                diversity_type=diversity_type,
                generalization_strategy=strategy
            )
            
            progress_bar.progress(85)
            status_text.text("🔄 Calculating diversity metrics...")
            
            if anonymized_df.empty:
                st.error("❌ l-Diversity failed to produce anonymized data. Try reducing l-value or k-value.")
                return pd.DataFrame()
            
            progress_bar.progress(100)
            status_text.text("✅ l-Diversity completed successfully!")
            
            # Display results
            st.success(f"✅ **l-Diversity Applied Successfully**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("k-value", k_val)
            with col2:
                st.metric("l-value", l_val)
            with col3:
                st.metric("Diversity Type", diversity_type.title())
            with col4:
                compliance = metrics.get('l_diversity_compliance', 0)
                st.metric("l-Diversity Compliance", f"{compliance:.1%}")
            
            # Show detailed metrics
            if show_metrics and metrics:
                with st.expander("📊 Detailed Privacy & Utility Metrics", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔒 Privacy Metrics**")
                        st.metric("k-Anonymous Groups", f"{metrics.get('anonymized_equivalence_classes', 'N/A')}")
                        st.metric("l-Diverse Groups", f"{metrics.get('l_diverse_groups', 'N/A')}")
                        st.metric("Min Group Size", metrics.get('min_group_size', 'N/A'))
                        st.metric("Min Diversity Score", f"{metrics.get('min_diversity_score', 0):.1f}")
                        
                    with col2:
                        st.markdown("**📈 Utility Metrics**")
                        st.metric("Information Loss", f"{metrics.get('information_loss', 0):.1f}%")
                        st.metric("Suppression Ratio", f"{metrics.get('suppression_ratio', 0):.1f}%")
                        st.metric("Avg Group Size", f"{metrics.get('avg_group_size', 0):.1f}")
                        st.metric("Max Diversity Score", f"{metrics.get('max_diversity_score', 0):.1f}")
                    
                    # Sensitive attribute analysis
                    st.markdown("**🎯 Sensitive Attribute Analysis**")
                    original_sa_dist = df_input[sensitive_col].value_counts().head(5)
                    anonymized_sa_dist = anonymized_df[sensitive_col].value_counts().head(5)
                    
                    analysis_df = pd.DataFrame({
                        'Value': original_sa_dist.index,
                        'Original Count': original_sa_dist.values,
                        'Original %': (original_sa_dist.values / len(df_input) * 100).round(1),
                        'Anonymized Count': [anonymized_sa_dist.get(val, 0) for val in original_sa_dist.index],
                        'Anonymized %': [(anonymized_sa_dist.get(val, 0) / len(anonymized_df) * 100) if len(anonymized_df) > 0 else 0 for val in original_sa_dist.index]
                    })
                    st.dataframe(analysis_df, use_container_width=True)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return anonymized_df

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ l-Diversity Error: {str(e)}")
            st.exception(e)
            return pd.DataFrame()

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """
        Builds the l-diversity specific configuration for export.
        """
        return {
            "qi_cols": st.session_state.get(f"{unique_key_prefix}_qi_cols", []),
            "k": st.session_state.get(f"{unique_key_prefix}_k_value", 2),
            "l": st.session_state.get(f"{unique_key_prefix}_l_value", 2),
            "sensitive_column": st.session_state.get(f"{unique_key_prefix}_sensitive_col"),
            "diversity_type": st.session_state.get(f"{unique_key_prefix}_diversity_type", "distinct"),
            "generalization_strategy": st.session_state.get(f"{unique_key_prefix}_generalization_strategy", "optimal"),
            "show_detailed_metrics": st.session_state.get(f"{unique_key_prefix}_show_metrics", True)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """
        Applies imported l-diversity configuration parameters to the session state.
        """
        # Set QI columns
        imported_qi_cols = config_params.get("qi_cols", [])
        st.session_state[f"{unique_key_prefix}_qi_cols"] = [col for col in imported_qi_cols if col in all_cols]
        
        # Set parameters
        st.session_state[f"{unique_key_prefix}_k_value"] = config_params.get("k", 2)
        st.session_state[f"{unique_key_prefix}_l_value"] = config_params.get("l", 2)
        
        # Set sensitive column
        sensitive_col = config_params.get("sensitive_column")
        if sensitive_col in all_cols:
            st.session_state[f"{unique_key_prefix}_sensitive_col"] = sensitive_col
        
        # Set diversity type
        diversity_type = config_params.get("diversity_type", "distinct")
        if diversity_type in ["distinct", "entropy", "recursive"]:
            st.session_state[f"{unique_key_prefix}_diversity_type"] = diversity_type
        
        # Set strategy
        strategy = config_params.get("generalization_strategy", "optimal")
        if strategy in ["optimal", "greedy", "binary"]:
            st.session_state[f"{unique_key_prefix}_generalization_strategy"] = strategy
        
        # Set metrics preference
        st.session_state[f"{unique_key_prefix}_show_metrics"] = config_params.get("show_detailed_metrics", True)


# Factory function for the plugin loader
def get_plugin():
    """Returns an instance of the LDiversityPlugin."""
    return LDiversityPlugin()
