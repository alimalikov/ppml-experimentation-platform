"""
Professional t-closeness plugin for the anonymization tool.
Provides comprehensive t-closeness implementation with multiple distance measures.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import json
import numpy as np
from ..base_anonymizer import Anonymizer
from ..t_closeness_core import apply_t_closeness

class TClosenessPlugin(Anonymizer):
    """
    Professional t-closeness plugin with multiple distance measures.
    """

    def __init__(self):
        """Initialize the t-closeness plugin."""
        self._name = "t-Closeness"
        self._description = ("Most advanced privacy model that builds on l-diversity by ensuring "
                           "the distribution of sensitive values in each equivalence class is "
                           "close to the overall distribution, protecting against attribute disclosure.")

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
        Renders the t-closeness specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🎯 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About t-Closeness"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Most comprehensive privacy protection available
            - Multiple distance measures (Earth Mover's, KL-Divergence, JS-Divergence)
            - Distribution-aware anonymization
            - Protection against attribute disclosure attacks
            
            **Best for:** Highly sensitive datasets, research data, datasets requiring maximum privacy
            """)

        # Define session state keys
        qi_key = f"{unique_key_prefix}_qi_cols"
        k_key = f"{unique_key_prefix}_k_value"
        l_key = f"{unique_key_prefix}_l_value"
        t_key = f"{unique_key_prefix}_t_value"
        sa_key = f"{unique_key_prefix}_sensitive_col"
        distance_key = f"{unique_key_prefix}_distance_metric"
        diversity_key = f"{unique_key_prefix}_diversity_type"
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
            help="Column containing sensitive information requiring distribution protection"
        )

        # Analyze sensitive attribute and global distribution
        sa_analysis = {}
        global_distribution = None
        if df_raw is not None and selected_sa_col in df_raw.columns:
            sa_values = df_raw[selected_sa_col].dropna()
            sa_analysis = {
                'total_records': len(sa_values),
                'unique_values': sa_values.nunique(),
                'value_counts': sa_values.value_counts(),
                'distribution': sa_values.value_counts(normalize=True).sort_index()
            }
            global_distribution = sa_analysis['distribution']
            
            with st.sidebar.expander(f"📊 Global Distribution Analysis: {selected_sa_col}"):
                st.metric("Total Records", sa_analysis['total_records'])
                st.metric("Unique Values", sa_analysis['unique_values'])
                
                # Show distribution
                st.markdown("**Global Distribution:**")
                for value, prob in sa_analysis['distribution'].head(5).items():
                    st.write(f"• {value}: {prob:.3f} ({prob*100:.1f}%)")
                
                # Distribution uniformity assessment
                uniformity = 1.0 / sa_analysis['unique_values'] if sa_analysis['unique_values'] > 0 else 0
                max_prob = sa_analysis['distribution'].max()
                skewness = max_prob / uniformity if uniformity > 0 else 0
                
                if skewness <= 2:
                    dist_assessment = "🟢 Well-balanced distribution"
                elif skewness <= 4:
                    dist_assessment = "🟡 Moderately skewed distribution"
                else:
                    dist_assessment = "🔴 Highly skewed distribution"
                
                st.info(f"{dist_assessment}\nSkewness ratio: {skewness:.1f}")

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
            max_k_val = min(30, len(df_raw) // 2)
            default_k = min(4, max_k_val)
        else:
            max_k_val = 20
            default_k = 4

        k_value = st.sidebar.slider(
            "k-value (k-anonymity requirement):",
            min_value=min_k,
            max_value=max_k_val,
            value=st.session_state.get(k_key, default_k),
            step=1,
            key=k_key,
            help="Minimum group size for k-anonymity"
        )

        # L-value configuration
        min_l = 2
        if sa_analysis.get('unique_values'):
            max_l_val = min(sa_analysis['unique_values'], 8)
            default_l = min(3, max_l_val)
        else:
            max_l_val = 5
            default_l = 2

        l_value = st.sidebar.slider(
            "l-value (diversity requirement):",
            min_value=min_l,
            max_value=max_l_val,
            value=st.session_state.get(l_key, default_l),
            step=1,
            key=l_key,
            help="Minimum diversity for l-diversity"
        )

        # T-value configuration
        st.sidebar.subheader("🎯 t-Closeness Configuration")
        
        min_t = 0.01
        max_t = 1.0
        default_t = 0.2
        
        t_value = st.sidebar.slider(
            "t-value (distribution similarity threshold):",
            min_value=min_t,
            max_value=max_t,
            value=st.session_state.get(t_key, default_t),
            step=0.01,
            key=t_key,
            help="Maximum allowed distance between local and global distributions (lower = stricter)"
        )

        # Privacy level assessment
        if t_value <= 0.1:
            privacy_level = "🔵 Maximum Privacy"
            privacy_desc = "Very strict distribution similarity"
        elif t_value <= 0.2:
            privacy_level = "🟢 High Privacy"
            privacy_desc = "Strong distribution protection"
        elif t_value <= 0.4:
            privacy_level = "🟡 Medium Privacy"
            privacy_desc = "Balanced privacy and utility"
        else:
            privacy_level = "🔴 Low Privacy"
            privacy_desc = "Relaxed distribution requirements"

        st.sidebar.info(f"**{privacy_level}**\n{privacy_desc}")

        # Distance Metric Selection
        distance_options = {
            "earth_movers": "Earth Mover's Distance - Optimal transport distance",
            "kl_divergence": "KL Divergence - Information-theoretic measure",
            "js_divergence": "Jensen-Shannon Divergence - Symmetric KL divergence"
        }
        
        current_distance = st.session_state.get(distance_key, "earth_movers")
        selected_distance_display = st.sidebar.selectbox(
            "Distance Metric:",
            options=list(distance_options.values()),
            index=list(distance_options.keys()).index(current_distance),
            key=f"{distance_key}_display",
            help="Method for measuring distribution similarity"
        )
        
        selected_distance = [k for k, v in distance_options.items() if v == selected_distance_display][0]
        st.session_state[distance_key] = selected_distance

        # Distance metric explanation
        distance_explanations = {
            "earth_movers": "Measures minimum cost to transform one distribution to another. Best for ordinal data.",
            "kl_divergence": "Measures information lost when using one distribution to approximate another.",
            "js_divergence": "Symmetric version of KL divergence. Good general-purpose measure."
        }
        st.sidebar.info(f"**{selected_distance.replace('_', ' ').title()}:** {distance_explanations[selected_distance]}")

        # Diversity Type for l-diversity
        diversity_options = {
            "distinct": "Distinct l-Diversity",
            "entropy": "Entropy l-Diversity",
            "recursive": "Recursive (c,l)-Diversity"
        }
        
        current_diversity = st.session_state.get(diversity_key, "distinct")
        selected_diversity = st.sidebar.selectbox(
            "l-Diversity Type:",
            options=list(diversity_options.values()),
            index=list(diversity_options.keys()).index(current_diversity),
            key=diversity_key,
            help="Type of diversity measure for l-diversity component"
        )
        
        selected_diversity_key = [k for k, v in diversity_options.items() if v == selected_diversity][0]

        # Generalization Strategy
        st.sidebar.subheader("⚙️ Algorithm Configuration")
        
        strategy_options = {
            "optimal": "Optimal - Best privacy/utility balance (slower)",
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
            
            # Computational complexity warning
            if df_raw is not None:
                complexity_score = 0
                if len(df_raw) > 1000:
                    complexity_score += 1
                if len(selected_qi_cols) > 3:
                    complexity_score += 1
                if t_value < 0.1:
                    complexity_score += 1
                if selected_distance in ["kl_divergence", "js_divergence"]:
                    complexity_score += 1
                
                if complexity_score >= 3:
                    st.warning("⚠️ High computational complexity expected. Consider reducing parameters.")
                elif complexity_score >= 2:
                    st.info("ℹ️ Moderate computational complexity. Processing may take time.")
                else:
                    st.success("✅ Low computational complexity expected.")

        # Validation and Warnings
        validation_passed = True
        
        if not selected_qi_cols:
            st.sidebar.error("⚠️ Please select at least one Quasi-Identifier column")
            validation_passed = False
        
        if not selected_sa_col or selected_sa_col == "No columns available":
            st.sidebar.error("⚠️ Please select a valid Sensitive Attribute column")
            validation_passed = False
        
        if l_value > k_value:
            st.sidebar.warning("⚠️ l-value > k-value may be difficult to achieve")
        
        if sa_analysis.get('unique_values') and l_value > sa_analysis['unique_values']:
            st.sidebar.error(f"⚠️ l-value ({l_value}) exceeds unique sensitive values ({sa_analysis['unique_values']})")
            validation_passed = False

        # Estimate feasibility
        if global_distribution is not None and t_value < 0.1:
            max_prob = global_distribution.max()
            if max_prob > 0.8:
                st.sidebar.warning("⚠️ Highly skewed distribution may make strict t-closeness difficult")

        if validation_passed:
            st.sidebar.success("✅ Configuration is valid")

        return {
            "qi_cols": selected_qi_cols,
            "k": k_value,
            "l": l_value,
            "t": t_value,
            "sensitive_column": selected_sa_col,
            "distance_metric": selected_distance,
            "diversity_type": selected_diversity_key,
            "generalization_strategy": selected_strategy,
            "show_detailed_metrics": show_detailed_metrics
        }

    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Performs t-closeness anonymization.
        """
        qi_cols = parameters.get("qi_cols", [])
        k_val = parameters.get("k", 2)
        l_val = parameters.get("l", 2)
        t_val = parameters.get("t", 0.2)
        sensitive_col = parameters.get("sensitive_column")
        distance_metric = parameters.get("distance_metric", "earth_movers")
        diversity_type = parameters.get("diversity_type", "distinct")
        strategy = parameters.get("generalization_strategy", "optimal")
        show_metrics = parameters.get("show_detailed_metrics", True)

        # Validation
        if not qi_cols:
            st.error("❌ t-Closeness Error: Please select at least one Quasi-Identifier column.")
            return pd.DataFrame()

        if not sensitive_col or sensitive_col not in df_input.columns:
            st.error("❌ t-Closeness Error: Please select a valid Sensitive Attribute column.")
            return pd.DataFrame()

        if k_val < 2:
            st.error("❌ t-Closeness Error: k-value must be at least 2.")
            return pd.DataFrame()
        
        if l_val < 2:
            st.error("❌ t-Closeness Error: l-value must be at least 2.")
            return pd.DataFrame()
        
        if not (0 < t_val <= 1):
            st.error("❌ t-Closeness Error: t-value must be between 0 and 1.")
            return pd.DataFrame()

        # Check sensitive attribute
        unique_sensitive_values = df_input[sensitive_col].nunique()
        if l_val > unique_sensitive_values:
            st.error(f"❌ t-Closeness Error: l-value ({l_val}) exceeds unique sensitive values ({unique_sensitive_values}).")
            return pd.DataFrame()

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing t-closeness algorithm...")
            progress_bar.progress(10)
            
            status_text.text("🔄 Analyzing global sensitive attribute distribution...")
            progress_bar.progress(20)
            
            status_text.text("🔄 Applying k-anonymity preprocessing...")
            progress_bar.progress(35)
            
            status_text.text("🔄 Enforcing l-diversity constraints...")
            progress_bar.progress(50)
            
            status_text.text(f"🔄 Calculating {distance_metric} distances...")
            progress_bar.progress(65)
            
            status_text.text("🔄 Enforcing t-closeness constraints...")
            progress_bar.progress(80)
            
            # Apply t-closeness
            anonymized_df, metrics = apply_t_closeness(
                df=df_input.copy(),
                k=k_val,
                l=l_val,
                t=t_val,
                qi_columns=qi_cols,
                sensitive_column=sensitive_col,
                distance_metric=distance_metric,
                diversity_type=diversity_type,
                generalization_strategy=strategy
            )
            
            progress_bar.progress(90)
            status_text.text("🔄 Calculating comprehensive metrics...")
            
            if anonymized_df.empty:
                st.error("❌ t-Closeness failed. Try relaxing constraints (higher t-value, lower k/l values).")
                return pd.DataFrame()
            
            progress_bar.progress(100)
            status_text.text("✅ t-Closeness completed successfully!")
            
            # Display results
            st.success(f"✅ **t-Closeness Applied Successfully**")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("k-value", k_val)
            with col2:
                st.metric("l-value", l_val)
            with col3:
                st.metric("t-value", f"{t_val:.3f}")
            with col4:
                st.metric("Distance Metric", distance_metric.replace('_', ' ').title())
            with col5:
                compliance = metrics.get('t_closeness_compliance', 0)
                st.metric("t-Close Compliance", f"{compliance:.1%}")
            
            # Show comprehensive metrics
            if show_metrics and metrics:
                with st.expander("📊 Comprehensive Privacy & Utility Analysis", expanded=True):
                    # Create three columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**🔒 Privacy Protection**")
                        st.metric("k-Anonymous Groups", f"{metrics.get('anonymized_equivalence_classes', 'N/A')}")
                        st.metric("l-Diverse Groups", f"{metrics.get('l_diverse_groups', 'N/A')}")
                        st.metric("t-Close Groups", f"{metrics.get('t_close_groups', 'N/A')}")
                        st.metric("Min Group Size", metrics.get('min_group_size', 'N/A'))
                        
                    with col2:
                        st.markdown("**📊 Distribution Analysis**")
                        st.metric("Min Distance", f"{metrics.get('min_distance', 0):.4f}")
                        st.metric("Max Distance", f"{metrics.get('max_distance', 0):.4f}")
                        st.metric("Avg Distance", f"{metrics.get('avg_distance', 0):.4f}")
                        st.metric("Distance Threshold", f"{t_val:.3f}")
                        
                    with col3:
                        st.markdown("**📈 Utility Metrics**")
                        st.metric("Information Loss", f"{metrics.get('information_loss', 0):.1f}%")
                        st.metric("Suppression Ratio", f"{metrics.get('suppression_ratio', 0):.1f}%")
                        st.metric("Avg Group Size", f"{metrics.get('avg_group_size', 0):.1f}")
                        st.metric("Diversity Score", f"{metrics.get('avg_diversity_score', 0):.1f}")
                    
                    # Distribution comparison
                    st.markdown("**🎯 Sensitive Attribute Distribution Comparison**")
                    
                    # Create side-by-side distribution analysis
                    col_orig, col_anon = st.columns(2)
                    
                    with col_orig:
                        st.markdown("**Original Distribution**")
                        orig_dist = df_input[sensitive_col].value_counts().head(8)
                        orig_df = pd.DataFrame({
                            'Value': orig_dist.index,
                            'Count': orig_dist.values,
                            'Percentage': (orig_dist.values / len(df_input) * 100).round(2)
                        })
                        st.dataframe(orig_df, use_container_width=True)
                    
                    with col_anon:
                        st.markdown("**Anonymized Distribution**")
                        anon_dist = anonymized_df[sensitive_col].value_counts().head(8)
                        anon_df = pd.DataFrame({
                            'Value': anon_dist.index,
                            'Count': anon_dist.values,
                            'Percentage': (anon_dist.values / len(anonymized_df) * 100).round(2) if len(anonymized_df) > 0 else [0] * len(anon_dist)
                        })
                        st.dataframe(anon_df, use_container_width=True)
                    
                    # Privacy assessment
                    st.markdown("**🏆 Privacy Assessment Summary**")
                    assessment_data = {
                        'Privacy Model': ['k-Anonymity', 'l-Diversity', 't-Closeness'],
                        'Requirement': [f'k ≥ {k_val}', f'l ≥ {l_val}', f't ≤ {t_val:.3f}'],
                        'Compliance': [
                            '✅ Satisfied' if metrics.get('min_group_size', 0) >= k_val else '❌ Failed',
                            '✅ Satisfied' if metrics.get('l_diversity_compliance', 0) >= 0.95 else '❌ Failed',
                            '✅ Satisfied' if metrics.get('t_closeness_compliance', 0) >= 0.95 else '❌ Failed'
                        ],
                        'Score': [
                            f"{metrics.get('min_group_size', 0)}/{k_val}",
                            f"{metrics.get('l_diversity_compliance', 0):.1%}",
                            f"{metrics.get('t_closeness_compliance', 0):.1%}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(assessment_data), use_container_width=True)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return anonymized_df

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ t-Closeness Error: {str(e)}")
            st.exception(e)
            return pd.DataFrame()

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """
        Builds the t-closeness specific configuration for export.
        """
        return {
            "qi_cols": st.session_state.get(f"{unique_key_prefix}_qi_cols", []),
            "k": st.session_state.get(f"{unique_key_prefix}_k_value", 2),
            "l": st.session_state.get(f"{unique_key_prefix}_l_value", 2),
            "t": st.session_state.get(f"{unique_key_prefix}_t_value", 0.2),
            "sensitive_column": st.session_state.get(f"{unique_key_prefix}_sensitive_col"),
            "distance_metric": st.session_state.get(f"{unique_key_prefix}_distance_metric", "earth_movers"),
            "diversity_type": st.session_state.get(f"{unique_key_prefix}_diversity_type", "distinct"),
            "generalization_strategy": st.session_state.get(f"{unique_key_prefix}_generalization_strategy", "optimal"),
            "show_detailed_metrics": st.session_state.get(f"{unique_key_prefix}_show_metrics", True)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """
        Applies imported t-closeness configuration parameters to the session state.
        """
        # Set QI columns
        imported_qi_cols = config_params.get("qi_cols", [])
        st.session_state[f"{unique_key_prefix}_qi_cols"] = [col for col in imported_qi_cols if col in all_cols]
        
        # Set parameters
        st.session_state[f"{unique_key_prefix}_k_value"] = config_params.get("k", 2)
        st.session_state[f"{unique_key_prefix}_l_value"] = config_params.get("l", 2)
        st.session_state[f"{unique_key_prefix}_t_value"] = config_params.get("t", 0.2)
        
        # Set sensitive column
        sensitive_col = config_params.get("sensitive_column")
        if sensitive_col in all_cols:
            st.session_state[f"{unique_key_prefix}_sensitive_col"] = sensitive_col
        
        # Set distance metric
        distance_metric = config_params.get("distance_metric", "earth_movers")
        if distance_metric in ["earth_movers", "kl_divergence", "js_divergence"]:
            st.session_state[f"{unique_key_prefix}_distance_metric"] = distance_metric
        
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
    """Returns an instance of the TClosenessPlugin."""
    return TClosenessPlugin()
