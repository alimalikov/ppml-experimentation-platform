"""
Rank Swapping Anonymization Plugin

This plugin implements rank swapping anonymization, where values in numerical
attributes are swapped with other values within a certain rank distance to
preserve statistical properties while providing privacy protection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import random
from ..base_anonymizer import Anonymizer


class RankSwappingAnonymizer(Anonymizer):
    """
    Rank Swapping anonymization implementation.
    
    Rank swapping reorders values within numerical columns by swapping values
    that are close in rank, preserving distribution properties while providing
    privacy protection.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Rank Swapping"
        self.category = "Data Transformation"
        self.swap_range = 5
        self.selected_columns = []
        self.preserve_order = False
        self.random_seed = None
        
    def get_name(self) -> str:
        return self.name
        
    def get_category(self) -> str:
        return self.category
        
    def get_sidebar_ui(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Streamlit sidebar UI for rank swapping configuration."""
        st.sidebar.markdown("### Rank Swapping Configuration")
        
        # Column selection
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            st.sidebar.warning("No numeric columns available for rank swapping")
            return {}
            
        selected_columns = st.sidebar.multiselect(
            "Select numeric columns for rank swapping:",
            options=numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))],
            help="Choose which numeric columns to apply rank swapping to"
        )
        
        # Swap range
        swap_range = st.sidebar.slider(
            "Swap Range (p):",
            min_value=1,
            max_value=min(20, len(data) // 10) if len(data) > 10 else 5,
            value=5,
            help="Maximum distance for rank swapping. Larger values provide more privacy but less utility."
        )
        
        # Additional options
        preserve_order = st.sidebar.checkbox(
            "Preserve relative order",
            value=False,
            help="Maintain relative ordering within swap ranges"
        )
        
        random_seed = st.sidebar.number_input(
            "Random seed (optional):",
            value=None,
            help="Set for reproducible results"
        )
        
        # Privacy-utility tradeoff info
        st.sidebar.info(
            f"🔒 **Privacy Level**: {'High' if swap_range > 10 else 'Medium' if swap_range > 5 else 'Low'}\n\n"
            f"📊 **Utility Level**: {'Low' if swap_range > 10 else 'Medium' if swap_range > 5 else 'High'}\n\n"
            f"Swap range of {swap_range} means values can be swapped with others "
            f"within ±{swap_range} positions in the sorted order."
        )
        
        return {
            'selected_columns': selected_columns,
            'swap_range': swap_range,
            'preserve_order': preserve_order,
            'random_seed': random_seed
        }
        
    def anonymize(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply rank swapping anonymization to the dataset."""
        if not config:
            return data.copy(), {"error": "No configuration provided"}
            
        selected_columns = config.get('selected_columns', [])
        if not selected_columns:
            return data.copy(), {"error": "No columns selected for rank swapping"}
            
        # Store configuration
        self.selected_columns = selected_columns
        self.swap_range = config.get('swap_range', 5)
        self.preserve_order = config.get('preserve_order', False)
        self.random_seed = config.get('random_seed')
        
        # Set random seed if provided
        if self.random_seed is not None:
            random.seed(int(self.random_seed))
            np.random.seed(int(self.random_seed))
        
        anonymized_data = data.copy()
        metrics = {
            "columns_processed": [],
            "total_swaps": 0,
            "swap_statistics": {}
        }
        
        try:
            for column in selected_columns:
                if column not in data.columns:
                    continue
                    
                original_values = data[column].copy()
                swapped_values, swap_count = self._apply_rank_swapping(
                    original_values, self.swap_range, self.preserve_order
                )
                
                anonymized_data[column] = swapped_values
                metrics["columns_processed"].append(column)
                metrics["total_swaps"] += swap_count
                metrics["swap_statistics"][column] = {
                    "swaps_performed": swap_count,
                    "swap_percentage": (swap_count / len(data)) * 100,
                    "original_mean": float(original_values.mean()),
                    "anonymized_mean": float(swapped_values.mean()),
                    "original_std": float(original_values.std()),
                    "anonymized_std": float(swapped_values.std())
                }
            
            # Calculate overall metrics
            metrics.update(self._calculate_privacy_utility_metrics(data, anonymized_data, selected_columns))
            
            return anonymized_data, metrics
            
        except Exception as e:
            return data.copy(), {"error": f"Rank swapping failed: {str(e)}"}
    
    def _apply_rank_swapping(self, values: pd.Series, swap_range: int, preserve_order: bool) -> Tuple[pd.Series, int]:
        """Apply rank swapping to a single column."""
        # Handle missing values
        mask = values.notna()
        if not mask.any():
            return values, 0
            
        valid_values = values[mask].copy()
        n = len(valid_values)
        
        if n < 2:
            return values, 0
        
        # Create sorted indices
        sorted_indices = valid_values.argsort()
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(n)
        
        # Apply rank swapping
        swapped_values = valid_values.copy()
        swap_count = 0
        
        for i in range(n):
            # Define swap range
            min_rank = max(0, ranks[i] - swap_range)
            max_rank = min(n - 1, ranks[i] + swap_range)
            
            # Find candidates for swapping
            candidates = []
            for j in range(n):
                if min_rank <= ranks[j] <= max_rank and i != j:
                    candidates.append(j)
            
            if candidates:
                # Choose swap partner
                if preserve_order:
                    # Prefer closer ranks
                    distances = [abs(ranks[i] - ranks[j]) for j in candidates]
                    min_distance = min(distances)
                    closest_candidates = [candidates[k] for k, d in enumerate(distances) if d == min_distance]
                    swap_partner = random.choice(closest_candidates)
                else:
                    swap_partner = random.choice(candidates)
                
                # Perform swap
                if random.random() < 0.5:  # 50% chance to actually swap
                    swapped_values.iloc[i], swapped_values.iloc[swap_partner] = \
                        swapped_values.iloc[swap_partner], swapped_values.iloc[i]
                    swap_count += 1
        
        # Reconstruct full series
        result = values.copy()
        result[mask] = swapped_values
        
        return result, swap_count
    
    def _calculate_privacy_utility_metrics(self, original: pd.DataFrame, anonymized: pd.DataFrame, 
                                         columns: List[str]) -> Dict[str, Any]:
        """Calculate privacy and utility metrics for rank swapping."""
        metrics = {}
        
        try:
            # Privacy metrics
            total_changes = 0
            total_records = len(original)
            
            for col in columns:
                changes = (original[col] != anonymized[col]).sum()
                total_changes += changes
            
            privacy_score = min(100, (total_changes / (total_records * len(columns))) * 100)
            
            # Utility metrics (correlation preservation)
            utility_scores = []
            for col in columns:
                if len(original[col].unique()) > 1:
                    orig_corr = original[columns].corr()[col].drop(col).abs().mean()
                    anon_corr = anonymized[columns].corr()[col].drop(col).abs().mean()
                    
                    if not np.isnan(orig_corr) and not np.isnan(anon_corr) and orig_corr > 0:
                        utility_score = (anon_corr / orig_corr) * 100
                        utility_scores.append(min(100, utility_score))
            
            utility_score = np.mean(utility_scores) if utility_scores else 100
            
            metrics.update({
                "privacy_score": round(privacy_score, 2),
                "utility_score": round(utility_score, 2),
                "records_modified": total_changes,
                "modification_rate": round((total_changes / (total_records * len(columns))) * 100, 2)
            })
            
        except Exception as e:
            metrics.update({
                "privacy_score": 0,
                "utility_score": 0,
                "error": f"Metrics calculation failed: {str(e)}"
            })
        
        return metrics
    
    def build_config_export(self) -> Dict[str, Any]:
        """Export current configuration."""
        return {
            "plugin_name": self.get_name(),
            "plugin_category": self.get_category(),
            "selected_columns": self.selected_columns,
            "swap_range": self.swap_range,
            "preserve_order": self.preserve_order,
            "random_seed": self.random_seed
        }
    
    def apply_config_import(self, config: Dict[str, Any]) -> None:
        """Import and apply configuration."""
        self.selected_columns = config.get("selected_columns", [])
        self.swap_range = config.get("swap_range", 5)
        self.preserve_order = config.get("preserve_order", False)
        self.random_seed = config.get("random_seed")


def get_plugin():
    """Factory function to create plugin instance."""
    return RankSwappingAnonymizer()
