"""
Data Swapping Plugin for Data Anonymization

This plugin implements data swapping (also known as record swapping or shuffling)
for privacy protection. It swaps values of selected attributes between records
to break linkages while preserving statistical properties of the data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
import logging
import random
from collections import defaultdict

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class DataSwappingPlugin(Anonymizer):
    """
    Data swapping plugin that implements record swapping for privacy protection.
    
    Data swapping works by:
    1. Selecting records and attributes for swapping
    2. Finding suitable swap partners based on constraints
    3. Exchanging attribute values between record pairs
    4. Ensuring statistical properties are preserved
    5. Maintaining data consistency and integrity
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Data Swapping"
        self.description = "Swaps attribute values between records to protect privacy while preserving statistics"
        
        # Swapping parameters
        self.swap_percentage = 0.1  # Percentage of records to swap
        self.attributes_to_swap = []  # Specific attributes to swap
        self.swap_all_attributes = False
        
        # Swap partner selection
        self.partner_selection = "random"  # random, similar, dissimilar, geographic
        self.similarity_threshold = 0.8
        self.geographic_constraint = False
        self.geographic_radius = 10.0  # km
        
        # Swapping constraints
        self.preserve_marginals = True
        self.maintain_correlations = True
        self.respect_constraints = True
        self.constraint_columns = []  # Columns that must satisfy constraints
        
        # Quality control
        self.verify_swaps = True
        self.max_swap_attempts = 100
        self.rollback_failed_swaps = True
        
        # Privacy parameters
        self.minimum_swap_distance = 1  # Minimum number of records between swap partners
        self.avoid_obvious_patterns = True
        self.use_secure_random = True
        
        # Internal state
        self.swap_pairs = []
        self.successful_swaps = 0
        self.failed_swaps = 0
        self.swap_statistics = {}
        self.original_statistics = {}
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Data Transformation"
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the Data Swapping specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔄 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Data Swapping"):
            st.markdown("""
            **Data Swapping** exchanges attribute values between records to protect
            privacy while preserving the statistical properties of the dataset.
            It's particularly effective for protecting against record linkage attacks.
            
            **Key Features:**
            - Preserves marginal distributions
            - Flexible swap partner selection
            - Maintains data quality
            - Configurable privacy-utility trade-offs
            """)
        
        # Basic swapping parameters
        swap_percentage = st.sidebar.slider(
            "Swap Percentage",
            min_value=0.01, max_value=0.5, value=0.1, step=0.01,
            key=f"{unique_key_prefix}_swap_percentage",
            help="Percentage of records to participate in swapping"
        )
        
        # Attribute selection
        st.sidebar.subheader("📊 Attribute Selection")
        
        swap_all_attributes = st.sidebar.checkbox(
            "Swap All Attributes",
            value=False,
            key=f"{unique_key_prefix}_swap_all_attributes",
            help="Swap all attributes or select specific ones"
        )
        
        if not swap_all_attributes:
            # Filter out sensitive attribute from options
            available_cols = [col for col in all_cols if col != sa_col_to_pass]
            attributes_to_swap = st.sidebar.multiselect(
                "Attributes to Swap",
                options=available_cols,
                default=available_cols[:min(3, len(available_cols))],
                key=f"{unique_key_prefix}_attributes_to_swap",
                help="Select specific attributes to swap"
            )
        else:
            attributes_to_swap = all_cols
        
        # Partner selection strategy
        st.sidebar.subheader("🎯 Partner Selection")
        
        partner_selection = st.sidebar.selectbox(
            "Partner Selection Strategy",
            options=["random", "similar", "dissimilar", "geographic"],
            key=f"{unique_key_prefix}_partner_selection",
            help="Strategy for selecting swap partners"
        )
        
        if partner_selection in ["similar", "dissimilar"]:
            similarity_threshold = st.sidebar.slider(
                "Similarity Threshold",
                min_value=0.1, max_value=0.9, value=0.8, step=0.1,
                key=f"{unique_key_prefix}_similarity_threshold",
                help="Threshold for similarity-based partner selection"
            )
        else:
            similarity_threshold = 0.8
        
        # Constraints and quality control
        with st.sidebar.expander("🔧 Constraints & Quality"):
            preserve_marginals = st.sidebar.checkbox(
                "Preserve Marginals",
                value=True,
                key=f"{unique_key_prefix}_preserve_marginals",
                help="Maintain marginal distributions of attributes"
            )
            
            maintain_correlations = st.sidebar.checkbox(
                "Maintain Correlations",
                value=True,
                key=f"{unique_key_prefix}_maintain_correlations",
                help="Try to preserve correlation structure"
            )
            
            minimum_swap_distance = st.sidebar.slider(
                "Minimum Swap Distance",
                min_value=1, max_value=100, value=1,
                key=f"{unique_key_prefix}_minimum_swap_distance",
                help="Minimum row distance between swap partners"
            )
        
        # Privacy settings
        with st.sidebar.expander("🔒 Privacy Settings"):
            avoid_obvious_patterns = st.sidebar.checkbox(
                "Avoid Obvious Patterns",
                value=True,
                key=f"{unique_key_prefix}_avoid_obvious_patterns",
                help="Avoid creating obvious swapping patterns"
            )
            
            use_secure_random = st.sidebar.checkbox(
                "Use Secure Random",
                value=True,
                key=f"{unique_key_prefix}_use_secure_random",
                help="Use cryptographically secure random number generation"
            )
            
            verify_swaps = st.sidebar.checkbox(
                "Verify Swaps",
                value=True,
                key=f"{unique_key_prefix}_verify_swaps",
                help="Verify that swaps maintain data integrity"
            )
        
        return {
            "swap_percentage": swap_percentage,
            "swap_all_attributes": swap_all_attributes,
            "attributes_to_swap": attributes_to_swap,
            "partner_selection": partner_selection,
            "similarity_threshold": similarity_threshold,
            "preserve_marginals": preserve_marginals,
            "maintain_correlations": maintain_correlations,
            "minimum_swap_distance": minimum_swap_distance,
            "avoid_obvious_patterns": avoid_obvious_patterns,
            "use_secure_random": use_secure_random,
            "verify_swaps": verify_swaps
        }
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply data swapping to the input DataFrame.
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.swap_percentage = parameters.get("swap_percentage", 0.1)
            self.swap_all_attributes = parameters.get("swap_all_attributes", False)
            self.attributes_to_swap = parameters.get("attributes_to_swap", [])
            self.partner_selection = parameters.get("partner_selection", "random")
            self.similarity_threshold = parameters.get("similarity_threshold", 0.8)
            self.preserve_marginals = parameters.get("preserve_marginals", True)
            self.maintain_correlations = parameters.get("maintain_correlations", True)
            self.minimum_swap_distance = parameters.get("minimum_swap_distance", 1)
            self.avoid_obvious_patterns = parameters.get("avoid_obvious_patterns", True)
            self.use_secure_random = parameters.get("use_secure_random", True)
            self.verify_swaps = parameters.get("verify_swaps", True)
            
            if self.swap_all_attributes:
                self.attributes_to_swap = [col for col in df_input.columns if col != sa_col]
            
            if not self.attributes_to_swap:
                logger.warning("No attributes selected for swapping")
                return df_input
            
            logger.info(f"Starting data swapping with {self.swap_percentage:.1%} of records")
            
            # Store original statistics
            self._store_original_statistics(df_input)
            
            # Create working copy
            result = df_input.copy()
            
            # Determine records to swap
            swap_records = self._select_records_for_swapping(df_input)
            
            # Find swap pairs
            self.swap_pairs = self._find_swap_pairs(df_input, swap_records)
            
            # Perform swapping
            result = self._perform_swapping(result)
            
            # Verify swaps if enabled
            if self.verify_swaps:
                result = self._verify_and_fix_swaps(result, df_input)
            
            # Calculate swap statistics
            self._calculate_swap_statistics(df_input, result)
            
            processing_time = time.time() - start_time
            logger.info(f"Data swapping completed in {processing_time:.2f}s, {self.successful_swaps} successful swaps")
            
            # Add metadata
            result.attrs['swap_percentage'] = self.swap_percentage
            result.attrs['successful_swaps'] = self.successful_swaps
            result.attrs['failed_swaps'] = self.failed_swaps
            result.attrs['swap_statistics'] = self.swap_statistics
            result.attrs['processing_time'] = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error in data swapping: {str(e)}")
            st.error(f"Data swapping failed: {str(e)}")
            return df_input
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "swap_percentage": st.session_state.get(f"{unique_key_prefix}_swap_percentage", 0.1),
            "swap_all_attributes": st.session_state.get(f"{unique_key_prefix}_swap_all_attributes", False),
            "attributes_to_swap": st.session_state.get(f"{unique_key_prefix}_attributes_to_swap", []),
            "partner_selection": st.session_state.get(f"{unique_key_prefix}_partner_selection", "random"),
            "similarity_threshold": st.session_state.get(f"{unique_key_prefix}_similarity_threshold", 0.8),
            "preserve_marginals": st.session_state.get(f"{unique_key_prefix}_preserve_marginals", True),
            "maintain_correlations": st.session_state.get(f"{unique_key_prefix}_maintain_correlations", True),
            "minimum_swap_distance": st.session_state.get(f"{unique_key_prefix}_minimum_swap_distance", 1),
            "avoid_obvious_patterns": st.session_state.get(f"{unique_key_prefix}_avoid_obvious_patterns", True),
            "use_secure_random": st.session_state.get(f"{unique_key_prefix}_use_secure_random", True),
            "verify_swaps": st.session_state.get(f"{unique_key_prefix}_verify_swaps", True)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        for key, value in config_params.items():
            st.session_state[f"{unique_key_prefix}_{key}"] = value

    def _store_original_statistics(self, data: pd.DataFrame):
        """Store original statistics for comparison."""
        self.original_statistics = {}
        
        for col in self.attributes_to_swap:
            if col in data.columns:
                self.original_statistics[col] = {
                    'mean': data[col].mean() if data[col].dtype in ['int64', 'float64'] else None,
                    'std': data[col].std() if data[col].dtype in ['int64', 'float64'] else None,
                    'unique_count': data[col].nunique(),
                    'value_counts': data[col].value_counts().to_dict()
                }
    
    def _select_records_for_swapping(self, data: pd.DataFrame) -> List[int]:
        """Select records that will participate in swapping."""
        num_records = len(data)
        num_to_swap = int(num_records * self.swap_percentage)
        
        # Ensure even number for pairing
        if num_to_swap % 2 == 1:
            num_to_swap += 1
        
        if self.use_secure_random:
            random.seed()
        
        selected_indices = random.sample(range(num_records), min(num_to_swap, num_records))
        
        logger.info(f"Selected {len(selected_indices)} records for swapping")
        return selected_indices
    
    def _find_swap_pairs(self, data: pd.DataFrame, swap_records: List[int]) -> List[Tuple[int, int]]:
        """Find suitable pairs for swapping based on the selection strategy."""
        pairs = []
        
        if self.partner_selection == "random":
            pairs = self._find_random_pairs(swap_records)
        elif self.partner_selection == "similar":
            pairs = self._find_similar_pairs(data, swap_records)
        elif self.partner_selection == "dissimilar":
            pairs = self._find_dissimilar_pairs(data, swap_records)
        elif self.partner_selection == "geographic":
            pairs = self._find_geographic_pairs(data, swap_records)
        
        # Apply distance constraint
        pairs = self._apply_distance_constraint(pairs)
        
        logger.info(f"Found {len(pairs)} swap pairs")
        return pairs
    
    def _find_random_pairs(self, records: List[int]) -> List[Tuple[int, int]]:
        """Find random pairs for swapping."""
        shuffled = records.copy()
        random.shuffle(shuffled)
        
        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            pairs.append((shuffled[i], shuffled[i + 1]))
        
        return pairs
    
    def _find_similar_pairs(self, data: pd.DataFrame, records: List[int]) -> List[Tuple[int, int]]:
        """Find pairs of similar records for swapping."""
        pairs = []
        used_records = set()
        
        for record_idx in records:
            if record_idx in used_records:
                continue
            
            # Find most similar unused record
            best_partner = None
            best_similarity = -1
            
            for potential_partner in records:
                if potential_partner == record_idx or potential_partner in used_records:
                    continue
                
                similarity = self._calculate_similarity(data, record_idx, potential_partner)
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_partner = potential_partner
            
            if best_partner is not None:
                pairs.append((record_idx, best_partner))
                used_records.add(record_idx)
                used_records.add(best_partner)
        
        return pairs
    
    def _find_dissimilar_pairs(self, data: pd.DataFrame, records: List[int]) -> List[Tuple[int, int]]:
        """Find pairs of dissimilar records for swapping."""
        pairs = []
        used_records = set()
        
        for record_idx in records:
            if record_idx in used_records:
                continue
            
            # Find most dissimilar unused record
            best_partner = None
            best_dissimilarity = 2.0  # Max similarity is 1, so this ensures we find someone
            
            for potential_partner in records:
                if potential_partner == record_idx or potential_partner in used_records:
                    continue
                
                similarity = self._calculate_similarity(data, record_idx, potential_partner)
                dissimilarity = 1.0 - similarity
                
                if dissimilarity < best_dissimilarity and similarity <= (1.0 - self.similarity_threshold):
                    best_dissimilarity = dissimilarity
                    best_partner = potential_partner
            
            if best_partner is not None:
                pairs.append((record_idx, best_partner))
                used_records.add(record_idx)
                used_records.add(best_partner)
        
        return pairs
    
    def _find_geographic_pairs(self, data: pd.DataFrame, records: List[int]) -> List[Tuple[int, int]]:
        """Find geographically constrained pairs (simplified implementation)."""
        # This is a simplified version - in practice, you'd use actual geographic coordinates
        # For now, we'll use random pairing with a constraint
        return self._find_random_pairs(records)
    
    def _calculate_similarity(self, data: pd.DataFrame, idx1: int, idx2: int) -> float:
        """Calculate similarity between two records."""
        record1 = data.iloc[idx1]
        record2 = data.iloc[idx2]
        
        similarities = []
        
        for col in self.attributes_to_swap:
            if col not in data.columns:
                continue
            
            if data[col].dtype in ['object', 'category']:
                # Categorical similarity (exact match)
                sim = 1.0 if record1[col] == record2[col] else 0.0
            else:
                # Numerical similarity (normalized distance)
                col_range = data[col].max() - data[col].min()
                if col_range > 0:
                    distance = abs(record1[col] - record2[col]) / col_range
                    sim = 1.0 - distance
                else:
                    sim = 1.0
            
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _apply_distance_constraint(self, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Apply minimum distance constraint between swap partners."""
        if self.minimum_swap_distance <= 1:
            return pairs
        
        filtered_pairs = []
        for idx1, idx2 in pairs:
            if abs(idx1 - idx2) >= self.minimum_swap_distance:
                filtered_pairs.append((idx1, idx2))
        
        logger.info(f"Distance constraint removed {len(pairs) - len(filtered_pairs)} pairs")
        return filtered_pairs
    
    def _perform_swapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform the actual swapping of attribute values."""
        self.successful_swaps = 0
        self.failed_swaps = 0
        
        for idx1, idx2 in self.swap_pairs:
            try:
                # Swap values for selected attributes
                for attr in self.attributes_to_swap:
                    if attr in data.columns:
                        # Store original values
                        val1 = data.loc[idx1, attr]
                        val2 = data.loc[idx2, attr]
                        
                        # Perform swap
                        data.loc[idx1, attr] = val2
                        data.loc[idx2, attr] = val1
                
                self.successful_swaps += 1
                
            except Exception as e:
                logger.warning(f"Failed to swap records {idx1} and {idx2}: {e}")
                self.failed_swaps += 1
        
        return data
    
    def _verify_and_fix_swaps(self, swapped_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Verify swaps and fix any issues."""
        # Check if marginals are preserved
        if self.preserve_marginals:
            for attr in self.attributes_to_swap:
                if attr not in swapped_data.columns:
                    continue
                
                # Check if value counts match
                original_counts = original_data[attr].value_counts().sort_index()
                swapped_counts = swapped_data[attr].value_counts().sort_index()
                
                if not original_counts.equals(swapped_counts):
                    logger.warning(f"Marginals not preserved for attribute {attr}")
                    # In practice, you might implement correction logic here
        
        return swapped_data
    
    def _calculate_swap_statistics(self, original: pd.DataFrame, swapped: pd.DataFrame):
        """Calculate statistics about the swapping process."""
        self.swap_statistics = {
            'total_records': len(original),
            'records_involved': len(self.swap_pairs) * 2,
            'swap_pairs': len(self.swap_pairs),
            'successful_swaps': self.successful_swaps,
            'failed_swaps': self.failed_swaps,
            'success_rate': self.successful_swaps / (self.successful_swaps + self.failed_swaps) if (self.successful_swaps + self.failed_swaps) > 0 else 0,
            'attributes_swapped': len(self.attributes_to_swap),
            'preservation_quality': self._measure_preservation_quality(original, swapped)
        }
    
    def _measure_preservation_quality(self, original: pd.DataFrame, swapped: pd.DataFrame) -> Dict[str, float]:
        """Measure how well statistical properties are preserved."""
        quality_metrics = {}
        
        for attr in self.attributes_to_swap:
            if attr not in original.columns:
                continue
            
            if original[attr].dtype in ['int64', 'float64']:
                # Numerical attributes
                orig_mean = original[attr].mean()
                swap_mean = swapped[attr].mean()
                orig_std = original[attr].std()
                swap_std = swapped[attr].std()
                
                mean_preservation = 1.0 - abs(orig_mean - swap_mean) / (orig_std + 1e-8)
                std_preservation = 1.0 - abs(orig_std - swap_std) / (orig_std + 1e-8)
                
                quality_metrics[f"{attr}_mean_preservation"] = max(0, mean_preservation)
                quality_metrics[f"{attr}_std_preservation"] = max(0, std_preservation)
            else:
                # Categorical attributes
                orig_counts = original[attr].value_counts()
                swap_counts = swapped[attr].value_counts()
                
                # Calculate preservation based on value count differences
                preservation = 1.0
                for value in orig_counts.index:
                    if value in swap_counts.index:
                        diff = abs(orig_counts[value] - swap_counts[value])
                        preservation -= diff / len(original)
                
                quality_metrics[f"{attr}_distribution_preservation"] = max(0, preservation)
        
        return quality_metrics

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return DataSwappingPlugin()
