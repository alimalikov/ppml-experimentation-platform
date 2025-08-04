"""
Incognito Anonymization Plugin for Data Anonymization

This plugin implements the Incognito algorithm for k-anonymity, which uses a
bottom-up approach to find all possible k-anonymous generalizations and selects
the optimal one based on information loss metrics. It explores the generalization
lattice systematically to find minimal generalizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
import logging
import itertools
from collections import defaultdict

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class IncognitoAnonymizationPlugin(Anonymizer):
    """
    Incognito k-anonymity plugin that implements the Incognito algorithm.
    
    The Incognito algorithm works by:
    1. Building a lattice of all possible generalizations
    2. Starting from the most specific (bottom) level
    3. Checking k-anonymity for each generalization combination
    4. Pruning the search space using monotonicity properties
    5. Finding the minimal k-anonymous generalization
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Incognito k-Anonymity"
        self.description = "Bottom-up lattice search for optimal k-anonymous generalizations"
        
        # k-anonymity parameters
        self.k_value = 3
        self.optimization_goal = "minimal_loss"  # minimal_loss, maximal_utility, balanced
        
        # Incognito algorithm parameters
        self.max_generalization_level = 5
        self.use_pruning = True
        self.lattice_exploration = "breadth_first"  # breadth_first, depth_first, best_first
        self.early_termination = True
        
        # Generalization hierarchy parameters
        self.auto_build_hierarchies = True
        self.hierarchy_depth = 4
        self.categorical_grouping = "frequency"  # frequency, semantic, manual
        self.numerical_intervals = 10
        
        # Quality parameters
        self.prefer_local_recoding = True
        self.allow_partial_suppression = True
        self.suppression_limit = 0.05
        
        # Performance parameters
        self.use_sampling = False
        self.sample_size = 1000
        self.parallel_evaluation = True
        self.cache_evaluations = True
        
        # Internal state
        self.generalization_lattice = {}
        self.hierarchy_trees = {}
        self.k_anonymous_nodes = set()
        self.optimal_generalization = None
        self.evaluation_cache = {}
        self.search_statistics = {}
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Clustering & Grouping"
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the Incognito k-anonymity specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔍 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Incognito k-Anonymity"):
            st.markdown("""
            **Incognito k-Anonymity** uses a bottom-up lattice search to find the
            optimal k-anonymous generalization. It systematically explores all possible
            generalizations and selects the one with minimal information loss.
            
            **Key Features:**
            - Systematic lattice exploration
            - Optimal generalization selection
            - Efficient pruning strategies
            - Multiple optimization goals
            """)
        
        # Basic k-anonymity parameters
        k_value = st.sidebar.slider(
            "k-anonymity value",
            min_value=2, max_value=20, value=3,
            key=f"{unique_key_prefix}_k_value",
            help="Minimum number of records in each equivalence class"
        )
        
        optimization_goal = st.sidebar.selectbox(
            "Optimization Goal",
            options=["minimal_loss", "maximal_utility", "balanced"],
            key=f"{unique_key_prefix}_optimization_goal",
            help="What to optimize for when selecting generalizations"
        )
        
        # Algorithm parameters
        st.sidebar.subheader("🔧 Algorithm Parameters")
        
        max_generalization_level = st.sidebar.slider(
            "Max Generalization Level",
            min_value=2, max_value=10, value=5,
            key=f"{unique_key_prefix}_max_generalization_level",
            help="Maximum depth of generalization hierarchy"
        )
        
        lattice_exploration = st.sidebar.selectbox(
            "Lattice Exploration",
            options=["breadth_first", "depth_first", "best_first"],
            key=f"{unique_key_prefix}_lattice_exploration",
            help="Strategy for exploring the generalization lattice"
        )
        
        use_pruning = st.sidebar.checkbox(
            "Use Pruning",
            value=True,
            key=f"{unique_key_prefix}_use_pruning",
            help="Use monotonicity-based pruning to reduce search space"
        )
        
        early_termination = st.sidebar.checkbox(
            "Early Termination",
            value=True,
            key=f"{unique_key_prefix}_early_termination",
            help="Stop search when first valid solution is found"
        )
        
        # Hierarchy parameters
        with st.sidebar.expander("📊 Generalization Hierarchies"):
            auto_build_hierarchies = st.sidebar.checkbox(
                "Auto-build Hierarchies",
                value=True,
                key=f"{unique_key_prefix}_auto_build_hierarchies",
                help="Automatically build generalization hierarchies"
            )
            
            if auto_build_hierarchies:
                hierarchy_depth = st.sidebar.slider(
                    "Hierarchy Depth",
                    min_value=2, max_value=8, value=4,
                    key=f"{unique_key_prefix}_hierarchy_depth",
                    help="Depth of auto-generated hierarchies"
                )
                
                categorical_grouping = st.sidebar.selectbox(
                    "Categorical Grouping",
                    options=["frequency", "semantic", "manual"],
                    key=f"{unique_key_prefix}_categorical_grouping",
                    help="Method for grouping categorical values"
                )
            else:
                hierarchy_depth = 4
                categorical_grouping = "frequency"
        
        # Performance settings
        with st.sidebar.expander("⚡ Performance Settings"):
            use_sampling = st.sidebar.checkbox(
                "Use Sampling",
                value=False,
                key=f"{unique_key_prefix}_use_sampling",
                help="Use sampling for large datasets"
            )
            
            if use_sampling:
                sample_size = st.sidebar.number_input(
                    "Sample Size",
                    min_value=100, max_value=10000, value=1000,
                    key=f"{unique_key_prefix}_sample_size",
                    help="Number of records to sample"
                )
            else:
                sample_size = 1000
            
            cache_evaluations = st.sidebar.checkbox(
                "Cache Evaluations",
                value=True,
                key=f"{unique_key_prefix}_cache_evaluations",
                help="Cache k-anonymity evaluations"
            )
        
        return {
            "k_value": k_value,
            "optimization_goal": optimization_goal,
            "max_generalization_level": max_generalization_level,
            "lattice_exploration": lattice_exploration,
            "use_pruning": use_pruning,
            "early_termination": early_termination,
            "auto_build_hierarchies": auto_build_hierarchies,
            "hierarchy_depth": hierarchy_depth,
            "categorical_grouping": categorical_grouping,
            "use_sampling": use_sampling,
            "sample_size": sample_size,
            "cache_evaluations": cache_evaluations
        }
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply Incognito k-anonymity algorithm to the input DataFrame.
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.k_value = parameters.get("k_value", 3)
            self.optimization_goal = parameters.get("optimization_goal", "minimal_loss")
            self.max_generalization_level = parameters.get("max_generalization_level", 5)
            self.lattice_exploration = parameters.get("lattice_exploration", "breadth_first")
            self.use_pruning = parameters.get("use_pruning", True)
            self.early_termination = parameters.get("early_termination", True)
            self.auto_build_hierarchies = parameters.get("auto_build_hierarchies", True)
            self.hierarchy_depth = parameters.get("hierarchy_depth", 4)
            self.categorical_grouping = parameters.get("categorical_grouping", "frequency")
            self.use_sampling = parameters.get("use_sampling", False)
            self.sample_size = parameters.get("sample_size", 1000)
            self.cache_evaluations = parameters.get("cache_evaluations", True)
            
            logger.info(f"Starting Incognito k-anonymity with k={self.k_value}")
            
            # Sample data if requested
            working_data = df_input
            if self.use_sampling and len(df_input) > self.sample_size:
                working_data = df_input.sample(n=self.sample_size, random_state=42)
                logger.info(f"Using sample of {len(working_data)} records")
            
            # Build generalization hierarchies
            if self.auto_build_hierarchies:
                self._build_generalization_hierarchies(working_data, sa_col)
            
            # Build generalization lattice
            self._build_generalization_lattice()
            
            # Search for optimal k-anonymous generalization
            self.optimal_generalization = self._search_optimal_generalization(working_data)
            
            if self.optimal_generalization is None:
                logger.warning("No k-anonymous generalization found")
                return df_input
            
            # Apply optimal generalization to full dataset
            anonymized_data = self._apply_generalization(df_input, self.optimal_generalization)
            
            processing_time = time.time() - start_time
            logger.info(f"Incognito anonymization completed in {processing_time:.2f}s")
            
            # Add metadata
            anonymized_data.attrs['incognito_k'] = self.k_value
            anonymized_data.attrs['generalization_levels'] = self.optimal_generalization
            anonymized_data.attrs['search_statistics'] = self.search_statistics
            anonymized_data.attrs['processing_time'] = processing_time
            
            return anonymized_data
            
        except Exception as e:
            logger.error(f"Error in Incognito anonymization: {str(e)}")
            st.error(f"Incognito anonymization failed: {str(e)}")
            return df_input
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "k_value": st.session_state.get(f"{unique_key_prefix}_k_value", 3),
            "optimization_goal": st.session_state.get(f"{unique_key_prefix}_optimization_goal", "minimal_loss"),
            "max_generalization_level": st.session_state.get(f"{unique_key_prefix}_max_generalization_level", 5),
            "lattice_exploration": st.session_state.get(f"{unique_key_prefix}_lattice_exploration", "breadth_first"),
            "use_pruning": st.session_state.get(f"{unique_key_prefix}_use_pruning", True),
            "early_termination": st.session_state.get(f"{unique_key_prefix}_early_termination", True),
            "auto_build_hierarchies": st.session_state.get(f"{unique_key_prefix}_auto_build_hierarchies", True),
            "hierarchy_depth": st.session_state.get(f"{unique_key_prefix}_hierarchy_depth", 4),
            "categorical_grouping": st.session_state.get(f"{unique_key_prefix}_categorical_grouping", "frequency"),
            "use_sampling": st.session_state.get(f"{unique_key_prefix}_use_sampling", False),
            "sample_size": st.session_state.get(f"{unique_key_prefix}_sample_size", 1000),
            "cache_evaluations": st.session_state.get(f"{unique_key_prefix}_cache_evaluations", True)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        for key, value in config_params.items():
            st.session_state[f"{unique_key_prefix}_{key}"] = value

    def _build_generalization_hierarchies(self, data: pd.DataFrame, sa_col: str | None):
        """Build generalization hierarchies for all attributes."""
        self.hierarchy_trees = {}
        
        for col in data.columns:
            if col == sa_col:
                continue  # Don't generalize sensitive attribute
            
            if data[col].dtype in ['object', 'category']:
                self.hierarchy_trees[col] = self._build_categorical_hierarchy(data[col])
            else:
                self.hierarchy_trees[col] = self._build_numerical_hierarchy(data[col])
    
    def _build_categorical_hierarchy(self, series: pd.Series) -> Dict[int, List[str]]:
        """Build hierarchy for categorical attribute."""
        hierarchy = {}
        unique_values = series.unique().tolist()
        
        # Level 0: Original values
        hierarchy[0] = unique_values
        
        if self.categorical_grouping == "frequency":
            # Group by frequency
            value_counts = series.value_counts()
            sorted_values = value_counts.index.tolist()
            
            for level in range(1, self.hierarchy_depth):
                group_size = max(2, len(sorted_values) // (2 ** level))
                groups = []
                
                for i in range(0, len(sorted_values), group_size):
                    group = sorted_values[i:i + group_size]
                    if len(group) == 1:
                        groups.append(group[0])
                    else:
                        groups.append(f"Group_{len(groups)+1}")
                
                hierarchy[level] = groups
                if len(groups) <= 1:
                    break
        
        # Top level: single value
        hierarchy[max(hierarchy.keys()) + 1] = ["*"]
        
        return hierarchy
    
    def _build_numerical_hierarchy(self, series: pd.Series) -> Dict[int, List[str]]:
        """Build hierarchy for numerical attribute."""
        hierarchy = {}
        min_val, max_val = series.min(), series.max()
        
        # Level 0: Original precision
        hierarchy[0] = series.unique().tolist()
        
        # Build interval-based hierarchy
        for level in range(1, self.hierarchy_depth):
            interval_count = max(2, self.numerical_intervals // level)
            interval_size = (max_val - min_val) / interval_count
            
            intervals = []
            for i in range(interval_count):
                start = min_val + i * interval_size
                end = min_val + (i + 1) * interval_size
                intervals.append(f"[{start:.2f}, {end:.2f}]")
            
            hierarchy[level] = intervals
            if interval_count <= 1:
                break
        
        # Top level: full range
        hierarchy[max(hierarchy.keys()) + 1] = [f"[{min_val:.2f}, {max_val:.2f}]"]
        
        return hierarchy
    
    def _build_generalization_lattice(self):
        """Build the lattice of all possible generalizations."""
        attributes = list(self.hierarchy_trees.keys())
        max_levels = [len(self.hierarchy_trees[attr]) - 1 for attr in attributes]
        
        # Generate all possible generalization combinations
        self.generalization_lattice = {}
        
        for levels in itertools.product(*[range(max_level + 1) for max_level in max_levels]):
            node_id = tuple(levels)
            self.generalization_lattice[node_id] = {
                'levels': levels,
                'attributes': attributes,
                'evaluated': False,
                'k_anonymous': None,
                'information_loss': float('inf')
            }
    
    def _search_optimal_generalization(self, data: pd.DataFrame) -> Optional[Tuple[int, ...]]:
        """Search for optimal k-anonymous generalization using the specified strategy."""
        self.search_statistics = {
            'nodes_evaluated': 0,
            'nodes_pruned': 0,
            'k_anonymous_found': 0
        }
        
        if self.lattice_exploration == "breadth_first":
            return self._breadth_first_search(data)
        elif self.lattice_exploration == "depth_first":
            return self._depth_first_search(data)
        else:  # best_first
            return self._best_first_search(data)
    
    def _breadth_first_search(self, data: pd.DataFrame) -> Optional[Tuple[int, ...]]:
        """Breadth-first search through the generalization lattice."""
        # Start from bottom level (most specific)
        queue = [node_id for node_id in self.generalization_lattice.keys() 
                if sum(node_id) == 0]  # All zeros = bottom level
        
        while queue:
            current_node = queue.pop(0)
            
            if self._is_pruned(current_node):
                self.search_statistics['nodes_pruned'] += 1
                continue
            
            # Evaluate k-anonymity
            if self._evaluate_k_anonymity(data, current_node):
                self.search_statistics['k_anonymous_found'] += 1
                self.k_anonymous_nodes.add(current_node)
                
                if self.early_termination:
                    return current_node
            
            # Add successors to queue
            successors = self._get_successors(current_node)
            queue.extend(successors)
        
        # Return best k-anonymous node if any found
        if self.k_anonymous_nodes:
            return min(self.k_anonymous_nodes, 
                      key=lambda node: self.generalization_lattice[node]['information_loss'])
        
        return None
    
    def _depth_first_search(self, data: pd.DataFrame) -> Optional[Tuple[int, ...]]:
        """Depth-first search through the generalization lattice."""
        def dfs(node_id):
            if self._is_pruned(node_id):
                self.search_statistics['nodes_pruned'] += 1
                return None
            
            if self._evaluate_k_anonymity(data, node_id):
                self.search_statistics['k_anonymous_found'] += 1
                self.k_anonymous_nodes.add(node_id)
                
                if self.early_termination:
                    return node_id
            
            # Explore successors
            for successor in self._get_successors(node_id):
                result = dfs(successor)
                if result is not None and self.early_termination:
                    return result
            
            return None
        
        # Start DFS from bottom
        start_node = tuple([0] * len(self.hierarchy_trees))
        result = dfs(start_node)
        
        if result:
            return result
        elif self.k_anonymous_nodes:
            return min(self.k_anonymous_nodes, 
                      key=lambda node: self.generalization_lattice[node]['information_loss'])
        
        return None
    
    def _best_first_search(self, data: pd.DataFrame) -> Optional[Tuple[int, ...]]:
        """Best-first search using information loss as heuristic."""
        import heapq
        
        # Priority queue: (information_loss, node_id)
        start_node = tuple([0] * len(self.hierarchy_trees))
        heap = [(0, start_node)]
        visited = set()
        
        while heap:
            _, current_node = heapq.heappop(heap)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            if self._is_pruned(current_node):
                self.search_statistics['nodes_pruned'] += 1
                continue
            
            if self._evaluate_k_anonymity(data, current_node):
                self.search_statistics['k_anonymous_found'] += 1
                return current_node
            
            # Add successors with their estimated information loss
            for successor in self._get_successors(current_node):
                if successor not in visited:
                    est_loss = self._estimate_information_loss(successor)
                    heapq.heappush(heap, (est_loss, successor))
        
        return None
    
    def _evaluate_k_anonymity(self, data: pd.DataFrame, node_id: Tuple[int, ...]) -> bool:
        """Evaluate if a generalization satisfies k-anonymity."""
        cache_key = node_id
        if self.cache_evaluations and cache_key in self.evaluation_cache:
            result = self.evaluation_cache[cache_key]
            self.generalization_lattice[node_id]['k_anonymous'] = result['k_anonymous']
            self.generalization_lattice[node_id]['information_loss'] = result['information_loss']
            return result['k_anonymous']
        
        self.search_statistics['nodes_evaluated'] += 1
        
        # Apply generalization and check k-anonymity
        generalized_data = self._apply_generalization_levels(data, node_id)
        
        # Group by generalized values
        groups = generalized_data.groupby(list(generalized_data.columns)).size()
        min_group_size = groups.min()
        
        is_k_anonymous = min_group_size >= self.k_value
        info_loss = self._calculate_information_loss(data, generalized_data, node_id)
        
        # Cache result
        if self.cache_evaluations:
            self.evaluation_cache[cache_key] = {
                'k_anonymous': is_k_anonymous,
                'information_loss': info_loss
            }
        
        # Update lattice node
        self.generalization_lattice[node_id]['evaluated'] = True
        self.generalization_lattice[node_id]['k_anonymous'] = is_k_anonymous
        self.generalization_lattice[node_id]['information_loss'] = info_loss
        
        return is_k_anonymous
    
    def _is_pruned(self, node_id: Tuple[int, ...]) -> bool:
        """Check if a node should be pruned based on monotonicity."""
        if not self.use_pruning:
            return False
        
        # Check if any predecessor is not k-anonymous
        for i, level in enumerate(node_id):
            if level > 0:
                predecessor = list(node_id)
                predecessor[i] -= 1
                predecessor_id = tuple(predecessor)
                
                if predecessor_id in self.generalization_lattice:
                    predecessor_node = self.generalization_lattice[predecessor_id]
                    if predecessor_node['evaluated'] and not predecessor_node['k_anonymous']:
                        return True
        
        return False
    
    def _get_successors(self, node_id: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get successor nodes in the lattice."""
        successors = []
        attributes = list(self.hierarchy_trees.keys())
        
        for i, level in enumerate(node_id):
            max_level = len(self.hierarchy_trees[attributes[i]]) - 1
            if level < max_level:
                successor = list(node_id)
                successor[i] += 1
                successor_id = tuple(successor)
                
                if successor_id in self.generalization_lattice:
                    successors.append(successor_id)
        
        return successors
    
    def _apply_generalization_levels(self, data: pd.DataFrame, levels: Tuple[int, ...]) -> pd.DataFrame:
        """Apply specific generalization levels to data."""
        result = data.copy()
        attributes = list(self.hierarchy_trees.keys())
        
        for i, (attr, level) in enumerate(zip(attributes, levels)):
            if attr in result.columns:
                hierarchy = self.hierarchy_trees[attr]
                if level in hierarchy:
                    # Apply generalization based on hierarchy level
                    if data[attr].dtype in ['object', 'category']:
                        result[attr] = self._generalize_categorical(data[attr], hierarchy, level)
                    else:
                        result[attr] = self._generalize_numerical(data[attr], hierarchy, level)
        
        return result
    
    def _generalize_categorical(self, series: pd.Series, hierarchy: Dict[int, List[str]], level: int) -> pd.Series:
        """Apply categorical generalization."""
        if level == 0:
            return series
        
        # Map original values to generalized groups
        generalized = series.copy()
        original_values = hierarchy[0]
        generalized_values = hierarchy[level]
        
        # Create mapping from original to generalized
        if len(generalized_values) == 1:
            generalized[:] = generalized_values[0]
        else:
            group_size = len(original_values) // len(generalized_values)
            for i, value in enumerate(original_values):
                group_index = min(i // group_size, len(generalized_values) - 1)
                generalized[generalized == value] = generalized_values[group_index]
        
        return generalized
    
    def _generalize_numerical(self, series: pd.Series, hierarchy: Dict[int, List[str]], level: int) -> pd.Series:
        """Apply numerical generalization."""
        if level == 0:
            return series
        
        generalized = series.copy()
        intervals = hierarchy[level]
        
        if len(intervals) == 1:
            generalized[:] = intervals[0]
        else:
            min_val, max_val = series.min(), series.max()
            interval_size = (max_val - min_val) / len(intervals)
            
            for i, value in enumerate(series):
                interval_index = min(int((value - min_val) / interval_size), len(intervals) - 1)
                generalized.iloc[i] = intervals[interval_index]
        
        return generalized
    
    def _calculate_information_loss(self, original: pd.DataFrame, generalized: pd.DataFrame, 
                                  levels: Tuple[int, ...]) -> float:
        """Calculate information loss for a generalization."""
        total_loss = 0.0
        attributes = list(self.hierarchy_trees.keys())
        
        for i, (attr, level) in enumerate(zip(attributes, levels)):
            if attr in original.columns:
                hierarchy = self.hierarchy_trees[attr]
                max_level = len(hierarchy) - 1
                
                # Normalized level as base loss
                level_loss = level / max_level if max_level > 0 else 0
                
                # Weight by attribute importance
                attribute_weight = 1.0  # Could be customized
                
                total_loss += level_loss * attribute_weight
        
        return total_loss / len(attributes) if attributes else 0.0
    
    def _estimate_information_loss(self, node_id: Tuple[int, ...]) -> float:
        """Estimate information loss for a node (for best-first search)."""
        attributes = list(self.hierarchy_trees.keys())
        total_loss = 0.0
        
        for i, level in enumerate(node_id):
            attr = attributes[i]
            max_level = len(self.hierarchy_trees[attr]) - 1
            normalized_level = level / max_level if max_level > 0 else 0
            total_loss += normalized_level
        
        return total_loss / len(attributes) if attributes else 0.0
    
    def _apply_generalization(self, data: pd.DataFrame, levels: Tuple[int, ...]) -> pd.DataFrame:
        """Apply the optimal generalization to the full dataset."""
        return self._apply_generalization_levels(data, levels)

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return IncognitoAnonymizationPlugin()
