"""
Core k-anonymity implementation using various generalization strategies.
This module provides the fundamental k-anonymity algorithm with multiple approaches.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class KAnonymityCore:
    """
    Core k-anonymity implementation with multiple generalization strategies.
    """
    
    def __init__(self, k: int, qi_columns: List[str], generalization_strategy: str = "optimal"):
        """
        Initialize k-anonymity core.
        
        Args:
            k: The k parameter for k-anonymity
            qi_columns: List of quasi-identifier columns
            generalization_strategy: Strategy for generalization ("optimal", "greedy", "binary")
        """
        self.k = k
        self.qi_columns = qi_columns
        self.generalization_strategy = generalization_strategy
        self.generalization_levels = {}
        self.original_ranges = {}
        
    def fit_generalization_hierarchies(self, df: pd.DataFrame) -> None:
        """
        Fit generalization hierarchies for each QI column.
        """
        print(f"DEBUG: Fitting generalization hierarchies for columns: {self.qi_columns}")
        for col in self.qi_columns:
            if col not in df.columns:
                print(f"DEBUG: Column {col} not found in dataframe")
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                self._fit_numeric_hierarchy(df, col)
                print(f"DEBUG: Created numeric hierarchy for {col}: {len(self.generalization_levels[col])} levels")
            else:
                self._fit_categorical_hierarchy(df, col)
                print(f"DEBUG: Created categorical hierarchy for {col}: {len(self.generalization_levels[col])} levels")
    
    def _fit_numeric_hierarchy(self, df: pd.DataFrame, col: str) -> None:
        """Fit generalization hierarchy for numeric columns."""
        values = df[col].dropna()
        if len(values) == 0:
            return
            
        min_val, max_val = values.min(), values.max()
        self.original_ranges[col] = (min_val, max_val)
        
        # Create generalization levels for numeric data
        levels = []
        
        # Level 0: Original values
        levels.append("original")
        
        # Level 1: Quantile-based ranges (quartiles)
        quantiles = values.quantile([0.25, 0.5, 0.75]).tolist()
        levels.append(f"quartiles_{quantiles}")
        
        # Level 2: Decile-based ranges
        deciles = values.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).tolist()
        levels.append(f"deciles_{deciles}")
        
        # Level 3: Min-Max range
        levels.append(f"range_{min_val}_{max_val}")
        
        # Level 4: Complete suppression
        levels.append("*")
        
        self.generalization_levels[col] = levels
    
    def _fit_categorical_hierarchy(self, df: pd.DataFrame, col: str) -> None:
        """Fit generalization hierarchy for categorical columns."""
        unique_values = df[col].dropna().unique().tolist()
        
        levels = []
        
        # Level 0: Original values
        levels.append("original")
        
        # Level 1: Group similar values (if possible)
        if len(unique_values) > 4:
            # Create groups based on frequency
            value_counts = df[col].value_counts()
            high_freq = value_counts[value_counts >= value_counts.median()].index.tolist()
            low_freq = value_counts[value_counts < value_counts.median()].index.tolist()
            levels.append(f"groups_high_{high_freq}_low_{low_freq}")
        
        # Level 2: Complete generalization
        levels.append("*")
        
        self.generalization_levels[col] = levels
    
    def anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply k-anonymity to the dataframe.
        
        Args:
            df: Input dataframe
              Returns:
            Anonymized dataframe
        """
        df_copy = df.copy()
        self.fit_generalization_hierarchies(df_copy)
        
        if self.generalization_strategy == "optimal":
            return self._optimal_anonymization(df_copy)
        elif self.generalization_strategy == "greedy":
            return self._greedy_anonymization(df_copy)
        elif self.generalization_strategy == "binary":
            return self._binary_search_anonymization(df_copy)
        else:
            return self._greedy_anonymization(df_copy)  # Default fallback
    
    def _optimal_anonymization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimal k-anonymity using dynamic programming approach.
        """
        # Debug: Check initial equivalence classes
        initial_classes = self._get_equivalence_class_sizes(df)
        print(f"DEBUG: Initial equivalence class sizes: min={min(initial_classes) if initial_classes else 0}, max={max(initial_classes) if initial_classes else 0}, count={len(initial_classes)}")
        
        # Check if already k-anonymous (this should be false for continuous data)
        if self._is_k_anonymous(df):
            print(f"DEBUG: Data is already {self.k}-anonymous - this is unexpected for continuous data!")
            return df
        
        print(f"DEBUG: Data is NOT {self.k}-anonymous, proceeding with generalization...")
          # Start with minimal generalization
        current_levels = {col: 0 for col in self.qi_columns}
        
        # Iteratively increase generalization until k-anonymous
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            generalized_df = self._apply_generalization(df, current_levels)
            
            # Debug: Check current state
            current_classes = self._get_equivalence_class_sizes(generalized_df)
            min_class_size = min(current_classes) if current_classes else 0
            print(f"DEBUG: Iteration {iteration}, levels={current_levels}, min_class_size={min_class_size}")
            
            if self._is_k_anonymous(generalized_df):
                print(f"DEBUG: K-anonymity achieved at iteration {iteration}")
                return generalized_df
            
            # Find the column that benefits most from generalization
            best_col = self._find_best_generalization_column(df, current_levels)
            
            if best_col and current_levels[best_col] < len(self.generalization_levels[best_col]) - 1:
                current_levels[best_col] += 1
                print(f"DEBUG: Increased {best_col} to level {current_levels[best_col]}")
            else:
                # Increase all columns if no single column helps
                increased = False
                for col in self.qi_columns:
                    if current_levels[col] < len(self.generalization_levels[col]) - 1:
                        current_levels[col] += 1
                        print(f"DEBUG: Increased {col} to level {current_levels[col]}")
                        increased = True
                        break
                
                if not increased:
                    print("DEBUG: No more generalization possible, breaking")
                    break
            
            iteration += 1
        
        # Final attempt with maximum generalization
        max_levels = {col: len(self.generalization_levels[col]) - 1 for col in self.qi_columns}
        return self._apply_generalization(df, max_levels)
    
    def _greedy_anonymization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Greedy k-anonymity approach.
        """
        current_levels = {col: 0 for col in self.qi_columns}
        
        while True:
            generalized_df = self._apply_generalization(df, current_levels)
            
            if self._is_k_anonymous(generalized_df):
                return generalized_df
            
            # Increase generalization level for all columns
            increased = False
            for col in self.qi_columns:
                if current_levels[col] < len(self.generalization_levels[col]) - 1:
                    current_levels[col] += 1
                    increased = True
            
            if not increased:
                break
        
        return generalized_df
    
    def _binary_search_anonymization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary search approach for k-anonymity.
        """
        # Try different combinations using binary search logic
        left, right = 0, sum(len(levels) - 1 for levels in self.generalization_levels.values())
        
        best_result = None
        
        while left <= right:
            mid = (left + right) // 2
            levels = self._distribute_generalization_budget(mid)
            generalized_df = self._apply_generalization(df, levels)
            
            if self._is_k_anonymous(generalized_df):
                best_result = generalized_df
                right = mid - 1  # Try with less generalization
            else:
                left = mid + 1   # Need more generalization
        
        return best_result if best_result is not None else df
    
    def _distribute_generalization_budget(self, budget: int) -> Dict[str, int]:
        """Distribute generalization budget across columns."""
        levels = {col: 0 for col in self.qi_columns}
        remaining_budget = budget
        
        # Distribute budget evenly first
        per_column = remaining_budget // len(self.qi_columns)
        for col in self.qi_columns:
            max_level = len(self.generalization_levels[col]) - 1
            levels[col] = min(per_column, max_level)
            remaining_budget -= levels[col]
        
        # Distribute remaining budget
        for col in self.qi_columns:
            if remaining_budget <= 0:
                break
            max_level = len(self.generalization_levels[col]) - 1
            if levels[col] < max_level:
                levels[col] += 1
                remaining_budget -= 1
        
        return levels
    
    def _find_best_generalization_column(self, df: pd.DataFrame, current_levels: Dict[str, int]) -> Optional[str]:
        """Find the column that would benefit most from additional generalization."""
        best_col = None
        best_improvement = 0
        
        for col in self.qi_columns:
            if current_levels[col] < len(self.generalization_levels[col]) - 1:
                # Test improvement if we generalize this column
                test_levels = current_levels.copy()
                test_levels[col] += 1
                
                test_df = self._apply_generalization(df, test_levels)
                improvement = self._calculate_anonymity_improvement(df, test_df)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_col = col
        
        return best_col
    
    def _calculate_anonymity_improvement(self, original_df: pd.DataFrame, generalized_df: pd.DataFrame) -> float:
        """Calculate improvement in anonymity."""
        original_groups = self._get_equivalence_class_sizes(original_df)
        generalized_groups = self._get_equivalence_class_sizes(generalized_df)
        
        # Count how many groups become k-anonymous
        improvement = 0
        for size in generalized_groups:
            if size >= self.k:
                improvement += size
        
        return improvement
    
    def _apply_generalization(self, df: pd.DataFrame, levels: Dict[str, int]) -> pd.DataFrame:
        """Apply generalization at specified levels."""
        result_df = df.copy()
        
        for col, level in levels.items():
            if col not in df.columns or level == 0:
                continue
            
            if level >= len(self.generalization_levels[col]):
                level = len(self.generalization_levels[col]) - 1
            
            if pd.api.types.is_numeric_dtype(df[col]):
                result_df = self._apply_numeric_generalization(result_df, col, level)
            else:
                result_df = self._apply_categorical_generalization(result_df, col, level)
        
        return result_df
    
    def _apply_numeric_generalization(self, df: pd.DataFrame, col: str, level: int) -> pd.DataFrame:
        """Apply numeric generalization."""
        if level == 0:
            return df
        
        df_copy = df.copy()
        
        if level == len(self.generalization_levels[col]) - 1:
            # Complete suppression
            df_copy[col] = "*"
        elif level == len(self.generalization_levels[col]) - 2:
            # Min-Max range
            min_val, max_val = self.original_ranges[col]
            df_copy[col] = f"[{min_val:.2f}-{max_val:.2f}]"
        else:
            # Quantile-based generalization
            if "quartiles" in self.generalization_levels[col][level]:
                df_copy = self._apply_quartile_generalization(df_copy, col)
            elif "deciles" in self.generalization_levels[col][level]:
                df_copy = self._apply_decile_generalization(df_copy, col)
        
        return df_copy
    
    def _apply_quartile_generalization(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply quartile-based generalization."""
        values = df[col].dropna()
        if len(values) == 0:
            return df
        
        df_copy = df.copy()
        q1, q2, q3 = values.quantile([0.25, 0.5, 0.75])
        
        def generalize_quartile(val):
            if pd.isna(val):
                return val
            if val <= q1:
                return f"Q1 [≤{q1:.2f}]"
            elif val <= q2:
                return f"Q2 [{q1:.2f}-{q2:.2f}]"
            elif val <= q3:
                return f"Q3 [{q2:.2f}-{q3:.2f}]"
            else:
                return f"Q4 [>{q3:.2f}]"
        
        df_copy[col] = df_copy[col].apply(generalize_quartile)
        return df_copy
    
    def _apply_decile_generalization(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply decile-based generalization."""
        values = df[col].dropna()
        if len(values) == 0:
            return df
        
        df_copy = df.copy()
        deciles = values.quantile([0.1 * i for i in range(1, 10)])
        
        def generalize_decile(val):
            if pd.isna(val):
                return val
            for i, decile in enumerate(deciles, 1):
                if val <= decile:
                    prev_decile = deciles.iloc[i-2] if i > 1 else values.min()
                    return f"D{i} [{prev_decile:.2f}-{decile:.2f}]"
            return f"D10 [>{deciles.iloc[-1]:.2f}]"
        
        df_copy[col] = df_copy[col].apply(generalize_decile)
        return df_copy
    
    def _apply_categorical_generalization(self, df: pd.DataFrame, col: str, level: int) -> pd.DataFrame:
        """Apply categorical generalization."""
        if level == 0:
            return df
        
        df_copy = df.copy()
        
        if level == len(self.generalization_levels[col]) - 1:
            # Complete suppression
            df_copy[col] = "*"
        else:
            # Apply grouping if available
            level_info = self.generalization_levels[col][level]
            if "groups_high" in level_info:
                # Parse group information and apply
                df_copy = self._apply_categorical_grouping(df_copy, col, level_info)
        
        return df_copy
    
    def _apply_categorical_grouping(self, df: pd.DataFrame, col: str, level_info: str) -> pd.DataFrame:
        """Apply categorical grouping."""
        # This is a simplified implementation
        # In practice, you'd parse the level_info and apply proper grouping
        value_counts = df[col].value_counts()
        median_count = value_counts.median()
        
        df_copy = df.copy()
        
        def generalize_category(val):
            if pd.isna(val):
                return val
            count = value_counts.get(val, 0)
            return "High_Frequency" if count >= median_count else "Low_Frequency"
        
        df_copy[col] = df_copy[col].apply(generalize_category)
        return df_copy
    
    def _is_k_anonymous(self, df: pd.DataFrame) -> bool:
        """Check if dataframe satisfies k-anonymity."""
        equivalence_classes = self._get_equivalence_class_sizes(df)
        return all(size >= self.k for size in equivalence_classes)
    
    def _get_equivalence_class_sizes(self, df: pd.DataFrame) -> List[int]:
        """Get sizes of equivalence classes."""
        if not self.qi_columns:
            return [len(df)]
        
        # Group by QI columns
        valid_qi_cols = [col for col in self.qi_columns if col in df.columns]
        if not valid_qi_cols:
            return [len(df)]
        
        groups = df.groupby(valid_qi_cols).size()
        return groups.tolist()
    
    def get_privacy_metrics(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy and utility metrics."""
        metrics = {}
        
        # K-anonymity metrics
        original_classes = self._get_equivalence_class_sizes(original_df)
        anonymized_classes = self._get_equivalence_class_sizes(anonymized_df)
        
        metrics['k_value'] = self.k
        metrics['original_equivalence_classes'] = len(original_classes)
        metrics['anonymized_equivalence_classes'] = len(anonymized_classes)
        metrics['min_group_size'] = min(anonymized_classes) if anonymized_classes else 0
        metrics['max_group_size'] = max(anonymized_classes) if anonymized_classes else 0
        metrics['avg_group_size'] = np.mean(anonymized_classes) if anonymized_classes else 0
        
        # Information loss
        metrics['information_loss'] = self._calculate_information_loss(original_df, anonymized_df)
        
        # Suppression ratio
        metrics['suppression_ratio'] = self._calculate_suppression_ratio(anonymized_df)
        
        return metrics
    
    def _calculate_information_loss(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame) -> float:
        """Calculate information loss as percentage of generalized values."""
        if len(original_df) == 0:
            return 0.0
        
        total_cells = 0
        generalized_cells = 0
        
        for col in self.qi_columns:
            if col in original_df.columns and col in anonymized_df.columns:
                total_cells += len(original_df)
                # Count cells that have been generalized (contain ranges, groups, or *)
                mask = anonymized_df[col].astype(str).str.contains(r'[\[\]\-\*Q]|High_|Low_', na=False)
                generalized_cells += mask.sum()
        
        return (generalized_cells / total_cells * 100) if total_cells > 0 else 0.0
    
    def _calculate_suppression_ratio(self, df: pd.DataFrame) -> float:
        """Calculate ratio of completely suppressed values."""
        if len(df) == 0:
            return 0.0
        
        total_cells = 0
        suppressed_cells = 0
        
        for col in self.qi_columns:
            if col in df.columns:
                total_cells += len(df)
                suppressed_cells += (df[col] == "*").sum()
        
        return (suppressed_cells / total_cells * 100) if total_cells > 0 else 0.0


def apply_k_anonymity(df: pd.DataFrame, k: int, qi_columns: List[str], 
                     generalization_strategy: str = "optimal") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply k-anonymity to a dataframe.
    
    Args:
        df: Input dataframe
        k: The k parameter for k-anonymity
        qi_columns: List of quasi-identifier columns
        generalization_strategy: Strategy for generalization
        
    Returns:
        Tuple of (anonymized_dataframe, metrics_dict)
    """
    anonymizer = KAnonymityCore(k, qi_columns, generalization_strategy)
    anonymized_df = anonymizer.anonymize(df)
    metrics = anonymizer.get_privacy_metrics(df, anonymized_df)
    
    return anonymized_df, metrics
