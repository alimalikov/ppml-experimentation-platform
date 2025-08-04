import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import io

def main():
    st.set_page_config(page_title="Dataset Preprocessor", layout="wide")
    
    st.title("🧹 Dataset Preprocessor - Missing Values Handler")
    st.markdown("Upload your dataset and handle missing values with various imputation strategies.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Dataset loaded successfully!")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Display basic info about the dataset
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.subheader("🔍 Missing Values Analysis")
                
                # Calculate missing values
                missing_info = df.isnull().sum()
                missing_percentage = (missing_info / len(df)) * 100
                
                missing_df = pd.DataFrame({
                    'Column': missing_info.index,
                    'Missing Count': missing_info.values,
                    'Missing %': missing_percentage.values
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                
                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True)
                    st.warning(f"Found missing values in {len(missing_df)} columns")
                else:
                    st.success("🎉 No missing values found!")
            
            # Missing values handling section
            if len(missing_df) > 0:
                st.markdown("---")
                st.subheader("🛠️ Missing Values Handling")
                
                # Select columns with missing values
                columns_with_missing = missing_df['Column'].tolist()
                
                # Strategy selection
                strategy_options = {
                    'mean': 'Mean (for numerical columns)',
                    'median': 'Median (for numerical columns)', 
                    'most_frequent': 'Most Frequent (for categorical columns)',
                    'constant': 'Constant Value (specify below)',
                    'drop_rows': 'Drop Rows with Missing Values',
                    'drop_columns': 'Drop Columns with Missing Values'
                }
                
                selected_strategy = st.selectbox(
                    "Choose imputation strategy:",
                    options=list(strategy_options.keys()),
                    format_func=lambda x: strategy_options[x],
                    help="Select how you want to handle missing values"
                )
                
                # Additional options based on strategy
                fill_value = None
                if selected_strategy == 'constant':
                    fill_value = st.text_input(
                        "Enter constant value:",
                        value="0",
                        help="Value to use for filling missing entries"
                    )
                
                # Threshold for dropping columns
                if selected_strategy == 'drop_columns':
                    threshold = st.slider(
                        "Drop columns with missing percentage above:",
                        min_value=0,
                        max_value=100,
                        value=50,
                        help="Columns with missing values above this percentage will be dropped"
                    )
                
                # Process button
                if st.button("🚀 Process Dataset", type="primary"):
                    processed_df = process_missing_values(
                        df.copy(), 
                        selected_strategy, 
                        fill_value, 
                        threshold if selected_strategy == 'drop_columns' else None
                    )
                    
                    if processed_df is not None:
                        st.success("✅ Dataset processed successfully!")
                        
                        # Show results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📈 Before Processing")
                            st.write(f"Shape: {df.shape}")
                            st.write(f"Missing values: {df.isnull().sum().sum()}")
                        
                        with col2:
                            st.subheader("📉 After Processing")
                            st.write(f"Shape: {processed_df.shape}")
                            st.write(f"Missing values: {processed_df.isnull().sum().sum()}")
                        
                        # Display processed dataset
                        st.subheader("🎯 Processed Dataset Preview")
                        st.dataframe(processed_df.head(10), use_container_width=True)
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        processed_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Processed Dataset",
                            data=csv_data,
                            file_name=f"processed_{uploaded_file.name}",
                            mime="text/csv",
                            help="Download the processed dataset as CSV"
                        )
                        
                        # Store in session state for further analysis
                        st.session_state['processed_df'] = processed_df
                        st.session_state['original_df'] = df
                        
                        # Add target column analysis
                        st.markdown("---")
                        st.subheader("🎯 Target Column Analysis for ML")
                        analyze_target_columns(processed_df)
            
            else:
                st.info("No missing values found. Your dataset is ready to use!")
                
                # Still provide download option for consistency
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Dataset",
                    data=csv_data,
                    file_name=f"clean_{uploaded_file.name}",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            st.write("Please make sure your file is a valid CSV format.")
    
    else:
        st.info("👆 Please upload a CSV file to get started.")
        
        # Show example of supported datasets
        st.markdown("### 📋 Supported Datasets")
        st.markdown("""
        This preprocessor works with:
        - **Iris Dataset** (small dataset)
        - **Breast Cancer Dataset** (medium dataset) 
        - **Handwritten Digits Dataset** (large dataset)
        - **Adult Income Dataset** (very large dataset)
        - Any CSV file with missing values
        """)

def process_missing_values(df, strategy, fill_value=None, threshold=None):
    """
    Process missing values in the dataset based on selected strategy
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Imputation strategy
    fill_value : str, optional
        Value for constant strategy
    threshold : int, optional
        Threshold for dropping columns
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe
    """
    try:
        if strategy == 'drop_rows':
            # Drop rows with any missing values
            initial_rows = len(df)
            df_processed = df.dropna()
            rows_dropped = initial_rows - len(df_processed)
            st.info(f"Dropped {rows_dropped} rows with missing values")
            
        elif strategy == 'drop_columns':
            # Drop columns based on missing percentage threshold
            missing_percentages = (df.isnull().sum() / len(df)) * 100
            columns_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
            
            if columns_to_drop:
                df_processed = df.drop(columns=columns_to_drop)
                st.info(f"Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
            else:
                df_processed = df.copy()
                st.info("No columns exceeded the threshold for dropping")
        
        else:
            # Use sklearn SimpleImputer
            df_processed = df.copy()
            
            # Separate numerical and categorical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if strategy in ['mean', 'median']:
                # Only apply to numerical columns
                if numerical_cols:
                    imputer = SimpleImputer(strategy=strategy)
                    df_processed[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                    st.info(f"Applied {strategy} imputation to {len(numerical_cols)} numerical columns")
                
                # Handle categorical columns with most_frequent if they have missing values
                if categorical_cols:
                    cat_with_missing = [col for col in categorical_cols if df[col].isnull().any()]
                    if cat_with_missing:
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        df_processed[cat_with_missing] = cat_imputer.fit_transform(df[cat_with_missing])
                        st.info(f"Applied most_frequent imputation to {len(cat_with_missing)} categorical columns")
            
            elif strategy == 'most_frequent':
                # Apply to all columns with missing values
                cols_with_missing = df.columns[df.isnull().any()].tolist()
                if cols_with_missing:
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_processed[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])
                    st.info(f"Applied most_frequent imputation to {len(cols_with_missing)} columns")
            
            elif strategy == 'constant':
                # Apply constant value to all missing entries
                if fill_value is not None:
                    # Try to convert fill_value to appropriate type
                    try:
                        # For numerical columns, try to convert to float
                        for col in numerical_cols:
                            if df[col].isnull().any():
                                df_processed[col] = df[col].fillna(float(fill_value))
                        
                        # For categorical columns, use string value
                        for col in categorical_cols:
                            if df[col].isnull().any():
                                df_processed[col] = df[col].fillna(str(fill_value))
                        
                        st.info(f"Filled missing values with constant: {fill_value}")
                    except ValueError:
                        # If conversion fails, use string for all
                        df_processed = df.fillna(str(fill_value))
                        st.info(f"Filled missing values with constant (as string): {fill_value}")
        
        return df_processed
        
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        return None

def show_dataset_info(df):
    """Display basic information about the dataset"""
    st.markdown("### 📋 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    
    with col2:
        st.metric("Total Columns", df.shape[1])
    
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data types
    st.markdown("### 🏷️ Column Data Types")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(dtype_df, use_container_width=True)

if __name__ == "__main__":
    main()
