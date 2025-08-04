# Session State Bridge Pattern Documentation

## Overview

The Session State Bridge pattern is an architectural design pattern implemented in this privacy-preserving machine learning system to enable loose coupling between the anonymization and ML experimentation modules. This pattern eliminates direct dependencies between modules while providing a robust communication mechanism through a centralized session state repository.

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│                     │    │                     │    │                     │
│  Anonymization      │    │   Session State     │    │  ML Experimentation │
│  Module (app.py)    │◄──►│   Bridge            │◄──►│  Module (ml_app.py) │
│                     │    │                     │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Key Components

### 1. Session State Bridge Structure

```python
st.session_state.datasets_collection = {
    'original': pd.DataFrame,           # Original dataset
    'anonymized_datasets': {            # Anonymized datasets by technique
        'technique_name': [dataset_entry1, dataset_entry2, ...]
    }
}

st.session_state.dataset_metadata = {   # Dataset metadata
    'dataset_id': {
        'technique': str,
        'sa_column': str,
        'original_shape': tuple,
        'anonymized_shape': tuple,
        'timestamp': str,
        'transfer_source': str
    }
}

st.session_state.ml_transfer_status = { # Transfer status tracking
    'success': bool,
    'timestamp': str,
    'technique': str,
    'original_rows': int,
    'anonymized_rows': int,
    'dataset_id': str
}

st.session_state.dataset_counters = {   # Dataset counters per technique
    'technique_name': int
}
```

### 2. Dataset Entry Format

Each anonymized dataset is stored as a structured entry:

```python
dataset_entry = {
    'dataframe': pd.DataFrame,          # The anonymized dataset
    'technique': str,                   # Anonymization technique used
    'sa_column': str,                   # Sensitive attribute column
    'timestamp': str,                   # Creation timestamp
    'dataset_id': str,                  # Unique identifier
    'source': str                       # Source module identifier
}
```

## Implementation Details

### 1. Data Transfer Function

```python
def transfer_data_to_ml_app(original_df, anonymized_df, technique_name, sa_col):
    """
    Transfer original and anonymized data to ML app through session state bridge.
    This implements the Session State Bridge pattern for loose coupling between modules.
    """
    try:
        # Initialize bridge structure
        if 'datasets_collection' not in st.session_state:
            st.session_state.datasets_collection = {
                'original': None,
                'anonymized_datasets': {}
            }
        
        # Store original dataset
        st.session_state.datasets_collection['original'] = original_df.copy()
        st.session_state.df_uploaded = original_df.copy()  # ML app compatibility
        
        # Create anonymized dataset entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Manage dataset counters
        if technique_name not in st.session_state.dataset_counters:
            st.session_state.dataset_counters[technique_name] = 0
        st.session_state.dataset_counters[technique_name] += 1
        counter = st.session_state.dataset_counters[technique_name]
        
        # Create structured dataset entry
        dataset_entry = {
            'dataframe': anonymized_df.copy(),
            'technique': technique_name,
            'sa_column': sa_col,
            'timestamp': timestamp,
            'dataset_id': f"{technique_name}_{counter}",
            'source': 'anonymization_app'
        }
        
        # Store in bridge
        if technique_name not in st.session_state.datasets_collection['anonymized_datasets']:
            st.session_state.datasets_collection['anonymized_datasets'][technique_name] = []
        st.session_state.datasets_collection['anonymized_datasets'][technique_name].append(dataset_entry)
        
        # Update metadata and status
        metadata_key = f"{technique_name}_{counter}"
        st.session_state.dataset_metadata[metadata_key] = {
            'technique': technique_name,
            'sa_column': sa_col,
            'original_shape': original_df.shape,
            'anonymized_shape': anonymized_df.shape,
            'timestamp': timestamp,
            'transfer_source': 'anonymization_app'
        }
        
        st.session_state.ml_transfer_status = {
            'success': True,
            'timestamp': timestamp,
            'technique': technique_name,
            'original_rows': len(original_df),
            'anonymized_rows': len(anonymized_df),
            'dataset_id': dataset_entry['dataset_id']
        }
        
        return True, f"Successfully transferred {technique_name} dataset to ML app"
        
    except Exception as e:
        st.session_state.ml_transfer_status = {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return False, f"Transfer failed: {str(e)}"
```

### 2. Status Display Function

```python
def get_transfer_status_display():
    """Get formatted transfer status for display"""
    if not st.session_state.ml_transfer_status:
        return None
    
    status = st.session_state.ml_transfer_status
    
    if status['success']:
        return {
            'type': 'success',
            'message': f"✅ **Transfer Complete** - {status['technique']} dataset transferred to ML app",
            'details': f"📊 Original: {status['original_rows']} rows → Anonymized: {status['anonymized_rows']} rows | 🕐 {status['timestamp']}"
        }
    else:
        return {
            'type': 'error',
            'message': f"❌ **Transfer Failed** - {status.get('error', 'Unknown error')}",
            'details': f"🕐 {status['timestamp']}"
        }
```

## Benefits

### 1. Loose Coupling
- Modules operate independently without direct dependencies
- Easy to modify or replace individual modules
- Reduced complexity in inter-module communication

### 2. Centralized Data Repository
- Single source of truth for shared data
- Consistent data access patterns
- Simplified data management

### 3. Extensibility
- Easy to add new modules to the system
- Simple to extend with additional data types
- Flexible architecture for future enhancements

### 4. Robust Communication
- Error handling at the bridge level
- Status tracking and monitoring
- Recovery mechanisms for failed transfers

## Usage Pattern

### 1. Anonymization App (app.py)
```python
# After successful anonymization
if st.button("🚀 Transfer to ML App"):
    success, message = transfer_data_to_ml_app(
        original_df=df_raw,
        anonymized_df=df_anonymized,
        technique_name=selected_technique,
        sa_col=sa_column
    )
    if success:
        st.success(message)
    else:
        st.error(message)
```

### 2. ML App (ml_app.py)
```python
# Access transferred data
if st.session_state.datasets_collection['original'] is not None:
    original_df = st.session_state.datasets_collection['original']
    
    # Access anonymized datasets
    for technique, datasets in st.session_state.datasets_collection['anonymized_datasets'].items():
        for dataset_entry in datasets:
            anonymized_df = dataset_entry['dataframe']
            technique_name = dataset_entry['technique']
            # Use in ML experiments
```

## Monitoring and Management

### 1. Transfer Status Tracking
- Real-time status updates
- Success/failure indicators
- Timestamp tracking
- Error message display

### 2. Dataset Management
- Dataset counters per technique
- Metadata tracking
- Clear/reset functionality
- Preview capabilities

### 3. UI Integration
- Sidebar status indicators
- Main area transfer buttons
- Progress indicators
- Status notifications

## Error Handling

### 1. Transfer Failures
- Graceful error handling
- Error message display
- Recovery mechanisms
- Status reset options

### 2. Data Validation
- DataFrame integrity checks
- Schema validation
- Size limit enforcement
- Type checking

### 3. Session State Management
- Initialization checks
- Cleanup procedures
- Memory management
- State persistence

## Best Practices

### 1. Data Copying
- Always copy DataFrames to prevent reference issues
- Use `.copy()` method for deep copies
- Avoid sharing mutable references

### 2. Unique Identifiers
- Generate unique dataset IDs
- Use technique name + counter pattern
- Include timestamps for tracking

### 3. Metadata Management
- Store comprehensive metadata
- Include source information
- Track transformation history

### 4. Error Recovery
- Implement graceful error handling
- Provide clear error messages
- Include recovery options

## Future Enhancements

### 1. Persistence
- Save session state to disk
- Load previous sessions
- Export/import capabilities

### 2. Multi-User Support
- User-specific session states
- Data isolation
- Access control

### 3. Advanced Monitoring
- Performance metrics
- Usage analytics
- System health monitoring

### 4. API Integration
- RESTful API endpoints
- External system integration
- Remote data access

## Conclusion

The Session State Bridge pattern provides a robust, scalable architecture for loose coupling between modules in privacy-preserving machine learning systems. It enables independent module development while maintaining seamless data communication through a centralized repository.

This pattern is particularly valuable in research environments where modules may evolve independently and new functionality needs to be added without disrupting existing components.
