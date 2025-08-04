#!/usr/bin/env python3
"""
Session State Bridge Demo Script
================================

This script demonstrates the Session State Bridge pattern implementation
for loose coupling between the anonymization and ML experimentation modules.

Usage:
    python session_state_bridge_demo.py

This will show how to:
1. Run the anonymization app
2. Transfer data using the Session State Bridge
3. Access the data in the ML app

The Session State Bridge pattern allows modules to communicate through
a centralized session state repository without direct dependencies.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n🔹 Step {step_num}: {description}")

def main():
    """Main demonstration function"""
    print_header("Session State Bridge Pattern Demo")
    
    print("""
The Session State Bridge pattern provides loose coupling between modules:

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Anonymization   │    │ Session State   │    │ ML              │
│ App (app.py)    │◄──►│ Bridge          │◄──►│ App (ml_app.py) │
└─────────────────┘    └─────────────────┘    └─────────────────┘

Key Benefits:
✅ Loose coupling - modules operate independently
✅ Centralized data repository
✅ Extensible architecture
✅ Robust communication
""")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    app_py_path = current_dir / "app.py"
    ml_app_py_path = current_dir / "ml_app.py"
    
    if not app_py_path.exists():
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the ml_models/core directory")
        return
    
    if not ml_app_py_path.exists():
        print("❌ Error: ml_app.py not found in current directory")
        print("Please run this script from the ml_models/core directory")
        return
    
    print_step(1, "Session State Bridge Architecture")
    print("""
    The bridge uses these key session state variables:
    
    • st.session_state.datasets_collection['original'] - Original dataset
    • st.session_state.datasets_collection['anonymized_datasets'] - Anonymized datasets
    • st.session_state.dataset_metadata - Dataset metadata
    • st.session_state.ml_transfer_status - Transfer status
    • st.session_state.dataset_counters - Dataset counters per technique
    """)
    
    print_step(2, "Data Transfer Flow")
    print("""
    1. Load dataset in anonymization app
    2. Apply anonymization technique
    3. Click "🚀 Transfer to ML App" button
    4. Data flows through session state bridge
    5. Access data in ML app's "📂 Anonymized Data" tab
    """)
    
    print_step(3, "Running the Applications")
    print("""
    To test the Session State Bridge pattern:
    
    Terminal 1 (Anonymization App):
    cd ml_models/core
    streamlit run app.py
    
    Terminal 2 (ML App):
    cd ml_models/core
    streamlit run ml_app.py
    """)
    
    print_step(4, "Testing the Bridge")
    print("""
    In the Anonymization App:
    1. Upload a dataset or use sample data
    2. Select an anonymization technique
    3. Configure parameters
    4. Click "Anonymize" button
    5. Click "🚀 Transfer to ML App" button
    6. Check transfer status in sidebar
    
    In the ML App:
    1. Go to "📂 Anonymized Data" tab
    2. See transferred datasets
    3. Run ML experiments on anonymized data
    4. Compare original vs anonymized performance
    """)
    
    print_step(5, "Monitoring the Bridge")
    print("""
    Session state monitoring features:
    
    • Transfer status indicators
    • Dataset counters and metadata
    • Clear buttons for data management
    • Real-time updates across modules
    • Error handling and recovery
    """)
    
    print_header("Demo Complete")
    print("""
The Session State Bridge pattern successfully decouples the anonymization
and ML modules while providing robust data communication.

Next Steps:
1. Run both applications in separate terminals
2. Test the data transfer functionality
3. Observe the loose coupling benefits
4. Extend with additional modules as needed
""")

if __name__ == "__main__":
    main()
