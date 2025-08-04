"""
Test script to verify ML plugin loading works correctly
"""

import os
import sys

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("Testing ML plugin loading...")

try:
    from src.ml_plugins.plugin_manager import get_plugin_manager
    print("✅ Plugin manager imported successfully")
    
    plugin_manager = get_plugin_manager()
    print("✅ Plugin manager instance created")
    
    # Test the correct method
    classification_plugins = plugin_manager.get_available_plugins("classification")
    regression_plugins = plugin_manager.get_available_plugins("regression")
    
    print(f"📊 Classification plugins: {len(classification_plugins)}")
    print(f"📈 Regression plugins: {len(regression_plugins)}")
    
    # Test a few plugins
    if classification_plugins:
        first_plugin_name, first_plugin = next(iter(classification_plugins.items()))
        print(f"🔧 Testing plugin: {first_plugin_name}")
        print(f"   Type: {type(first_plugin)}")
        print(f"   Has fit method: {hasattr(first_plugin, 'fit')}")
        
        # Test create_model_instance if available
        if hasattr(first_plugin, 'create_model_instance'):
            model = first_plugin.create_model_instance({})
            print(f"   Model type: {type(model)}")
            print(f"   Model has fit: {hasattr(model, 'fit')}")
        else:
            print(f"   Plugin IS the model (no create_model_instance)")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
