import os
import importlib.util
import inspect
import re
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def scan_algorithm_plugins_for_metrics():
    """Scan all algorithm plugins and extract their internal metrics"""
    
    algorithms_dir = project_root / "src" / "ml_plugins" / "algorithms"
    found_metrics = {}
    
    print(f"🔍 Scanning directory: {algorithms_dir}")
    
    # Get all Python files ending with _plugin.py
    plugin_files = list(algorithms_dir.glob("*_plugin.py"))
    print(f"📁 Found {len(plugin_files)} plugin files")
    
    for py_file in plugin_files:
        print(f"\n📄 Scanning: {py_file.name}")
        
        try:
            # Read the file content
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for metric-related methods and return values
            plugin_metrics = {
                'score_methods': [],
                'analysis_methods': [],
                'metric_calculations': [],
                'result_dictionaries': []
            }
            
            # Patterns to find metric-related code
            patterns = {
                'score_methods': [
                    r'def\s+.*score.*?\(',
                    r'def\s+.*effectiveness.*?\(',
                    r'def\s+.*performance.*?\('
                ],
                'analysis_methods': [
                    r'def\s+.*analysis.*?\(',
                    r'def\s+.*analyze.*?\(',
                    r'def\s+get_.*_analysis.*?\('
                ],
                'metric_calculations': [
                    r'def\s+.*metric.*?\(',
                    r'def\s+calculate_.*?\(',
                    r'def\s+compute_.*?\('
                ],
                'result_dictionaries': [
                    r'return\s*\{[^}]*score[^}]*\}',
                    r'return\s*\{[^}]*metric[^}]*\}',
                    r'return\s*\{[^}]*analysis[^}]*\}'
                ]
            }
            
            # Search for each pattern
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        # Clean up the match
                        clean_match = match.replace('def ', '').split('(')[0].strip()
                        if clean_match and clean_match not in plugin_metrics[category]:
                            plugin_metrics[category].append(clean_match)
            
            # Only keep plugins that have metrics
            if any(plugin_metrics.values()):
                found_metrics[py_file.stem] = plugin_metrics
                print(f"✅ Found metrics in {py_file.name}")
            else:
                print(f"❌ No metrics found in {py_file.name}")
                
        except Exception as e:
            print(f"⚠️ Error scanning {py_file.name}: {e}")
    
    return found_metrics

def print_scan_results(metrics_dict):
    """Print the scan results in a readable format"""
    print("\n" + "="*60)
    print("📊 ALGORITHM PLUGIN METRICS SCAN RESULTS")
    print("="*60)
    
    for plugin_name, metric_categories in metrics_dict.items():
        print(f"\n🧠 {plugin_name.upper()}")
        print("-" * 40)
        
        for category, methods in metric_categories.items():
            if methods:
                print(f"  📈 {category.replace('_', ' ').title()}:")
                for method in methods:
                    print(f"    • {method}")
        
        if not any(metric_categories.values()):
            print("  ❌ No metrics found")

if __name__ == "__main__":
    print("🚀 Starting algorithm metrics scan...")
    metrics = scan_algorithm_plugins_for_metrics()
    print_scan_results(metrics)
    
    # Save results to file
    output_file = project_root / "scripts" / "found_algorithm_metrics.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")