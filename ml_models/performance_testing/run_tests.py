"""
Performance Testing Suite Runner
===============================

Easy-to-use interface for running various performance tests on the anonymization platform.

Usage:
    python run_tests.py                    # Interactive menu
    python run_tests.py --quick            # Quick benchmark
    python run_tests.py --comprehensive    # Full performance test
    python run_tests.py --stress           # Stress testing
    python run_tests.py --compare          # Compare specific techniques

Author: Bachelor Thesis Project
Date: July 2025
"""

import os
import sys
import subprocess
from datetime import datetime

def print_banner():
    """Print testing suite banner."""
    print("=" * 60)
    print("🔬 ANONYMIZATION PLATFORM TESTING SUITE")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def show_menu():
    """Show interactive menu."""
    print("Available Testing Options:")
    print()
    print("1. 🚀 Quick Benchmark (2-5 minutes)")
    print("   - Fast performance check of all techniques")
    print("   - Choose dataset sizes (tiny/small/medium/large/huge)")
    print("   - Shows execution times and throughput")
    print("   - Option to save terminal output to file")
    print()
    print("2. 📊 Comprehensive Performance Test (10-30 minutes)")
    print("   - Detailed performance analysis with dataset size selection")
    print("   - Choose dataset sizes (tiny/small/medium/large/huge)")
    print("   - Multiple dataset types and memory usage tracking")
    print("   - Scalability analysis and visualization")
    print("   - Generates charts and detailed reports")
    print("   - Option to save terminal output to file")
    print()
    print("3. 💪 Stress Testing (5-15 minutes)")
    print("   - Edge case handling")
    print("   - Robustness testing")
    print("   - Data quality preservation analysis")
    print("   - Choose test level: Simple/Medium/Full")
    print("   - Option to save terminal output to file")
    print()
    print("4. ⚖️  Compare Specific Techniques")
    print("   - Side-by-side comparison")
    print("   - Choose which techniques to compare")
    print()
    print("5. 📋 List Available Techniques")
    print()
    print("6. ❌ Exit")
    print()

def run_quick_benchmark():
    """Run quick benchmark."""
    print("🚀 Starting Quick Benchmark...")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join(os.path.dirname(__file__), "quick_benchmark.py")
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print("\n✅ Quick benchmark completed successfully!")
        else:
            print("\n❌ Quick benchmark failed!")
            
    except Exception as e:
        print(f"❌ Error running quick benchmark: {e}")

def run_comprehensive_test():
    """Run comprehensive performance test."""
    print("📊 Starting Comprehensive Performance Test...")
    print("⚠️  This may take 10-30 minutes depending on your system.")
    print("-" * 50)
    
    confirm = input("Continue? (y/N): ").lower().strip()
    if confirm != 'y':
        print("Test cancelled.")
        return
    
    try:
        result = subprocess.run([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "performance_tester.py")
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print("\n✅ Comprehensive test completed successfully!")
            print("📁 Check the 'performance_results' directory for detailed reports and charts.")
        else:
            print("\n❌ Comprehensive test failed!")
            
    except Exception as e:
        print(f"❌ Error running comprehensive test: {e}")

def run_stress_test():
    """Run stress testing."""
    print("💪 Starting Stress Testing...")
    print("🧪 Testing edge cases and robustness...")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "stress_tester.py")
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print("\n✅ Stress testing completed successfully!")
        else:
            print("\n❌ Stress testing failed!")
            
    except Exception as e:
        print(f"❌ Error running stress test: {e}")

def compare_techniques():
    """Compare specific techniques."""
    print("⚖️  Technique Comparison")
    print("-" * 30)
    
    # First show available techniques
    try:
        subprocess.run([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "quick_benchmark.py"),
            "--list"
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
    except:
        print("❌ Could not list techniques")
        return
    
    print("\nEnter technique names to compare (separated by spaces):")
    print("Example: 'K-Anonymity Simple' 'Differential Privacy Core'")
    
    user_input = input("Techniques to compare: ").strip()
    if not user_input:
        print("No techniques selected.")
        return
    
    # Parse technique names (handle quoted names)
    import shlex
    try:
        technique_names = shlex.split(user_input)
    except:
        technique_names = user_input.split()
    
    if len(technique_names) < 2:
        print("❌ Please select at least 2 techniques to compare.")
        return
    
    print(f"🔄 Comparing {len(technique_names)} techniques...")
    
    try:
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "quick_benchmark.py")] + technique_names
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print("\n✅ Comparison completed successfully!")
        else:
            print("\n❌ Comparison failed!")
            
    except Exception as e:
        print(f"❌ Error running comparison: {e}")

def list_techniques():
    """List available techniques."""
    print("📋 Available Anonymization Techniques:")
    print("-" * 40)
    
    try:
        subprocess.run([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "quick_benchmark.py"),
            "--list"
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        print(f"❌ Error listing techniques: {e}")

def check_dependencies():
    """Check if required modules are available."""
    print("🔍 Checking dependencies...")
    
    required_modules = ['pandas', 'numpy']
    optional_modules = ['matplotlib', 'seaborn', 'psutil']
    
    missing_required = []
    missing_optional = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_required.append(module)
    
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)
    
    if missing_required:
        print(f"❌ Missing required modules: {', '.join(missing_required)}")
        print("Please install them with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional modules: {', '.join(missing_optional)}")
        print("Some features may be limited. Install with: pip install " + " ".join(missing_optional))
    
    print("✅ Dependencies check completed!")
    return True

def main():
    """Main function with command line interface."""
    print_banner()
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == "--quick":
            run_quick_benchmark()
        elif arg == "--comprehensive":
            run_comprehensive_test()
        elif arg == "--stress":
            run_stress_test()
        elif arg == "--compare":
            compare_techniques()
        elif arg == "--list":
            list_techniques()
        elif arg == "--help" or arg == "-h":
            print("Performance Testing Suite")
            print("Usage:")
            print("  python run_tests.py                    # Interactive menu")
            print("  python run_tests.py --quick            # Quick benchmark")
            print("  python run_tests.py --comprehensive    # Full performance test")
            print("  python run_tests.py --stress           # Stress testing")
            print("  python run_tests.py --compare          # Compare techniques")
            print("  python run_tests.py --list             # List techniques")
        else:
            print(f"❌ Unknown option: {arg}")
            print("Use --help for available options")
        
        return
    
    # Interactive menu
    while True:
        show_menu()
        
        try:
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                run_quick_benchmark()
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                run_comprehensive_test()
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                run_stress_test()
                input("\nPress Enter to continue...")
                
            elif choice == '4':
                compare_techniques()
                input("\nPress Enter to continue...")
                
            elif choice == '5':
                list_techniques()
                input("\nPress Enter to continue...")
                
            elif choice == '6':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-6.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
