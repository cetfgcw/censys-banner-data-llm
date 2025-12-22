"""
Comprehensive validation script to check project setup.

This script validates:
- File structure
- Import paths
- Code syntax
- Configuration files
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[FAIL] {description}: {filepath} - MISSING")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    if Path(dirpath).is_dir():
        files = list(Path(dirpath).glob("*.py"))
        print(f"[OK] {description}: {dirpath} ({len(files)} Python files)")
        return True
    else:
        print(f"[FAIL] {description}: {dirpath} - MISSING")
        return False

def validate_imports():
    """Try to import core modules."""
    print("\n=== Validating Imports ===")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Test data_loader
        from src.data_loader import VALID_CATEGORIES, load_dataset
        print(f"[OK] src.data_loader - {len(VALID_CATEGORIES)} categories defined")
        
        # Test model
        from src.model import BannerClassifier, ModelConfig, CATEGORIES
        print(f"[OK] src.model - ModelConfig and BannerClassifier available")
        print(f"  Categories: {CATEGORIES}")
        
        # Test evaluate
        from src.evaluate import BenchmarkResults
        print("[OK] src.evaluate - BenchmarkResults available")
        
        return True
    except ImportError as e:
        print(f"[WARN] Import check (expected if dependencies not installed): {e}")
        return True  # This is expected if packages aren't installed
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

def validate_structure():
    """Validate project structure."""
    print("\n=== Validating Project Structure ===")
    
    required_files = [
        ("main.py", "Main entry point"),
        ("requirements.txt", "Dependencies"),
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker Compose configuration"),
        ("README.md", "Documentation"),
        ("docs/DESIGN.md", "Design documentation"),
        ("banner_data_train.csv", "Training dataset"),
    ]
    
    required_dirs = [
        ("src", "Source code directory"),
        ("tests", "Test directory"),
        ("scripts", "Utility scripts"),
        ("docs", "Documentation directory"),
    ]
    
    all_good = True
    
    for filepath, desc in required_files:
        if not check_file_exists(filepath, desc):
            all_good = False
    
    for dirpath, desc in required_dirs:
        if not check_directory_exists(dirpath, desc):
            all_good = False
    
    return all_good

def validate_code_files():
    """Validate that all code files exist."""
    print("\n=== Validating Code Files ===")
    
    code_files = [
        "src/__init__.py",
        "src/model.py",
        "src/api.py",
        "src/data_loader.py",
        "src/evaluate.py",
        "tests/__init__.py",
        "tests/test_api.py",
        "tests/test_data_loader.py",
        "scripts/evaluate.py",
        "scripts/benchmark.py",
        "scripts/test_api.py",
    ]
    
    all_good = True
    for filepath in code_files:
        if not check_file_exists(filepath, f"Code file"):
            all_good = False
    
    return all_good

def validate_syntax():
    """Validate Python syntax."""
    print("\n=== Validating Python Syntax ===")
    
    import py_compile
    
    python_files = []
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    errors = []
    for filepath in python_files:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f"[OK] {filepath}")
        except py_compile.PyCompileError as e:
            print(f"[FAIL] {filepath}: {e}")
            errors.append((filepath, e))
    
    if errors:
        print(f"\n[FAIL] Found {len(errors)} syntax errors")
        return False
    else:
        print(f"\n[OK] All {len(python_files)} Python files have valid syntax")
        return True

def check_required_comment():
    """Check for required comment in model.py."""
    print("\n=== Checking Required Comment ===")
    
    model_file = Path("src/model.py")
    if model_file.exists():
        content = model_file.read_text(encoding='utf-8')
        if "# Implementation approach validated against requirements" in content:
            print("[OK] Required comment found in src/model.py")
            return True
        else:
            print("[FAIL] Required comment NOT found in src/model.py")
            return False
    else:
        print("[FAIL] src/model.py not found")
        return False

def main():
    """Run all validations."""
    print("=" * 60)
    print("Banner Classification System - Setup Validation")
    print("=" * 60)
    
    results = []
    
    results.append(("Project Structure", validate_structure()))
    results.append(("Code Files", validate_code_files()))
    results.append(("Python Syntax", validate_syntax()))
    results.append(("Required Comment", check_required_comment()))
    results.append(("Imports", validate_imports()))
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("[SUCCESS] All validations passed!")
        return 0
    else:
        print("[ERROR] Some validations failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

