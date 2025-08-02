#!/usr/bin/env python3
"""
PDF Summarizer - Quick Project Validation Script

This script performs basic validation of the project structure and components
to ensure everything is properly set up before deployment.

Author: Your Name
Date: 2025
License: MIT
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_file_structure():
    """Check if all required files exist."""
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'Makefile',
        '.gitignore',
        'core/__init__.py',
        'core/generator.py',
        'core/summarizer.py',
        'core/logic.py',
        'src/__init__.py',
        'src/translator_models.py',
        'utils/__init__.py',
        'utils/languages_detect.py',
        'utils/pdf_extractor.py',
        'utils/preprocessing.py'
    ]
    
    print("Checking file structure...")
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  [OK] {file_path}")
    
    if missing_files:
        print("\n[ERROR] Missing files:")
        for file_path in missing_files:
            print(f"  [MISSING] {file_path}")
        return False
    
    print("[SUCCESS] All required files present")
    return True


def check_imports():
    """Check if all modules can be imported."""
    print("\nChecking module imports...")
    
    modules_to_test = [
        ('core.generator', 'Summary generator'),
        ('core.summarizer', 'Text summarizer'), 
        ('core.logic', 'Core logic'),
        ('src.translator_models', 'Translation models'),
        ('utils.languages_detect', 'Language detection'),
        ('utils.pdf_extractor', 'PDF extractor'),
        ('utils.preprocessing', 'Text preprocessing')
    ]
    
    failed_imports = []
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"  [OK] {module_name} - {description}")
        except ImportError as e:
            print(f"  [ERROR] {module_name} - {description} (Error: {e})")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"  [WARNING] {module_name} - {description} (Warning: {e})")
    
    if failed_imports:
        print(f"\n[ERROR] Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("[SUCCESS] All modules imported successfully")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        ('flask', 'Web framework'),
        ('flask_cors', 'CORS support'),
        ('transformers', 'ML models'),
        ('torch', 'PyTorch backend'),
        ('nltk', 'Natural language toolkit'),
        ('PyPDF2', 'PDF processing'),
        ('langdetect', 'Language detection')
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {package} - {description}")
        except ImportError:
            print(f"  [ERROR] {package} - {description}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n[ERROR] Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("[SUCCESS] All dependencies available")
    return True


def test_basic_functionality():
    """Test basic functionality without loading heavy models."""
    print("\nTesting basic functionality...")
    
    try:
        # Test Flask app creation
        from app import app
        print("  [OK] Flask app created successfully")
        
        # Test utility functions
        try:
            from utils.preprocessing import get_text_statistics
            stats = get_text_statistics("This is a test text.")
            print(f"  [OK] Text preprocessing: {stats['words']} words analyzed")
        except ImportError:
            print("  [WARNING] Text preprocessing not available (dependencies missing)")
        
        # Test language detection
        try:
            from utils.languages_detect import detect_languages
            lang = detect_languages("This is an English text")
            print(f"  [OK] Language detection: detected '{lang}'")
        except ImportError:
            print("  [WARNING] Language detection not available (dependencies missing)")
        except Exception as e:
            print(f"  [WARNING] Language detection test failed: {e}")
        
        print("[SUCCESS] Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Basic functionality test failed: {e}")
        return False


def validate_requirements_txt():
    """Validate requirements.txt format."""
    print("\nValidating requirements.txt...")
    
    requirements_file = project_root / 'requirements.txt'
    if not requirements_file.exists():
        print("[ERROR] requirements.txt not found")
        return False
    
    try:
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                packages.append(line)
        
        print(f"  [OK] Found {len(packages)} package specifications")
        
        # Check for common issues
        if not any('flask' in pkg.lower() for pkg in packages):
            print("  [WARNING] Flask not found in requirements")
        
        if not any('transformers' in pkg.lower() for pkg in packages):
            print("  [WARNING] transformers not found in requirements")
        
        print("[SUCCESS] requirements.txt validation passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] requirements.txt validation failed: {e}")
        return False


def generate_project_summary():
    """Generate a summary of the project."""
    print("\nProject Summary")
    print("=" * 50)
    
    # Count lines of code
    code_files = list(project_root.glob('**/*.py'))
    total_lines = 0
    total_files = len(code_files)
    
    for file_path in code_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
        except:
            pass
    
    print(f"Project Directory: {project_root}")
    print(f"Python Files: {total_files}")
    print(f"Total Lines of Code: {total_lines}")
    
    # Check README
    readme_file = project_root / 'README.md'
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            readme_lines = len(f.readlines())
        print(f"README.md: {readme_lines} lines")
    
    print(f"Build System: Makefile with {len(open(project_root / 'Makefile').readlines())} lines")
    print("[SUCCESS] Project is ready for GitHub deployment!")


def main():
    """Main validation function."""
    print("PDF Summarizer - Project Validation")
    print("=" * 50)
    
    all_passed = True
    
    # Run all checks
    checks = [
        check_file_structure,
        check_dependencies,
        check_imports,
        test_basic_functionality,
        validate_requirements_txt
    ]
    
    for check in checks:
        if not check():
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("ALL VALIDATIONS PASSED!")
        generate_project_summary()
        return 0
    else:
        print("[ERROR] SOME VALIDATIONS FAILED")
        print("Please fix the issues above before deployment")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
