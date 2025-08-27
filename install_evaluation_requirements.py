#!/usr/bin/env python3
"""
Script untuk menginstall dependensi yang diperlukan untuk evaluasi
"""

import subprocess
import sys

def install_package(package):
    """Install package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("Installing evaluation dependencies...")
    print("="*50)
    
    # Required packages for evaluation
    packages = [
        "sacrebleu",           # For BLEU score calculation
        "matplotlib",          # For plotting
        "seaborn",            # For better plots
        "pandas",             # For data handling
        "numpy",              # Numeric operations
        "torch",              # PyTorch (if not already installed)
        "tqdm",               # Progress bars
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"\nInstalling {package}...")
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "="*50)
    if failed_packages:
        print("❌ Some packages failed to install:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\nPlease install them manually using:")
        print(f"pip install {' '.join(failed_packages)}")
    else:
        print("✅ All packages installed successfully!")
        print("You can now run the evaluation scripts.")

if __name__ == "__main__":
    main()