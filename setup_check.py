"""
Setup & Configuration Helper
=============================
Helper script to verify environment, download models, and initialize system.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

def print_header(title):
    """Print formatted header."""
    print(f"\n{'=' * 80}")
    print(f"{title.center(80)}")
    print(f"{'=' * 80}\n")

def print_section(title):
    """Print section header."""
    print(f"{title}")
    print(f"{'-' * 80}")

def check_python_version():
    """Check Python version compatibility."""
    print_section("1. Checking Python Version")
    
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    print(f"Platform: {platform.platform()}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    
    print("✅ Python version OK")
    return True

def check_memory():
    """Check system RAM."""
    print_section("2. Checking System Memory")
    
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"Total RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 16:
            print(f"⚠️  16 GB RAM recommended (you have {memory_gb:.1f} GB)")
            print("   CPU inference will be slow. Consider GPU.")
            return True  # Still OK, just slow
        
        print("✅ Sufficient RAM")
        return True
    
    except ImportError:
        print("⚠️  Could not check RAM (psutil not installed)")
        print("   Proceeding anyway...")
        return True

def check_gpu():
    """Check GPU availability."""
    print_section("3. Checking GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("ℹ️  No GPU detected. Will use CPU (slower, but fine).")
            return True
    
    except ImportError:
        print("⚠️  PyTorch not installed yet. Will check later.")
        return True

def check_dependencies():
    """Check if dependencies are installed."""
    print_section("4. Checking Dependencies")
    
    required = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pydantic', 'pydantic')
    ]
    
    missing = []
    installed = []
    
    for display_name, import_name in required:
        try:
            __import__(import_name)
            installed.append(display_name)
            print(f"✅ {display_name}")
        except ImportError:
            missing.append(display_name)
            print(f"❌ {display_name}")
    
    if missing:
        print(f"\n⚠️  Missing: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print(f"\n✅ All {len(installed)} dependencies installed")
    return True

def download_model():
    """Pre-download transformer model."""
    print_section("5. Downloading Sentiment Model")
    
    print("This may take a few minutes on first run...")
    print("Downloading distilbert-base-uncased to cache...")
    
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        
        print("• Downloading tokenizer...")
        DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
        print("• Downloading model...")
        DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3
        )
        
        print("✅ Model downloaded to Hugging Face cache")
        return True
    
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("   You can download manually by running example_workflow.py")
        return False

def test_import():
    """Test importing core modules."""
    print_section("6. Testing Module Imports")
    
    modules = [
        'sentiment_analyzer',
        'aspect_extractor',
        'reputation_scorer',
        'anti_manipulation'
    ]
    
    all_ok = True
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            all_ok = False
    
    return all_ok

def run_quick_test():
    """Run quick sentiment analysis test."""
    print_section("7. Running Quick Sentiment Analysis Test")
    
    try:
        from sentiment_analyzer import SentimentAnalyzer
        
        print("Initializing sentiment analyzer...")
        analyzer = SentimentAnalyzer(device='cpu')
        
        test_text = "I love working here! Great team and culture."
        print(f"Test text: '{test_text}'")
        
        result = analyzer.predict(test_text)
        
        print(f"✅ Result:")
        print(f"   Label: {result['label']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Sentiment Signal: {result['sentiment_signal']:+.3f}")
        
        return True
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print_section("8. Creating Directories")
    
    dirs = [
        'data',
        'models',
        'logs',
        'cache'
    ]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ {dir_name}/")

def generate_summary(results):
    """Generate summary report."""
    print_header("SETUP SUMMARY")
    
    checks = [
        ("Python Version", results.get('python', False)),
        ("System Memory", results.get('memory', False)),
        ("GPU Support", results.get('gpu', False)),
        ("Dependencies", results.get('deps', False)),
        ("Model Download", results.get('model', False)),
        ("Module Imports", results.get('imports', False)),
        ("Quick Test", results.get('test', False)),
    ]
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    print(f"Results: {passed}/{total} checks passed\n")
    
    for name, ok in checks:
        status = "✅" if ok else "❌"
        print(f"{status} {name}")
    
    print(f"\n{'=' * 80}")
    
    if passed == total:
        print("✅ ✅ ✅  SETUP COMPLETE - YOU'RE READY TO GO!  ✅ ✅ ✅")
        print("\nNext steps:")
        print("1. Run: python example_workflow.py")
        print("2. Read: TECHNICAL_DESIGN.md")
        print("3. Start API: python sentiment_analysis_api.py")
        return True
    else:
        print(f"❌ Setup incomplete ({passed}/{total} checks passed)")
        print("\nFix the issues above and run this script again.")
        return False

def main():
    """Run all setup checks."""
    print_header("SETUP & CONFIGURATION CHECK")
    
    results = {}
    
    # Run checks
    results['python'] = check_python_version()
    results['memory'] = check_memory()
    results['gpu'] = check_gpu()
    results['deps'] = check_dependencies()
    
    if not results['deps']:
        print("\n⚠️  Please install dependencies first:")
        print("   pip install -r requirements.txt")
        print("\nThen run this script again.")
        return False
    
    results['model'] = download_model()
    results['imports'] = test_import()
    
    if results['imports']:
        results['test'] = run_quick_test()
    
    create_directories()
    
    # Generate summary
    return generate_summary(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
