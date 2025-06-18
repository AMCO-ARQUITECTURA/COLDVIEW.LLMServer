#!/usr/bin/env python3
"""
Verify that llama-cpp-python is properly installed with Metal support
"""
import sys

def check_installation():
    try:
        import llama_cpp
        print("✅ llama-cpp-python is installed")
        print(f"Version: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'Unknown'}")
        
        # Try to check for Metal support
        try:
            from llama_cpp import Llama
            print("✅ Llama class can be imported")
            
            # Check if we're on Apple Silicon
            import platform
            if platform.machine() == 'arm64' and platform.system() == 'Darwin':
                print("✅ Running on Apple Silicon")
                print("Metal acceleration should be available")
            else:
                print(f"ℹ️  Running on {platform.system()} {platform.machine()}")
            
        except Exception as e:
            print(f"❌ Error importing Llama class: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ llama-cpp-python is not installed: {e}")
        print("\nTo install with Metal support, run:")
        print("CMAKE_ARGS='-DLLAMA_METAL=on' python -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir")
        return False
    
    return True

def check_other_dependencies():
    dependencies = ['fastapi', 'uvicorn', 'pydantic']
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} is installed")
        except ImportError:
            print(f"❌ {dep} is not installed")
            print(f"Install with: python -m pip install {dep}")

if __name__ == "__main__":
    print("Checking COLDVIEW.LLMServer dependencies...\n")
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}\n")
    
    if check_installation():
        print("\n✅ llama-cpp-python installation verified!")
    else:
        print("\n❌ llama-cpp-python installation failed!")
        sys.exit(1)
    
    print("\nChecking other dependencies:")
    check_other_dependencies()
    
    print("\n🎉 All checks completed!")