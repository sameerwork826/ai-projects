#!/usr/bin/env python3
"""
Setup script for Knowledge Distillation Showcase

This script helps set up the environment and run the application.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False


def create_dummy_models():
    """Create dummy models for demonstration"""
    print("Creating dummy models...")
    try:
        from extract_models import create_dummy_models
        create_dummy_models()
        return True
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        return False


def run_streamlit_app():
    """Run the Streamlit application"""
    print("Starting Streamlit application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")


def main():
    """Main setup function"""
    print("🧠 Knowledge Distillation Showcase Setup")
    print("=" * 40)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Create models directory and dummy models
    if not os.path.exists("models"):
        if not create_dummy_models():
            return
    
    print("\n🎉 Setup complete!")
    print("\nYou can now:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Train models: python train_models.py")
    print("3. Open the notebook: jupyter notebook Knowledge-Distillation-Notebook.ipynb")
    
    # Ask if user wants to run the app
    choice = input("\nWould you like to run the Streamlit app now? (y/n): ").lower()
    if choice in ['y', 'yes']:
        run_streamlit_app()


if __name__ == "__main__":
    main()
