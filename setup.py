#!/usr/bin/env python3
"""
Setup script for Contact Center AI
"""
import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    return True

def main():
    print("🚀 Setting up Contact Center AI...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("# Contact Center AI Configuration\n")
            f.write("OPENAI_API_KEY=\n")
            f.write("DATABASE_URL=sqlite:///./contact_center.db\n")
            f.write("LOG_LEVEL=INFO\n")
        print("✅ .env file created (add your OpenAI API key if available)")
    
    # Generate sample data
    if not run_command("python scripts/generate_data.py", "Generating sample data"):
        print("⚠️  Sample data generation failed, but you can run it manually later")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your OpenAI API key to .env (optional)")
    print("2. Start API: python run_api.py")
    print("3. Start Dashboard: python run_dashboard.py")
    print("4. Visit http://localhost:8000/docs for API documentation")
    print("5. Visit http://localhost:8501 for the dashboard")

if __name__ == "__main__":
    main()