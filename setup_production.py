"""
Production setup script for Playlist Auto-DJ.
Helps configure Spotify API credentials and install dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path
import webbrowser

def print_header():
    """Print setup header."""
    print("=" * 60)
    print("🎵 PLAYLIST AUTO-DJ - PRODUCTION SETUP")
    print("=" * 60)
    print()

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_spotify_credentials():
    """Guide user through Spotify API setup."""
    print("\n🎵 Setting up Spotify API credentials...")
    print()
    
    # Check if credentials already exist
    if os.getenv('SPOTIFY_CLIENT_ID') and os.getenv('SPOTIFY_CLIENT_SECRET'):
        print("✅ Spotify credentials found in environment variables!")
        return True
    
    print("To use the full version, you need Spotify API credentials:")
    print()
    print("1. Go to https://developer.spotify.com/dashboard")
    print("2. Log in with your Spotify account")
    print("3. Click 'Create App'")
    print("4. Fill in the app details:")
    print("   - App Name: Playlist Auto-DJ")
    print("   - App Description: Personal music recommendation system")
    print("   - Redirect URI: http://localhost:8080/callback")
    print("5. Copy your Client ID and Client Secret")
    print()
    
    # Ask if user wants to open the browser
    open_browser = input("Would you like to open the Spotify Developer Dashboard? (y/n): ").lower().strip()
    if open_browser == 'y':
        webbrowser.open('https://developer.spotify.com/dashboard')
    
    print()
    print("After creating your Spotify app, enter your credentials:")
    
    client_id = input("Enter your Spotify Client ID: ").strip()
    client_secret = input("Enter your Spotify Client Secret: ").strip()
    
    if not client_id or not client_secret:
        print("❌ Both Client ID and Client Secret are required!")
        return False
    
    # Create .env file
    env_content = f"""# Spotify API Credentials
SPOTIFY_CLIENT_ID={client_id}
SPOTIFY_CLIENT_SECRET={client_secret}

# Optional: Set custom redirect URI (default: http://localhost:8080/callback)
# SPOTIFY_REDIRECT_URI=http://localhost:8080/callback
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Credentials saved to .env file!")
    print()
    print("⚠️  IMPORTANT: Add the following to your shell profile (.bashrc, .zshrc, etc.):")
    print(f"export SPOTIFY_CLIENT_ID={client_id}")
    print(f"export SPOTIFY_CLIENT_SECRET={client_secret}")
    print()
    print("Or run these commands in your current terminal:")
    print(f"export SPOTIFY_CLIENT_ID={client_id}")
    print(f"export SPOTIFY_CLIENT_SECRET={client_secret}")
    
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        'data/cache',
        'data/raw',
        'data/processed',
        'data/models',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created!")

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import pandas
        import numpy
        import sklearn
        import streamlit
        import spotipy
        import plotly
        
        print("✅ All core packages imported successfully!")
        
        # Test LightGBM (optional)
        try:
            import lightgbm
            print("✅ LightGBM available (advanced ML features enabled)")
        except ImportError:
            print("⚠️  LightGBM not available (will use Random Forest fallback)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🚀 SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print()
    print("1. Make sure your environment variables are set:")
    print("   export SPOTIFY_CLIENT_ID=your_client_id")
    print("   export SPOTIFY_CLIENT_SECRET=your_client_secret")
    print()
    print("2. Run the production application:")
    print("   streamlit run src/app/production_app.py")
    print()
    print("3. Or use the quick start script:")
    print("   python run_production.py")
    print()
    print("🎵 Enjoy your personalized music recommendations!")
    print()

def main():
    """Main setup function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    create_directories()
    
    # Setup Spotify credentials
    if not setup_spotify_credentials():
        return False
    
    # Test installation
    if not test_installation():
        return False
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

