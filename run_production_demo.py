"""
Production demo runner that works without LightGBM dependency issues.
This version uses the working components and provides full functionality.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def check_environment():
    """Check if environment is properly configured."""
    logger.info("Checking environment configuration...")
    
    # Check for Spotify credentials
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        logger.warning("Spotify credentials not found - will run in demo mode")
        print("\n⚠️  Spotify API credentials not configured!")
        print("\nRunning in DEMO MODE with sample data.")
        print("\nTo use with real Spotify data:")
        print("1. Run: python setup_production.py")
        print("2. Set environment variables:")
        print("   export SPOTIFY_CLIENT_ID=your_client_id")
        print("   export SPOTIFY_CLIENT_SECRET=your_client_secret")
        print("3. Run: python run_production.py")
        return False
    
    logger.info("✅ Spotify credentials found")
    return True

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path('.env')
    if env_file.exists():
        logger.info("Loading environment variables from .env file...")
        
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        
        logger.info("✅ Environment variables loaded")

def run_demo_app():
    """Run the demo application."""
    logger.info("Starting Playlist Auto-DJ demo application...")
    
    try:
        # Run the working demo app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "test_streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start application: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return True
    
    return True

def run_production_app():
    """Run the production application with real Spotify integration."""
    logger.info("Starting Playlist Auto-DJ production application...")
    
    try:
        # Run production app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "src/app/production_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start application: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return True
    
    return True

def print_startup_info(has_credentials=False):
    """Print startup information."""
    print("=" * 60)
    print("🎵 PLAYLIST AUTO-DJ")
    print("=" * 60)
    print()
    
    if has_credentials:
        print("🚀 PRODUCTION MODE - Real Spotify Integration")
        print()
        print("Features:")
        print("  ✅ Your actual Spotify library")
        print("  ✅ Real machine learning taste modeling")
        print("  ✅ Personalized recommendations")
        print("  ✅ Create playlists in your Spotify account")
    else:
        print("🎮 DEMO MODE - Sample Data")
        print()
        print("Features:")
        print("  ✅ Interactive web interface")
        print("  ✅ Mood-based recommendations")
        print("  ✅ Machine learning insights")
        print("  ✅ Playlist generation demo")
        print()
        print("💡 To use with your Spotify data, run setup first:")
        print("   python setup_production.py")
    
    print()
    print("The application will open in your browser automatically.")
    print("If it doesn't, go to: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the application.")
    print("=" * 60)
    print()

def main():
    """Main function."""
    # Load environment variables
    load_env_file()
    
    # Check for Spotify credentials
    has_credentials = check_environment()
    
    # Print startup info
    print_startup_info(has_credentials)
    
    # Create necessary directories
    required_dirs = ['data/cache', 'data/raw', 'data/processed', 'data/models', 'logs']
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Run appropriate app
    if has_credentials:
        print("🚀 Starting production app with Spotify integration...")
        success = run_production_app()
    else:
        print("🎮 Starting demo app with sample data...")
        success = run_demo_app()
    
    if success:
        print("\n✅ Application stopped successfully")
    else:
        print("\n❌ Application encountered an error")
        sys.exit(1)

if __name__ == "__main__":
    main()
