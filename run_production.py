"""
Production runner script for Playlist Auto-DJ.
Handles environment setup and launches the production application.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging
import socket, subprocess, sys, shutil



def find_free_port(candidates=(8501, 8502, 8503, 8504, 8505)):
    for p in candidates:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    # last resort: ask OS
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def run_streamlit():
    port = find_free_port()
    py = sys.executable  # use current venv’s python
    cmd = [
        py, "-m", "streamlit", "run", "src/app/production_app.py",
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"🚀 Starting Streamlit on http://127.0.0.1:{port}")
    subprocess.check_call(cmd)

if __name__ == "__main__":
    run_streamlit()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_environment():
    """Check if environment is properly configured."""
    logger.info("Checking environment configuration...")
    
    # Check for Spotify credentials
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        logger.error("Spotify credentials not found!")
        print("\n❌ Spotify API credentials not configured!")
        print("\nPlease run the setup first:")
        print("  python setup_production.py")
        print("\nOr set environment variables manually:")
        print("  export SPOTIFY_CLIENT_ID=your_client_id")
        print("  export SPOTIFY_CLIENT_SECRET=your_client_secret")
        return False
    
    logger.info("✅ Spotify credentials found")
    
    # Check required directories
    required_dirs = ['data/cache', 'data/raw', 'data/processed', 'data/models', 'logs']
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directory structure verified")
    
    # Check if production app exists
    app_path = Path('src/app/production_app.py')
    if not app_path.exists():
        logger.error(f"Production app not found at {app_path}")
        return False
    
    logger.info("✅ Production app found")
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

def run_production_app():
    """Run the production Streamlit application."""
    logger.info("Starting Playlist Auto-DJ production application...")
    
    try:
        # Run Streamlit app
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

def print_startup_info():
    """Print startup information."""
    print("=" * 60)
    print("🎵 PLAYLIST AUTO-DJ - PRODUCTION MODE")
    print("=" * 60)
    print()
    print("🚀 Starting your personalized music recommendation system...")
    print()
    print("Features:")
    print("  ✅ Real Spotify API integration")
    print("  ✅ Machine learning taste modeling")
    print("  ✅ Advanced mood-based recommendations")
    print("  ✅ Interactive web interface")
    print("  ✅ Playlist generation and export")
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
    
    # Print startup info
    print_startup_info()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run the application
    success = run_production_app()
    
    if success:
        print("\n✅ Application stopped successfully")
    else:
        print("\n❌ Application encountered an error")
        sys.exit(1)

if __name__ == "__main__":
    main()
