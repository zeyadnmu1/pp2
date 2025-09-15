"""
Simple script to run the Streamlit web application.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit application."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Path to the Streamlit app
    app_path = script_dir / "src" / "app" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    # Set environment variables if .env file exists
    env_path = script_dir / ".env"
    if env_path.exists():
        print("Loading environment variables from .env file...")
        from dotenv import load_dotenv
        load_dotenv(env_path)
    
    # Check for required environment variables
    required_vars = ["SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Warning: Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables or create a .env file with your Spotify API credentials.")
        print("You can still run the app, but Spotify integration won't work.")
        print()
    
    # Run Streamlit
    print("🎵 Starting Playlist Auto-DJ...")
    print(f"📁 App location: {app_path}")
    print("🌐 The app will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Run streamlit with the app path
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Playlist Auto-DJ...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
