# 🎵 Playlist Auto-DJ - Production Guide

## 🚀 Quick Start (Production Version)

### Step 1: Setup
```bash
# Clone or navigate to the project directory
cd playlist-auto-dj

# Run the automated setup
python setup_production.py
```

### Step 2: Configure Spotify API
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app with these settings:
   - **App Name**: Playlist Auto-DJ
   - **App Description**: Personal music recommendation system
   - **Redirect URI**: `http://localhost:8080/callback`
3. Copy your Client ID and Client Secret
4. Set environment variables:
   ```bash
   export SPOTIFY_CLIENT_ID=your_client_id_here
   export SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```

### Step 3: Run the Application
```bash
# Option 1: Use the production runner (recommended)
python run_production.py

# Option 2: Run directly with Streamlit
streamlit run src/app/production_app.py
```

### Step 4: First-Time Usage
1. **Authenticate**: Click "Connect to Spotify" in the sidebar
2. **Load Data**: The app will automatically fetch your Spotify library
3. **Train Model**: Click "Train Your Personal Taste Model"
4. **Generate Playlists**: Use the Playlist Generator tab

---

## 🎯 How It Works

### 1. Data Collection
- **Your Liked Songs**: Positive examples for taste modeling
- **Your Playlists**: Additional preference data
- **Top Tracks**: Your most-played songs across different time periods
- **Audio Features**: Spotify's audio analysis for each track

### 2. Machine Learning Pipeline
- **Feature Engineering**: Creates 79+ features from 9 base audio features
- **Mood Classification**: Categorizes tracks into 6 moods (happy, sad, energetic, calm, focus, party)
- **Taste Modeling**: Trains personalized ML model (LightGBM or Random Forest)
- **Recommendation Engine**: Multi-criteria filtering with diversity optimization

### 3. Playlist Generation
- **Mood-Based Filtering**: Select target mood and audio characteristics
- **Preference Scoring**: Uses your trained taste model
- **Diversity Control**: Balance between similar and varied tracks
- **Real-Time Generation**: Creates playlists in seconds

---

## 🎛️ Features Overview

### 📊 Your Music Library Tab
- **Library Statistics**: Total tracks, artists, popularity metrics
- **Mood Distribution**: Visual breakdown of your music moods
- **Audio Profile**: Radar chart of your musical characteristics
- **Feature Correlations**: How different audio features relate in your library

### 🎵 Generate Playlist Tab
- **Mood Selection**: Choose from 6 different moods
- **Audio Controls**: Fine-tune energy, positivity, and tempo
- **Diversity Slider**: Control similarity vs variety
- **Real-Time Generation**: Instant playlist creation
- **Export Options**: Download CSV or create Spotify playlist

### 🤖 AI Insights Tab
- **Model Performance**: Accuracy, precision, recall metrics
- **Feature Importance**: What the AI learned about your taste
- **Prediction Confidence**: How certain the model is about recommendations

### ⚙️ Settings Tab
- **Data Management**: Refresh Spotify data, clear cache
- **Model Retraining**: Update your taste model with new data
- **Cache Information**: Storage usage and performance stats

---

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Required
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

# Optional
SPOTIFY_REDIRECT_URI=http://localhost:8080/callback  # Default
```

### Configuration File (`config/config.yaml`)
```yaml
# Customize mood thresholds
features:
  mood_thresholds:
    happy:
      valence_min: 0.6
      energy_min: 0.5
    # ... other moods

# Adjust model parameters
models:
  taste_model:
    params:
      learning_rate: 0.05
      num_leaves: 31
      # ... other parameters
```

---

## 🎵 Spotify Integration Details

### Required Permissions
The app requests these Spotify scopes:
- `user-library-read`: Access your saved tracks
- `playlist-read-private`: Read your playlists
- `playlist-modify-public`: Create public playlists
- `playlist-modify-private`: Create private playlists
- `user-top-read`: Access your top tracks

### Data Privacy
- **Local Processing**: All ML training happens on your machine
- **No Data Sharing**: Your music data never leaves your computer
- **Spotify API**: Only standard API calls, no data storage on external servers
- **Cache Management**: Local caching for performance, can be cleared anytime

---

## 🚨 Troubleshooting

### Common Issues

#### 1. "Spotify credentials not found"
**Solution**: 
```bash
# Make sure environment variables are set
echo $SPOTIFY_CLIENT_ID
echo $SPOTIFY_CLIENT_SECRET

# If empty, set them:
export SPOTIFY_CLIENT_ID=your_client_id
export SPOTIFY_CLIENT_SECRET=your_client_secret
```

#### 2. "No training data available"
**Cause**: No liked songs or playlists found
**Solution**: 
- Like some songs on Spotify first
- Create playlists with your favorite tracks
- Use Spotify for a while to build listening history

#### 3. "LightGBM not available"
**Solution**: The app automatically falls back to Random Forest
```bash
# Optional: Install LightGBM for better performance
pip install lightgbm
```

#### 4. "Authentication failed"
**Solutions**:
- Check your Client ID and Secret are correct
- Ensure Redirect URI is set to `http://localhost:8080/callback`
- Try refreshing the browser page
- Clear browser cache and cookies

#### 5. "Error loading data"
**Solutions**:
- Check internet connection
- Verify Spotify API credentials
- Clear cache: Settings → Clear Cache
- Restart the application

### Performance Tips

#### For Large Libraries (10K+ tracks)
- **Enable Caching**: Keeps processed data for faster loading
- **Batch Processing**: The app automatically handles large datasets
- **Memory Management**: Close other applications if needed

#### For Better Recommendations
- **More Training Data**: Like more songs, create diverse playlists
- **Regular Retraining**: Retrain model as your taste evolves
- **Feedback Loop**: Use generated playlists and like/dislike tracks

---

## 🔄 Data Flow

```
1. Spotify API → Raw Music Data
2. Data Processor → Cleaned Dataset
3. Feature Extractor → 79+ Engineered Features
4. Mood Analyzer → Mood Classifications
5. Taste Model → Personal Preference Scores
6. Recommender → Filtered & Ranked Tracks
7. UI → Interactive Playlist Generation
```

---

## 📈 Performance Metrics

### Expected Performance
- **Data Loading**: 30-60 seconds for 1K tracks
- **Model Training**: 10-30 seconds depending on data size
- **Playlist Generation**: <5 seconds
- **Memory Usage**: 200-500MB depending on library size

### Optimization Features
- **Intelligent Caching**: Avoids re-processing unchanged data
- **Batch API Calls**: Efficient Spotify API usage
- **Feature Selection**: Uses most important features for speed
- **Progressive Loading**: Shows progress during long operations

---

## 🎯 Best Practices

### For Optimal Results
1. **Diverse Training Data**: Like songs from different genres and moods
2. **Regular Usage**: Use Spotify regularly to build listening history
3. **Playlist Variety**: Create playlists for different contexts
4. **Model Updates**: Retrain periodically as your taste evolves
5. **Feedback**: Use generated playlists and provide implicit feedback

### For Better Performance
1. **Stable Internet**: Ensure good connection for Spotify API
2. **Sufficient Memory**: Close unnecessary applications
3. **Regular Cache Cleanup**: Clear cache if it gets too large
4. **Environment Consistency**: Keep credentials and config stable

---

## 🆘 Support

### Getting Help
1. **Check Logs**: Look at `logs/app.log` for detailed error messages
2. **GitHub Issues**: Report bugs or request features
3. **Documentation**: Refer to README.md and code comments
4. **Community**: Share experiences and solutions

### Reporting Issues
When reporting issues, please include:
- Error messages from logs
- Your Python version
- Operating system
- Steps to reproduce the problem
- Size of your Spotify library (approximate)

---

## 🎵 Enjoy Your Personalized Music Experience!

The Playlist Auto-DJ learns your unique musical taste and creates perfect playlists for any mood or activity. The more you use it, the better it gets at understanding your preferences.

**Happy listening! 🎧**
