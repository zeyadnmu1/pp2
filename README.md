# Playlist Auto-DJ: Mood-Aware Music Recommender

A machine learning-powered music recommendation system that learns your taste and generates personalized playlists based on target moods and vibes.

## 🎵 Overview

This system integrates with Spotify to analyze your music library, learn your preferences, and automatically generate playlists optimized for specific moods and contexts (e.g., "calm focus", "energetic workout", "chill evening").

## 🚀 Features

- **Spotify Integration**: Seamless connection to your Spotify library via Spotipy
- **Mood Analysis**: AI-powered mood classification using audio features (valence, energy, danceability)
- **Taste Learning**: Personalized recommendation engine with LightGBM
- **Smart Playlist Generation**: Context-aware playlist creation with exploration-exploitation balance
- **Interactive Web Interface**: Streamlit-based UI with real-time controls
- **Feedback Loop**: Continuous learning from user interactions

## 🏗️ Architecture

```
Data Pipeline → Feature Engineering → Taste Modeling → Recommendation Engine → Web Interface
     ↓               ↓                    ↓                    ↓                ↓
Spotify API → Mood Classification → User Preference → Playlist Generation → Streamlit UI
```

## 📁 Project Structure

```
playlist-auto-dj/
├── src/
│   ├── data/
│   │   ├── spotify_client.py      # Spotify API integration
│   │   ├── data_processor.py      # Data cleaning and preprocessing
│   │   └── cache_manager.py       # Caching and storage management
│   ├── features/
│   │   ├── mood_analyzer.py       # Mood classification and feature engineering
│   │   └── feature_extractor.py   # Audio feature processing
│   ├── models/
│   │   ├── taste_model.py         # User taste modeling (LightGBM)
│   │   └── recommender.py         # Recommendation engine
│   ├── evaluation/
│   │   └── metrics.py             # Evaluation and feedback processing
│   └── app/
│       ├── streamlit_app.py       # Main web application
│       └── components/            # UI components
├── data/
│   ├── raw/                       # Raw Spotify data
│   ├── processed/                 # Cleaned and engineered features
│   └── models/                    # Trained model artifacts
├── config/
│   └── config.yaml               # Configuration settings
├── tests/
├── requirements.txt
└── setup.py
```

## 🛠️ Tech Stack

- **Backend**: Python 3.8+, pandas, numpy, scikit-learn, LightGBM
- **API Integration**: Spotipy (Spotify Web API)
- **Frontend**: Streamlit with custom components
- **Data Storage**: Parquet files with efficient caching
- **ML Pipeline**: Feature engineering, model training, and inference
- **Testing**: pytest, GitHub Actions CI/CD

## 📊 Success Criteria

- **Technical**: >80% mood classification accuracy, >70% precision@10 for recommendations
- **Performance**: Handle 10K+ tracks with <5s playlist generation
- **UX**: Intuitive interface with real-time feedback and seamless Spotify integration

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   git clone <repository>
   cd playlist-auto-dj
   pip install -r requirements.txt
   ```

2. **Configure Spotify API**:
   - Create Spotify App at https://developer.spotify.com/
   - Add credentials to `config/config.yaml`

3. **Run Application**:
   ```bash
   streamlit run src/app/streamlit_app.py
   ```

## 📈 Development Roadmap

### MVP (Phase 1)
- [ ] Basic Spotify integration
- [ ] Simple mood classification
- [ ] Baseline recommendation engine
- [ ] Basic Streamlit interface

### Full Version (Phase 2)
- [ ] Advanced ML models
- [ ] Feedback loop integration
- [ ] Enhanced UI/UX
- [ ] Playlist export to Spotify
- [ ] Performance optimization

## 🧪 Testing & Evaluation

- **Offline Metrics**: Precision@K, Recall@K, Diversity Score
- **Online Evaluation**: User feedback (thumbs up/down), playlist completion rates
- **A/B Testing**: Compare different recommendation strategies

## 📝 License

MIT License - See LICENSE file for details
