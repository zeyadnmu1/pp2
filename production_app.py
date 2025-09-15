"""
Production Streamlit application for Playlist Auto-DJ.
Integrates real Spotify data with machine learning for personalized recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.spotify_client import SpotifyClient
from data.data_processor import DataProcessor
from data.cache_manager import CacheManager
from features.mood_analyzer import MoodAnalyzer
from features.feature_extractor import FeatureExtractor
from models.taste_model import TasteModel
from models.recommender import PlaylistRecommender
from evaluation.metrics import RecommendationEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🎵 Playlist Auto-DJ",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
    }
    .stButton > button {
        background-color: #1DB954;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1ed760;
    }
</style>
""", unsafe_allow_html=True)

class ProductionApp:
    """Production Playlist Auto-DJ Application."""
    
    def __init__(self):
        """Initialize the production application."""
        self.spotify_client = None
        self.data_processor = DataProcessor()
        self.cache_manager = CacheManager()
        self.mood_analyzer = MoodAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.taste_model = TasteModel()
        self.recommender = PlaylistRecommender()
        self.evaluator = RecommendationEvaluator()
        
        # Session state initialization
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_data' not in st.session_state:
            st.session_state.user_data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = False
    
    def authenticate_spotify(self):
        """Handle Spotify authentication."""
        st.sidebar.header("🎵 Spotify Authentication")
        
        if not st.session_state.authenticated:
            st.sidebar.warning("Please authenticate with Spotify to continue")
            
            # Check for environment variables
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                st.sidebar.error("Spotify credentials not found!")
                st.sidebar.info("""
                Please set your Spotify API credentials:
                1. Create a Spotify app at https://developer.spotify.com/
                2. Set environment variables:
                   - SPOTIFY_CLIENT_ID=your_client_id
                   - SPOTIFY_CLIENT_SECRET=your_client_secret
                3. Restart the application
                """)
                return False
            
            if st.sidebar.button("🔐 Connect to Spotify"):
                try:
                    with st.spinner("Connecting to Spotify..."):
                        self.spotify_client = SpotifyClient()
                        st.session_state.authenticated = True
                        st.sidebar.success("✅ Connected to Spotify!")
                        st.experimental_rerun()
                except Exception as e:
                    st.sidebar.error(f"Authentication failed: {str(e)}")
                    return False
        else:
            st.sidebar.success("✅ Connected to Spotify")
            if self.spotify_client is None:
                self.spotify_client = SpotifyClient()
        
        return st.session_state.authenticated
    
    def load_user_data(self):
        """Load and process user's Spotify data."""
        if st.session_state.user_data is not None:
            return st.session_state.user_data
        
        st.info("🔄 Loading your Spotify data...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Get user's comprehensive dataset
            status_text.text("Fetching your music library...")
            progress_bar.progress(20)
            
            user_tracks_df = self.spotify_client.get_comprehensive_dataset()
            
            if user_tracks_df.empty:
                st.error("No music data found. Please make sure you have liked songs or playlists in your Spotify account.")
                return None
            
            # Step 2: Get audio features
            status_text.text("Analyzing audio features...")
            progress_bar.progress(40)
            
            track_ids = user_tracks_df['track_id'].tolist()
            audio_features = self.spotify_client.get_audio_features(track_ids)
            audio_features_df = pd.DataFrame(audio_features)
            
            # Step 3: Process and merge data
            status_text.text("Processing and engineering features...")
            progress_bar.progress(60)
            
            processed_df, feature_columns, stats = self.data_processor.process_full_pipeline(
                user_tracks_df, audio_features_df
            )
            
            # Step 4: Mood analysis
            status_text.text("Analyzing moods...")
            progress_bar.progress(80)
            
            processed_df = self.mood_analyzer.calculate_mood_scores(processed_df)
            processed_df = self.mood_analyzer.classify_primary_mood(processed_df)
            
            # Step 5: Feature extraction
            status_text.text("Extracting advanced features...")
            progress_bar.progress(90)
            
            processed_df, engineered_features = self.feature_extractor.extract_all_features(processed_df)
            
            progress_bar.progress(100)
            status_text.text("✅ Data loading complete!")
            
            # Cache the processed data
            self.cache_manager.save_processed_dataset(
                processed_df, feature_columns + engineered_features, stats
            )
            
            st.session_state.user_data = {
                'processed_df': processed_df,
                'feature_columns': feature_columns + engineered_features,
                'stats': stats
            }
            
            st.success(f"✅ Loaded {len(processed_df)} tracks from your Spotify library!")
            time.sleep(1)
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Data loading error: {e}")
            return None
        
        return st.session_state.user_data
    
    def train_taste_model(self, data):
        """Train the user taste model."""
        if st.session_state.trained_model:
            return True
        
        st.info("🤖 Training your personal taste model...")
        progress_bar = st.progress(0)
        
        try:
            processed_df = data['processed_df']
            feature_columns = data['feature_columns']
            
            # Prepare training data
            progress_bar.progress(25)
            X, y = self.taste_model.prepare_training_data(
                processed_df, feature_columns, target_column='user_preference'
            )
            
            # Train model (try LightGBM first, fallback to Random Forest)
            progress_bar.progress(50)
            try:
                metrics = self.taste_model.train_lightgbm_model(X, y)
                model_type = "LightGBM"
            except ImportError:
                st.warning("LightGBM not available, using Random Forest...")
                metrics = self.taste_model.train_random_forest_model(X, y)
                model_type = "Random Forest"
            
            progress_bar.progress(75)
            
            # Save model
            model_path = "data/models/user_taste_model.pkl"
            self.taste_model.save_model(model_path, metadata={
                'training_date': datetime.now().isoformat(),
                'model_type': model_type,
                'metrics': metrics
            })
            
            progress_bar.progress(100)
            st.session_state.trained_model = True
            
            st.success(f"✅ {model_type} model trained! AUC: {metrics['auc']:.3f}")
            return True
            
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            logger.error(f"Model training error: {e}")
            return False
    
    def render_data_overview(self, data):
        """Render data overview tab."""
        st.header("📊 Your Music Library Overview")
        
        processed_df = data['processed_df']
        stats = data['stats']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tracks", len(processed_df))
        
        with col2:
            st.metric("Unique Artists", stats.get('unique_artists', 'N/A'))
        
        with col3:
            st.metric("Avg Popularity", f"{stats.get('avg_popularity', 0):.1f}")
        
        with col4:
            st.metric("Avg Duration", f"{stats.get('avg_duration_minutes', 0):.1f} min")
        
        # Mood distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎭 Mood Distribution")
            mood_counts = processed_df['primary_mood'].value_counts()
            
            fig = px.pie(
                values=mood_counts.values,
                names=mood_counts.index,
                title="Your Music Moods",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Audio Features")
            
            # Select key audio features
            audio_features = ['valence', 'energy', 'danceability', 'acousticness']
            feature_means = processed_df[audio_features].mean()
            
            fig = go.Figure(data=go.Scatterpolar(
                r=feature_means.values,
                theta=feature_means.index,
                fill='toself',
                name='Your Music Profile'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Your Audio Feature Profile"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlations
        st.subheader("🔗 Feature Correlations")
        
        correlation_features = ['valence', 'energy', 'danceability', 'acousticness', 'popularity']
        corr_matrix = processed_df[correlation_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Audio Feature Correlations",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_playlist_generator(self, data):
        """Render playlist generator tab."""
        st.header("🎵 Generate Your Perfect Playlist")
        
        processed_df = data['processed_df']
        
        # Playlist configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎛️ Playlist Settings")
            
            # Target mood
            target_mood = st.selectbox(
                "Target Mood",
                options=['happy', 'sad', 'energetic', 'calm', 'focus', 'party'],
                index=0
            )
            
            # Playlist size
            playlist_size = st.slider("Playlist Size", 10, 50, 20)
            
            # Audio feature controls
            st.subheader("🎚️ Fine-tune Your Vibe")
            
            energy_range = st.slider(
                "Energy Level",
                0.0, 1.0, (0.3, 0.8),
                step=0.1
            )
            
            valence_range = st.slider(
                "Positivity",
                0.0, 1.0, (0.3, 0.8),
                step=0.1
            )
            
            tempo_range = st.slider(
                "Tempo (BPM)",
                60, 200, (90, 140),
                step=10
            )
            
            # Diversity control
            diversity_weight = st.slider(
                "Diversity (0=Similar, 1=Varied)",
                0.0, 1.0, 0.3,
                step=0.1
            )
            
            # Generate button
            generate_playlist = st.button("🎵 Generate Playlist", type="primary")
        
        with col2:
            if generate_playlist:
                st.subheader("🎧 Your Generated Playlist")
                
                with st.spinner("Creating your perfect playlist..."):
                    try:
                        # Filter tracks based on criteria
                        filtered_df = processed_df[
                            (processed_df['energy'] >= energy_range[0]) &
                            (processed_df['energy'] <= energy_range[1]) &
                            (processed_df['valence'] >= valence_range[0]) &
                            (processed_df['valence'] <= valence_range[1]) &
                            (processed_df['tempo'] >= tempo_range[0]) &
                            (processed_df['tempo'] <= tempo_range[1])
                        ].copy()
                        
                        if filtered_df.empty:
                            st.warning("No tracks match your criteria. Try adjusting the filters.")
                            return
                        
                        # Use taste model if trained
                        if st.session_state.trained_model:
                            filtered_df = self.taste_model.predict_preferences(filtered_df)
                            # Sort by preference score
                            filtered_df = filtered_df.sort_values('preference_score', ascending=False)
                        
                        # Apply mood filter
                        mood_filtered = filtered_df[
                            filtered_df['primary_mood'] == target_mood
                        ]
                        
                        if mood_filtered.empty:
                            st.warning(f"No {target_mood} tracks found. Using closest matches...")
                            mood_filtered = filtered_df.nlargest(playlist_size * 2, f'{target_mood}_score')
                        
                        # Generate final playlist
                        if len(mood_filtered) > playlist_size:
                            # Apply diversity if requested
                            if diversity_weight > 0:
                                playlist_tracks = self._apply_diversity_sampling(
                                    mood_filtered, playlist_size, diversity_weight
                                )
                            else:
                                playlist_tracks = mood_filtered.head(playlist_size)
                        else:
                            playlist_tracks = mood_filtered
                        
                        # Display playlist
                        self._display_playlist(playlist_tracks, target_mood)
                        
                        # Playlist analytics
                        self._display_playlist_analytics(playlist_tracks)
                        
                        # Export options
                        self._display_export_options(playlist_tracks)
                        
                    except Exception as e:
                        st.error(f"Error generating playlist: {str(e)}")
                        logger.error(f"Playlist generation error: {e}")
    
    def _apply_diversity_sampling(self, df, size, diversity_weight):
        """Apply diversity sampling to playlist selection."""
        if len(df) <= size:
            return df
        
        # Start with top track
        selected = [df.iloc[0]]
        remaining = df.iloc[1:].copy()
        
        for _ in range(size - 1):
            if remaining.empty:
                break
            
            # Calculate diversity scores
            diversity_scores = []
            for idx, track in remaining.iterrows():
                # Calculate average distance to selected tracks
                distances = []
                for selected_track in selected:
                    # Use audio features for distance calculation
                    features = ['valence', 'energy', 'danceability', 'acousticness']
                    distance = np.sqrt(sum([
                        (track[f] - selected_track[f]) ** 2 for f in features
                    ]))
                    distances.append(distance)
                
                avg_distance = np.mean(distances)
                diversity_scores.append(avg_distance)
            
            remaining['diversity_score'] = diversity_scores
            
            # Combine preference and diversity
            if 'preference_score' in remaining.columns:
                remaining['combined_score'] = (
                    (1 - diversity_weight) * remaining['preference_score'] +
                    diversity_weight * remaining['diversity_score']
                )
            else:
                remaining['combined_score'] = remaining['diversity_score']
            
            # Select next track
            next_track = remaining.loc[remaining['combined_score'].idxmax()]
            selected.append(next_track)
            remaining = remaining.drop(next_track.name)
        
        return pd.DataFrame(selected)
    
    def _display_playlist(self, playlist_df, mood):
        """Display the generated playlist."""
        st.success(f"✅ Generated {len(playlist_df)} {mood} tracks for you!")
        
        # Create display dataframe
        display_df = playlist_df[[
            'name', 'artists', 'primary_mood', 'valence', 'energy', 'danceability'
        ]].copy()
        
        # Format artists
        display_df['artists'] = display_df['artists'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
        
        # Round numerical values
        for col in ['valence', 'energy', 'danceability']:
            display_df[col] = display_df[col].round(2)
        
        # Rename columns
        display_df.columns = ['Track', 'Artists', 'Mood', 'Valence', 'Energy', 'Danceability']
        
        st.dataframe(display_df, use_container_width=True)
    
    def _display_playlist_analytics(self, playlist_df):
        """Display playlist analytics."""
        st.subheader("📈 Playlist Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_valence = playlist_df['valence'].mean()
            st.metric("Average Valence", f"{avg_valence:.2f}")
        
        with col2:
            avg_energy = playlist_df['energy'].mean()
            st.metric("Average Energy", f"{avg_energy:.2f}")
        
        with col3:
            # Calculate artist diversity
            all_artists = [artist for artists in playlist_df['artists'] for artist in artists]
            unique_artists = len(set(all_artists))
            diversity = unique_artists / len(all_artists) if all_artists else 0
            st.metric("Artist Diversity", f"{diversity:.2f}")
        
        # Feature distribution
        features = ['valence', 'energy', 'danceability', 'acousticness']
        feature_data = playlist_df[features].mean()
        
        fig = go.Figure(data=go.Scatterpolar(
            r=feature_data.values,
            theta=feature_data.index,
            fill='toself',
            name='Playlist Profile'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Playlist Audio Profile"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_export_options(self, playlist_df):
        """Display export options for the playlist."""
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv_data = playlist_df[['name', 'artists', 'primary_mood']].to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"playlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Spotify playlist creation (if possible)
            if st.button("🎵 Create Spotify Playlist"):
                try:
                    track_ids = playlist_df['track_id'].tolist()
                    playlist_name = f"Auto-DJ Playlist {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    
                    playlist_id = self.spotify_client.create_playlist(
                        name=playlist_name,
                        description="Generated by Playlist Auto-DJ",
                        track_ids=track_ids
                    )
                    
                    if playlist_id:
                        st.success("✅ Playlist created in your Spotify account!")
                    else:
                        st.error("Failed to create Spotify playlist")
                        
                except Exception as e:
                    st.error(f"Error creating Spotify playlist: {str(e)}")
    
    def render_model_insights(self, data):
        """Render model insights tab."""
        st.header("🤖 AI Model Insights")
        
        if not st.session_state.trained_model:
            st.warning("Train your taste model first to see insights!")
            return
        
        # Model performance
        st.subheader("📊 Model Performance")
        
        metrics = self.taste_model.training_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        
        with col4:
            st.metric("AUC Score", f"{metrics.get('auc', 0):.3f}")
        
        # Feature importance
        st.subheader("🎯 What the AI Learned About Your Taste")
        
        importance_data = self.taste_model.get_feature_importance(20)
        
        if importance_data['top_features']:
            features, scores = zip(*importance_data['top_features'])
            
            fig = px.bar(
                x=list(scores),
                y=list(features),
                orientation='h',
                title="Top 20 Features That Predict Your Music Taste",
                labels={'x': 'Importance Score', 'y': 'Features'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model summary
        st.subheader("🔍 Model Summary")
        summary = self.taste_model.get_model_summary()
        
        st.json(summary)
    
    def run(self):
        """Run the production application."""
        # Header
        st.markdown('<h1 class="main-header">🎵 Playlist Auto-DJ</h1>', unsafe_allow_html=True)
        st.markdown("### Your AI-Powered Personal Music Curator")
        
        # Authentication
        if not self.authenticate_spotify():
            st.stop()
        
        # Load user data
        user_data = self.load_user_data()
        if user_data is None:
            st.stop()
        
        # Train taste model
        if not st.session_state.trained_model:
            if st.button("🤖 Train Your Personal Taste Model", type="primary"):
                self.train_taste_model(user_data)
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Your Music Library",
            "🎵 Generate Playlist", 
            "🤖 AI Insights",
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_data_overview(user_data)
        
        with tab2:
            self.render_playlist_generator(user_data)
        
        with tab3:
            self.render_model_insights(user_data)
        
        with tab4:
            st.header("⚙️ Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔄 Data Management")
                
                if st.button("🔄 Refresh Spotify Data"):
                    st.session_state.user_data = None
                    st.experimental_rerun()
                
                if st.button("🗑️ Clear Cache"):
                    self.cache_manager.clear_cache()
                    st.success("Cache cleared!")
                
                if st.button("🤖 Retrain Model"):
                    st.session_state.trained_model = False
                    st.experimental_rerun()
            
            with col2:
                st.subheader("📊 Cache Info")
                cache_info = self.cache_manager.get_cache_info()
                st.json(cache_info)

if __name__ == "__main__":
    app = ProductionApp()
    app.run()
