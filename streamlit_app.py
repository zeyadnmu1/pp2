"""
Streamlit web application for the Playlist Auto-DJ system.
Provides an intuitive interface for mood-based playlist generation and music exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import logging
from typing import Dict, List, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import modules directly to avoid __init__.py LightGBM issues
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
    
    .mood-button {
        width: 100%;
        margin: 0.25rem 0;
    }
    
    .playlist-track {
        background-color: #fafafa;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border-left: 3px solid #1DB954;
    }
</style>
""", unsafe_allow_html=True)

class PlaylistAutoDJApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = self.load_config()
        self.initialize_components()
        
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return {}
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            self.cache_manager = CacheManager()
            self.data_processor = DataProcessor()
            self.mood_analyzer = MoodAnalyzer()
            self.feature_extractor = FeatureExtractor()
            self.taste_model = TasteModel()
            self.recommender = PlaylistRecommender()
            self.evaluator = RecommendationEvaluator()
            
            # Initialize session state
            if 'spotify_client' not in st.session_state:
                st.session_state.spotify_client = None
            if 'user_data' not in st.session_state:
                st.session_state.user_data = None
            if 'processed_data' not in st.session_state:
                st.session_state.processed_data = None
            if 'current_playlist' not in st.session_state:
                st.session_state.current_playlist = None
                
        except Exception as e:
            st.error(f"Error initializing components: {e}")
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🎵 Playlist Auto-DJ</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                AI-powered mood-aware music recommendations powered by your Spotify data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls and settings."""
        st.sidebar.title("🎛️ Controls")
        
        # Spotify Connection
        st.sidebar.subheader("🔗 Spotify Connection")
        
        if st.sidebar.button("Connect to Spotify", key="connect_spotify"):
            self.connect_spotify()
        
        # Connection status
        if st.session_state.spotify_client:
            st.sidebar.success("✅ Connected to Spotify")
        else:
            st.sidebar.warning("⚠️ Not connected to Spotify")
        
        st.sidebar.divider()
        
        # Data Management
        st.sidebar.subheader("📊 Data Management")
        
        if st.sidebar.button("Load User Data", key="load_data"):
            self.load_user_data()
        
        if st.sidebar.button("Process Data", key="process_data"):
            self.process_user_data()
        
        if st.sidebar.button("Train Taste Model", key="train_model"):
            self.train_taste_model()
        
        st.sidebar.divider()
        
        # Cache Management
        st.sidebar.subheader("🗄️ Cache Management")
        
        cache_info = self.cache_manager.get_cache_info()
        st.sidebar.metric("Cache Files", cache_info.get('cache_files_count', 0))
        st.sidebar.metric("Cache Size (MB)", f"{cache_info.get('cache_files_size_mb', 0):.1f}")
        
        if st.sidebar.button("Clear Cache", key="clear_cache"):
            cleared = self.cache_manager.clear_cache()
            st.sidebar.success(f"Cleared {cleared} cache files")
    
    def connect_spotify(self):
        """Connect to Spotify API."""
        try:
            with st.spinner("Connecting to Spotify..."):
                # Check for credentials
                if not os.getenv('SPOTIFY_CLIENT_ID') or not os.getenv('SPOTIFY_CLIENT_SECRET'):
                    st.error("Spotify credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
                    return
                
                spotify_client = SpotifyClient()
                st.session_state.spotify_client = spotify_client
                st.success("Successfully connected to Spotify!")
                
        except Exception as e:
            st.error(f"Error connecting to Spotify: {e}")
    
    def load_user_data(self):
        """Load user data from Spotify."""
        if not st.session_state.spotify_client:
            st.error("Please connect to Spotify first")
            return
        
        try:
            with st.spinner("Loading your music data..."):
                user_data = st.session_state.spotify_client.get_comprehensive_dataset()
                st.session_state.user_data = user_data
                st.success(f"Loaded {len(user_data)} tracks from your Spotify library!")
                
        except Exception as e:
            st.error(f"Error loading user data: {e}")
    
    def process_user_data(self):
        """Process user data for ML models."""
        if st.session_state.user_data is None or st.session_state.user_data.empty:
            st.error("Please load user data first")
            return
        
        try:
            with st.spinner("Processing your music data..."):
                # Get audio features
                track_ids = st.session_state.user_data['track_id'].tolist()
                audio_features = st.session_state.spotify_client.get_audio_features(track_ids)
                features_df = pd.DataFrame(audio_features)
                
                # Process data
                processed_data, feature_columns, stats = self.data_processor.process_full_pipeline(
                    st.session_state.user_data, features_df
                )
                
                st.session_state.processed_data = {
                    'data': processed_data,
                    'features': feature_columns,
                    'stats': stats
                }
                
                st.success(f"Processed {len(processed_data)} tracks with {len(feature_columns)} features!")
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
    
    def train_taste_model(self):
        """Train the taste model."""
        if not st.session_state.processed_data:
            st.error("Please process data first")
            return
        
        try:
            with st.spinner("Training your taste model..."):
                data = st.session_state.processed_data['data']
                features = st.session_state.processed_data['features']
                
                # Prepare training data
                X, y = self.taste_model.prepare_training_data(data, features)
                
                # Train model (using LightGBM by default)
                metrics = self.taste_model.train_lightgbm_model(X, y)
                
                # Set up recommender
                self.recommender.set_taste_model(self.taste_model)
                self.recommender.set_mood_analyzer(self.mood_analyzer)
                self.recommender.set_track_features(data)
                
                st.success(f"Taste model trained! AUC: {metrics['auc']:.3f}")
                
        except Exception as e:
            st.error(f"Error training model: {e}")
    
    def render_explore_tab(self):
        """Render the explore tab."""
        st.header("🔍 Explore Your Music")
        
        if not st.session_state.processed_data:
            st.info("Please load and process your Spotify data to explore your music library.")
            return
        
        data = st.session_state.processed_data['data']
        stats = st.session_state.processed_data['stats']
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tracks", stats['total_tracks'])
        with col2:
            st.metric("Unique Artists", stats['unique_artists'])
        with col3:
            st.metric("Avg Popularity", f"{stats['avg_popularity']:.1f}")
        with col4:
            st.metric("Avg Duration", f"{stats['avg_duration_minutes']:.1f} min")
        
        # Mood distribution
        st.subheader("🎭 Your Mood Distribution")
        
        if 'mood_distribution' in stats:
            mood_df = pd.DataFrame(list(stats['mood_distribution'].items()), 
                                 columns=['Mood', 'Count'])
            
            fig = px.pie(mood_df, values='Count', names='Mood', 
                        title="Distribution of Moods in Your Library")
            st.plotly_chart(fig, use_container_width=True)
        
        # Audio features radar chart
        st.subheader("🎵 Your Audio Feature Profile")
        
        audio_features = ['valence', 'energy', 'danceability', 'acousticness', 
                         'instrumentalness', 'speechiness']
        
        if all(feature in data.columns for feature in audio_features):
            feature_means = [data[feature].mean() for feature in audio_features]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=feature_means,
                theta=audio_features,
                fill='toself',
                name='Your Profile'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Your Music Taste Profile"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlations
        st.subheader("🔗 Feature Correlations")
        
        if 'feature_correlations' in stats:
            corr_data = stats['feature_correlations']
            corr_df = pd.DataFrame(corr_data)
            
            fig = px.imshow(corr_df, 
                           title="Audio Feature Correlations",
                           color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_recommend_tab(self):
        """Render the recommend tab."""
        st.header("🎯 Get Recommendations")
        
        if not st.session_state.processed_data:
            st.info("Please load and process your data first.")
            return
        
        # Recommendation controls
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎛️ Recommendation Settings")
            
            # Target mood
            mood_options = self.config['ui']['mood_options']
            target_mood = st.selectbox("Target Mood", mood_options)
            
            # Map UI mood to internal mood
            mood_mapping = {
                "Happy & Upbeat": "happy",
                "Calm & Relaxed": "calm",
                "Energetic & Pumped": "energetic",
                "Melancholic & Reflective": "sad",
                "Focus & Concentration": "focus",
                "Party & Dance": "party"
            }
            internal_mood = mood_mapping.get(target_mood, "happy")
            
            # Playlist size
            playlist_size = st.slider("Playlist Size", 5, 50, 20)
            
            # Audio feature ranges
            st.subheader("🎵 Audio Preferences")
            
            energy_range = st.slider("Energy Level", 0.0, 1.0, (0.0, 1.0), 0.1)
            valence_range = st.slider("Positivity", 0.0, 1.0, (0.0, 1.0), 0.1)
            tempo_range = st.slider("Tempo (BPM)", 60, 200, (60, 200), 10)
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                exploration_ratio = st.slider("Exploration vs Exploitation", 0.0, 0.5, 0.2, 0.05)
                st.caption("Higher values = more diverse/unexpected recommendations")
        
        with col2:
            st.subheader("🎵 Your Recommendations")
            
            if st.button("🎲 Generate Playlist", type="primary", use_container_width=True):
                self.generate_recommendations(
                    internal_mood, playlist_size, energy_range, 
                    valence_range, tempo_range, exploration_ratio
                )
            
            # Display current playlist
            if st.session_state.current_playlist is not None:
                self.display_playlist(st.session_state.current_playlist)
    
    def generate_recommendations(self, mood: str, size: int, energy_range: tuple,
                               valence_range: tuple, tempo_range: tuple, exploration_ratio: float):
        """Generate playlist recommendations."""
        try:
            with st.spinner("🎵 Generating your personalized playlist..."):
                result = self.recommender.generate_playlist(
                    target_mood=mood,
                    playlist_size=size,
                    energy_range=energy_range,
                    valence_range=valence_range,
                    tempo_range=tempo_range,
                    exploration_ratio=exploration_ratio
                )
                
                st.session_state.current_playlist = result
                
                if not result['playlist'].empty:
                    st.success(f"✨ Generated playlist with {len(result['playlist'])} tracks!")
                else:
                    st.warning("No tracks found matching your criteria. Try adjusting the settings.")
                    
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
    
    def display_playlist(self, playlist_result: Dict):
        """Display the generated playlist."""
        playlist = playlist_result['playlist']
        metadata = playlist_result['metadata']
        
        if playlist.empty:
            st.warning("No tracks in playlist")
            return
        
        # Playlist metadata
        st.subheader("📊 Playlist Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tracks", metadata.get('track_count', 0))
        with col2:
            duration = metadata.get('total_duration_minutes', 0)
            st.metric("Duration", f"{duration:.1f} min" if duration else "N/A")
        with col3:
            diversity = metadata.get('diversity_score', 0)
            st.metric("Diversity", f"{diversity:.2f}")
        with col4:
            avg_valence = metadata.get('avg_valence', 0)
            st.metric("Avg Positivity", f"{avg_valence:.2f}" if avg_valence else "N/A")
        
        # Track list
        st.subheader("🎵 Tracks")
        
        for idx, track in playlist.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    artists = ", ".join(track['artists']) if isinstance(track['artists'], list) else str(track['artists'])
                    st.markdown(f"**{track['name']}** by {artists}")
                
                with col2:
                    if 'taste_score' in track:
                        taste_score = track['taste_score']
                        st.metric("Taste", f"{taste_score:.2f}")
                
                with col3:
                    if 'recommendation_type' in track:
                        rec_type = track['recommendation_type']
                        color = "🎯" if rec_type == "exploitation" else "🔍"
                        st.write(f"{color} {rec_type.title()}")
                
                # Audio features bar
                if all(col in track for col in ['valence', 'energy', 'danceability']):
                    features = {
                        'Valence': track['valence'],
                        'Energy': track['energy'],
                        'Danceability': track['danceability']
                    }
                    
                    feature_text = " | ".join([f"{k}: {v:.2f}" for k, v in features.items()])
                    st.caption(feature_text)
                
                st.divider()
        
        # Export options
        st.subheader("📤 Export Playlist")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Download as CSV", use_container_width=True):
                self.export_playlist_csv(playlist)
        
        with col2:
            if st.button("🎵 Create Spotify Playlist", use_container_width=True):
                self.create_spotify_playlist(playlist)
    
    def export_playlist_csv(self, playlist: pd.DataFrame):
        """Export playlist as CSV."""
        try:
            # Select relevant columns
            export_columns = ['name', 'artists', 'album', 'popularity', 'valence', 'energy', 'danceability']
            available_columns = [col for col in export_columns if col in playlist.columns]
            
            csv_data = playlist[available_columns].to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="playlist_auto_dj_recommendations.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error exporting CSV: {e}")
    
    def create_spotify_playlist(self, playlist: pd.DataFrame):
        """Create playlist on Spotify."""
        if not st.session_state.spotify_client:
            st.error("Please connect to Spotify first")
            return
        
        try:
            with st.spinner("Creating playlist on Spotify..."):
                track_ids = playlist['track_id'].tolist()
                
                playlist_name = f"Auto-DJ Playlist {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
                playlist_description = "Generated by Playlist Auto-DJ - AI-powered mood-aware recommendations"
                
                playlist_id = st.session_state.spotify_client.create_playlist(
                    name=playlist_name,
                    description=playlist_description,
                    track_ids=track_ids
                )
                
                if playlist_id:
                    st.success(f"✅ Created playlist '{playlist_name}' on Spotify!")
                    st.info(f"Playlist ID: {playlist_id}")
                else:
                    st.error("Failed to create playlist on Spotify")
                    
        except Exception as e:
            st.error(f"Error creating Spotify playlist: {e}")
    
    def render_build_tab(self):
        """Render the build playlist tab."""
        st.header("🏗️ Build Custom Playlist")
        
        if not st.session_state.processed_data:
            st.info("Please load and process your data first.")
            return
        
        st.subheader("🎭 Multi-Mood Playlist")
        
        # Mood selection
        available_moods = ["happy", "sad", "energetic", "calm", "focus", "party"]
        selected_moods = st.multiselect("Select Moods", available_moods, default=["happy", "energetic"])
        
        if not selected_moods:
            st.warning("Please select at least one mood")
            return
        
        # Mood distribution
        st.subheader("📊 Mood Distribution")
        
        mood_weights = {}
        cols = st.columns(len(selected_moods))
        
        for i, mood in enumerate(selected_moods):
            with cols[i]:
                weight = st.slider(f"{mood.title()}", 0.0, 1.0, 1.0/len(selected_moods), 0.1, key=f"weight_{mood}")
                mood_weights[mood] = weight
        
        # Normalize weights
        total_weight = sum(mood_weights.values())
        if total_weight > 0:
            mood_weights = {mood: weight/total_weight for mood, weight in mood_weights.items()}
        
        # Playlist settings
        col1, col2 = st.columns(2)
        
        with col1:
            playlist_size = st.slider("Total Playlist Size", 10, 100, 30)
        
        with col2:
            transition_style = st.selectbox("Transition Style", 
                                          ["Mixed", "Gradual", "Clustered"])
        
        # Generate multi-mood playlist
        if st.button("🎵 Generate Multi-Mood Playlist", type="primary"):
            try:
                with st.spinner("Creating your multi-mood playlist..."):
                    # Generate playlist for each mood
                    mood_playlists = []
                    
                    for mood, weight in mood_weights.items():
                        n_tracks = int(playlist_size * weight)
                        if n_tracks > 0:
                            result = self.recommender.generate_playlist(
                                target_mood=mood,
                                playlist_size=n_tracks
                            )
                            
                            if not result['playlist'].empty:
                                mood_playlist = result['playlist'].copy()
                                mood_playlist['target_mood'] = mood
                                mood_playlists.append(mood_playlist)
                    
                    if mood_playlists:
                        # Combine playlists
                        combined_playlist = pd.concat(mood_playlists, ignore_index=True)
                        
                        # Apply transition style
                        if transition_style == "Mixed":
                            combined_playlist = combined_playlist.sample(frac=1).reset_index(drop=True)
                        elif transition_style == "Gradual":
                            # Sort by mood transition logic
                            combined_playlist = combined_playlist.sort_values(['target_mood', 'valence'])
                        # Clustered keeps the original grouping
                        
                        # Create result structure
                        result = {
                            'playlist': combined_playlist,
                            'metadata': {
                                'track_count': len(combined_playlist),
                                'mood_distribution': combined_playlist['target_mood'].value_counts().to_dict(),
                                'transition_style': transition_style
                            }
                        }
                        
                        st.session_state.current_playlist = result
                        st.success(f"✨ Generated multi-mood playlist with {len(combined_playlist)} tracks!")
                        
                        # Display the playlist
                        self.display_playlist(result)
                    else:
                        st.warning("Could not generate playlist with selected moods")
                        
            except Exception as e:
                st.error(f"Error generating multi-mood playlist: {e}")
    
    def render_analytics_tab(self):
        """Render the analytics tab."""
        st.header("📈 Analytics & Insights")
        
        if not st.session_state.processed_data:
            st.info("Please load and process your data first.")
            return
        
        data = st.session_state.processed_data['data']
        
        # Taste model performance
        if hasattr(self.taste_model, 'training_metrics') and self.taste_model.training_metrics:
            st.subheader("🎯 Taste Model Performance")
            
            metrics = self.taste_model.training_metrics
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            with col4:
                st.metric("AUC", f"{metrics.get('auc', 0):.3f}")
        
        # Feature importance
        if hasattr(self.taste_model, 'feature_importance') and self.taste_model.feature_importance:
            st.subheader("🔍 Feature Importance")
            
            importance_data = self.taste_model.get_feature_importance(15)
            
            if importance_data['top_features']:
                features, importances = zip(*importance_data['top_features'])
                
                fig = px.bar(
                    x=list(importances),
                    y=list(features),
                    orientation='h',
                    title="Top Features for Taste Prediction"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation evaluation
        if st.session_state.current_playlist:
            st.subheader("📊 Current Playlist Analysis")
            
            playlist = st.session_state.current_playlist['playlist']
            
            # Create dummy user preferences for evaluation
            user_preferences = {}
            if 'taste_score' in playlist.columns:
                for _, track in playlist.iterrows():
                    user_preferences[track['track_id']] = track['taste_score']
            
            if user_preferences:
                evaluation_results = self.evaluator.evaluate_recommendations(
                    playlist, user_preferences
                )
                
                # Display evaluation metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    precision_10 = evaluation_results.get('precision_at_10', 0)
                    st.metric("Precision@10", f"{precision_10:.3f}")
                
                with col2:
                    diversity = evaluation_results.get('diversity_score', 0)
                    st.metric("Diversity Score", f"{diversity:.3f}")
                
                with col3:
                    novelty = evaluation_results.get('novelty_score', 0)
                    st.metric("Novelty Score", f"{novelty:.3f}")
                
                # Evaluation report
                with st.expander("📋 Detailed Evaluation Report"):
                    report = self.evaluator.generate_evaluation_report(evaluation_results)
                    st.text(report)
        
        # Data insights
        st.subheader("💡 Music Library Insights")
        
        # Tempo distribution
        if 'tempo' in data.columns:
            fig = px.histogram(data, x='tempo', nbins=30, 
                             title="Tempo Distribution in Your Library")
            st.plotly_chart(fig, use_container_width=True)
        
        # Valence vs Energy scatter
        if 'valence' in data.columns and 'energy' in data.columns:
            fig = px.scatter(data, x='valence', y='energy', 
                           color='primary_mood' if 'primary_mood' in data.columns else None,
                           title="Valence vs Energy Distribution",
                           labels={'valence': 'Positivity', 'energy': 'Energy Level'})
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Explore", "🎯 Recommend", "🏗️ Build", "📈 Analytics"])
        
        with tab1:
            self.render_explore_tab()
        
        with tab2:
            self.render_recommend_tab()
        
        with tab3:
            self.render_build_tab()
        
        with tab4:
            self.render_analytics_tab()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>🎵 Playlist Auto-DJ - AI-Powered Music Recommendations</p>
            <p>Built with Streamlit, Spotify Web API, and Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app."""
    app = PlaylistAutoDJApp()
    app.run()

if __name__ == "__main__":
    main()
