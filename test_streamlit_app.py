"""
Test version of Streamlit app that bypasses LightGBM import issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock data for testing UI
def create_mock_data():
    """Create mock data for testing the UI."""
    np.random.seed(42)
    n_tracks = 100
    
    tracks_data = {
        'track_id': [f'track_{i}' for i in range(n_tracks)],
        'name': [f'Song {i}' for i in range(n_tracks)],
        'artists': [[f'Artist {i}'] for i in range(n_tracks)],
        'album': [f'Album {i}' for i in range(n_tracks)],
        'popularity': np.random.randint(0, 100, n_tracks),
        'duration_ms': np.random.randint(120000, 300000, n_tracks),
        'explicit': np.random.choice([True, False], n_tracks),
        'acousticness': np.random.random(n_tracks),
        'danceability': np.random.random(n_tracks),
        'energy': np.random.random(n_tracks),
        'instrumentalness': np.random.random(n_tracks),
        'liveness': np.random.random(n_tracks),
        'loudness': np.random.uniform(-60, 0, n_tracks),
        'speechiness': np.random.random(n_tracks),
        'tempo': np.random.uniform(60, 200, n_tracks),
        'valence': np.random.random(n_tracks),
        'user_preference': np.random.random(n_tracks),
        'primary_mood': np.random.choice(['happy', 'sad', 'energetic', 'calm', 'focus', 'party'], n_tracks),
        'mood_confidence': np.random.random(n_tracks),
        'preference_score': np.random.random(n_tracks)
    }
    
    return pd.DataFrame(tracks_data)

def main():
    """Main Streamlit application."""
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎵 Playlist Auto-DJ</h1>', unsafe_allow_html=True)
    st.markdown("### Mood-Aware Music Recommender System")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Create mock data
    df = create_mock_data()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🎭 Mood Analyzer", "🎵 Playlist Generator", "🤖 Model Insights"])
    
    with tab1:
        st.header("📊 Music Library Explorer")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tracks", len(df))
        with col2:
            st.metric("Avg Popularity", f"{df['popularity'].mean():.1f}")
        with col3:
            st.metric("Avg Energy", f"{df['energy'].mean():.2f}")
        with col4:
            st.metric("Avg Valence", f"{df['valence'].mean():.2f}")
        
        # Feature distribution plots
        st.subheader("Audio Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='energy', nbins=20, title='Energy Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(df, x='valence', nbins=20, title='Valence Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='danceability', nbins=20, title='Danceability Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(df, x='acousticness', nbins=20, title='Acousticness Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                         'liveness', 'speechiness', 'valence']
        corr_matrix = df[audio_features].corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Audio Feature Correlation Matrix",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🎭 Mood Analysis")
        
        # Mood distribution
        mood_counts = df['primary_mood'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=mood_counts.values, names=mood_counts.index, 
                        title="Mood Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=mood_counts.index, y=mood_counts.values,
                        title="Tracks per Mood Category")
            st.plotly_chart(fig, use_container_width=True)
        
        # Mood vs Audio Features
        st.subheader("Mood Characteristics")
        
        feature_to_plot = st.selectbox("Select Audio Feature", 
                                     ['energy', 'valence', 'danceability', 'acousticness'])
        
        fig = px.box(df, x='primary_mood', y=feature_to_plot,
                    title=f"{feature_to_plot.title()} by Mood")
        st.plotly_chart(fig, use_container_width=True)
        
        # Mood confidence
        st.subheader("Mood Classification Confidence")
        fig = px.histogram(df, x='mood_confidence', nbins=20,
                          title="Mood Classification Confidence Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🎵 Playlist Generator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Playlist Settings")
            
            # Target mood
            target_mood = st.selectbox("Target Mood", 
                                     ['happy', 'sad', 'energetic', 'calm', 'focus', 'party'])
            
            # Energy range
            energy_range = st.slider("Energy Range", 0.0, 1.0, (0.3, 0.8), 0.1)
            
            # Valence range
            valence_range = st.slider("Valence Range", 0.0, 1.0, (0.3, 0.8), 0.1)
            
            # Tempo range
            tempo_range = st.slider("Tempo Range (BPM)", 60, 200, (100, 150), 10)
            
            # Playlist size
            playlist_size = st.slider("Playlist Size", 5, 50, 20)
            
            # Generate button
            if st.button("🎵 Generate Playlist", type="primary"):
                # Filter tracks based on criteria
                filtered_df = df[
                    (df['primary_mood'] == target_mood) &
                    (df['energy'] >= energy_range[0]) & (df['energy'] <= energy_range[1]) &
                    (df['valence'] >= valence_range[0]) & (df['valence'] <= valence_range[1]) &
                    (df['tempo'] >= tempo_range[0]) & (df['tempo'] <= tempo_range[1])
                ]
                
                if len(filtered_df) >= playlist_size:
                    # Sort by preference score and take top tracks
                    playlist = filtered_df.nlargest(playlist_size, 'preference_score')
                    st.session_state['generated_playlist'] = playlist
                    st.success(f"✅ Generated playlist with {len(playlist)} tracks!")
                else:
                    st.warning(f"⚠️ Only {len(filtered_df)} tracks match your criteria. Try adjusting the filters.")
        
        with col2:
            st.subheader("Generated Playlist")
            
            if 'generated_playlist' in st.session_state:
                playlist = st.session_state['generated_playlist']
                
                # Playlist metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Tracks", len(playlist))
                with col_b:
                    st.metric("Avg Energy", f"{playlist['energy'].mean():.2f}")
                with col_c:
                    st.metric("Avg Valence", f"{playlist['valence'].mean():.2f}")
                with col_d:
                    st.metric("Avg Tempo", f"{playlist['tempo'].mean():.0f}")
                
                # Playlist visualization
                fig = px.scatter(playlist, x='energy', y='valence', 
                               size='preference_score', color='primary_mood',
                               hover_data=['name', 'artists'],
                               title="Playlist Tracks: Energy vs Valence")
                st.plotly_chart(fig, use_container_width=True)
                
                # Track list
                st.subheader("Track List")
                display_cols = ['name', 'artists', 'energy', 'valence', 'tempo', 'preference_score']
                st.dataframe(playlist[display_cols], use_container_width=True)
                
                # Export options
                st.subheader("Export Options")
                col_x, col_y = st.columns(2)
                with col_x:
                    csv = playlist.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, "playlist.csv", "text/csv")
                with col_y:
                    st.button("🎵 Export to Spotify", disabled=True, 
                             help="Spotify integration requires API setup")
            else:
                st.info("👆 Configure your preferences and click 'Generate Playlist' to create a custom playlist!")
    
    with tab4:
        st.header("🤖 Model Insights")
        
        st.subheader("System Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Processing", "✅ Active")
        with col2:
            st.metric("Mood Classification", "✅ Active")
        with col3:
            st.metric("Taste Model", "⚠️ Demo Mode")
        
        st.info("🔧 This is a demo version. Full ML models require proper setup and training data.")
        
        # Feature importance (mock)
        st.subheader("Feature Importance (Demo)")
        features = ['energy', 'valence', 'danceability', 'acousticness', 'tempo']
        importance = np.random.random(len(features))
        
        fig = px.bar(x=features, y=importance, title="Feature Importance for Taste Modeling")
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance (mock)
        st.subheader("Model Performance (Demo)")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
            'Score': [0.85, 0.82, 0.88, 0.85, 0.90]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(metrics_df, x='Metric', y='Score', 
                    title="Taste Model Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
