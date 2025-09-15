"""
Playlist recommendation engine with mood-based filtering, taste modeling,
and exploration-exploitation balance for diverse playlist generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import random
import yaml

logger = logging.getLogger(__name__)


class PlaylistRecommender:
    """Advanced playlist recommendation engine with multiple strategies."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize recommender with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.rec_config = self.config['recommendation']
        self.taste_model = None
        self.mood_analyzer = None
        self.scaler = StandardScaler()
        
        # Recommendation parameters
        self.exploration_ratio = self.rec_config['exploration_ratio']
        self.diversity_weight = self.rec_config['diversity_weight']
        self.novelty_weight = self.rec_config['novelty_weight']
        
    def set_taste_model(self, taste_model):
        """Set the trained taste model."""
        self.taste_model = taste_model
        
    def set_mood_analyzer(self, mood_analyzer):
        """Set the mood analyzer."""
        self.mood_analyzer = mood_analyzer
    
    def filter_by_mood(self, df: pd.DataFrame, target_mood: str,
                      mood_threshold: float = 0.6) -> pd.DataFrame:
        """Filter tracks by target mood."""
        logger.info(f"Filtering tracks by mood: {target_mood}")
        
        if target_mood == 'any':
            return df
        
        # Check if mood scores are available
        mood_score_col = f'{target_mood}_score'
        if mood_score_col in df.columns:
            filtered_df = df[df[mood_score_col] >= mood_threshold].copy()
        elif 'primary_mood' in df.columns:
            filtered_df = df[df['primary_mood'] == target_mood].copy()
        else:
            logger.warning(f"No mood information available. Returning all tracks.")
            return df
        
        logger.info(f"Filtered to {len(filtered_df)} tracks for mood: {target_mood}")
        return filtered_df
    
    def filter_by_audio_features(self, df: pd.DataFrame,
                                energy_range: Optional[Tuple[float, float]] = None,
                                valence_range: Optional[Tuple[float, float]] = None,
                                tempo_range: Optional[Tuple[int, int]] = None,
                                danceability_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """Filter tracks by audio feature ranges."""
        logger.info("Filtering tracks by audio features")
        
        filtered_df = df.copy()
        
        if energy_range and 'energy' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['energy'] >= energy_range[0]) &
                (filtered_df['energy'] <= energy_range[1])
            ]
        
        if valence_range and 'valence' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['valence'] >= valence_range[0]) &
                (filtered_df['valence'] <= valence_range[1])
            ]
        
        if tempo_range and 'tempo' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['tempo'] >= tempo_range[0]) &
                (filtered_df['tempo'] <= tempo_range[1])
            ]
        
        if danceability_range and 'danceability' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['danceability'] >= danceability_range[0]) &
                (filtered_df['danceability'] <= danceability_range[1])
            ]
        
        logger.info(f"Filtered to {len(filtered_df)} tracks by audio features")
        return filtered_df
    
    def calculate_taste_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate taste preference scores using trained model."""
        if self.taste_model is None:
            logger.warning("No taste model available. Using random scores.")
            df_scored = df.copy()
            df_scored['taste_score'] = np.random.random(len(df))
            return df_scored
        
        try:
            df_scored = self.taste_model.predict_preferences(df)
            df_scored['taste_score'] = df_scored['preference_score']
            return df_scored
        except Exception as e:
            logger.error(f"Error calculating taste scores: {e}")
            df_scored = df.copy()
            df_scored['taste_score'] = np.random.random(len(df))
            return df_scored
    
    def calculate_diversity_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate diversity scores based on feature similarity."""
        logger.info("Calculating diversity scores")
        
        # Select features for diversity calculation
        diversity_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'valence', 'tempo', 'loudness'
        ]
        
        available_features = [f for f in diversity_features if f in df.columns]
        
        if len(available_features) < 3:
            logger.warning("Not enough features for diversity calculation")
            df_diversity = df.copy()
            df_diversity['diversity_score'] = np.random.random(len(df))
            return df_diversity
        
        # Normalize features
        feature_matrix = df[available_features].fillna(0)
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(feature_matrix_scaled)
        
        # Diversity score = 1 - average similarity to other tracks
        diversity_scores = []
        for i in range(len(df)):
            # Average similarity to all other tracks
            similarities = similarity_matrix[i]
            avg_similarity = np.mean([sim for j, sim in enumerate(similarities) if i != j])
            diversity_score = 1 - avg_similarity
            diversity_scores.append(max(0, diversity_score))  # Ensure non-negative
        
        df_diversity = df.copy()
        df_diversity['diversity_score'] = diversity_scores
        
        return df_diversity
    
    def calculate_novelty_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate novelty scores based on popularity and user history."""
        logger.info("Calculating novelty scores")
        
        df_novelty = df.copy()
        
        # Novelty based on popularity (less popular = more novel)
        if 'popularity' in df.columns:
            # Invert popularity (0-100) to novelty (1-0)
            df_novelty['popularity_novelty'] = 1 - (df['popularity'] / 100)
        else:
            df_novelty['popularity_novelty'] = 0.5
        
        # Novelty based on user preference (unknown preference = more novel)
        if 'user_preference' in df.columns:
            # Tracks with no user preference are more novel
            df_novelty['preference_novelty'] = 1 - df['user_preference'].fillna(0)
        else:
            df_novelty['preference_novelty'] = 0.5
        
        # Combined novelty score
        df_novelty['novelty_score'] = (
            df_novelty['popularity_novelty'] * 0.6 +
            df_novelty['preference_novelty'] * 0.4
        )
        
        return df_novelty
    
    def calculate_composite_scores(self, df: pd.DataFrame,
                                  taste_weight: float = 0.6,
                                  mood_weight: float = 0.3,
                                  diversity_weight: float = 0.1,
                                  target_mood: str = 'any') -> pd.DataFrame:
        """Calculate composite recommendation scores."""
        logger.info("Calculating composite recommendation scores")
        
        df_composite = df.copy()
        
        # Ensure all required scores exist
        if 'taste_score' not in df_composite.columns:
            df_composite['taste_score'] = 0.5
        
        if 'diversity_score' not in df_composite.columns:
            df_composite['diversity_score'] = 0.5
        
        # Mood score
        if target_mood != 'any' and f'{target_mood}_score' in df_composite.columns:
            mood_score = df_composite[f'{target_mood}_score']
        elif 'mood_confidence' in df_composite.columns:
            mood_score = df_composite['mood_confidence']
        else:
            mood_score = 0.5
        
        # Calculate composite score
        df_composite['recommendation_score'] = (
            df_composite['taste_score'] * taste_weight +
            mood_score * mood_weight +
            df_composite['diversity_score'] * diversity_weight
        )
        
        return df_composite
    
    def apply_exploration_exploitation(self, df: pd.DataFrame,
                                     exploration_ratio: float = None) -> pd.DataFrame:
        """Apply exploration-exploitation strategy."""
        if exploration_ratio is None:
            exploration_ratio = self.exploration_ratio
        
        logger.info(f"Applying exploration-exploitation (ratio: {exploration_ratio})")
        
        df_ee = df.copy()
        
        # Sort by recommendation score (exploitation)
        df_sorted = df_ee.sort_values('recommendation_score', ascending=False)
        
        # Calculate split point
        total_tracks = len(df_sorted)
        exploitation_count = int(total_tracks * (1 - exploration_ratio))
        
        # Exploitation tracks (top scored)
        exploitation_tracks = df_sorted.head(exploitation_count).copy()
        exploitation_tracks['selection_strategy'] = 'exploitation'
        
        # Exploration tracks (random from remaining)
        remaining_tracks = df_sorted.tail(total_tracks - exploitation_count)
        if len(remaining_tracks) > 0:
            exploration_tracks = remaining_tracks.sample(
                n=min(len(remaining_tracks), total_tracks - exploitation_count),
                random_state=42
            ).copy()
            exploration_tracks['selection_strategy'] = 'exploration'
            
            # Combine
            final_tracks = pd.concat([exploitation_tracks, exploration_tracks])
        else:
            final_tracks = exploitation_tracks
        
        # Shuffle to mix exploitation and exploration
        final_tracks = final_tracks.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return final_tracks
    
    def ensure_diversity(self, df: pd.DataFrame, max_same_artist: int = 2) -> pd.DataFrame:
        """Ensure artist diversity in recommendations."""
        logger.info(f"Ensuring artist diversity (max {max_same_artist} per artist)")
        
        if 'artists' not in df.columns:
            return df
        
        diverse_tracks = []
        artist_counts = {}
        
        # Sort by recommendation score to prioritize better tracks
        df_sorted = df.sort_values('recommendation_score', ascending=False)
        
        for _, track in df_sorted.iterrows():
            artists = track['artists']
            if isinstance(artists, list):
                primary_artist = artists[0] if artists else 'Unknown'
            else:
                primary_artist = str(artists)
            
            current_count = artist_counts.get(primary_artist, 0)
            
            if current_count < max_same_artist:
                diverse_tracks.append(track)
                artist_counts[primary_artist] = current_count + 1
        
        diverse_df = pd.DataFrame(diverse_tracks)
        logger.info(f"Ensured diversity: {len(diverse_df)} tracks from {len(artist_counts)} artists")
        
        return diverse_df
    
    def generate_playlist(self, df: pd.DataFrame,
                         target_mood: str = 'any',
                         playlist_size: int = 20,
                         energy_range: Optional[Tuple[float, float]] = None,
                         valence_range: Optional[Tuple[float, float]] = None,
                         tempo_range: Optional[Tuple[int, int]] = None,
                         danceability_range: Optional[Tuple[float, float]] = None,
                         exploration_ratio: Optional[float] = None,
                         ensure_diversity: bool = True) -> pd.DataFrame:
        """Generate a complete playlist with all recommendation strategies."""
        logger.info(f"Generating playlist: mood={target_mood}, size={playlist_size}")
        
        if df.empty:
            logger.error("No tracks available for playlist generation")
            return pd.DataFrame()
        
        # Step 1: Filter by mood
        filtered_df = self.filter_by_mood(df, target_mood)
        
        if filtered_df.empty:
            logger.warning(f"No tracks found for mood {target_mood}. Using all tracks.")
            filtered_df = df.copy()
        
        # Step 2: Filter by audio features
        filtered_df = self.filter_by_audio_features(
            filtered_df, energy_range, valence_range, tempo_range, danceability_range
        )
        
        if filtered_df.empty:
            logger.warning("No tracks match audio feature criteria. Relaxing filters.")
            filtered_df = df.copy()
        
        # Step 3: Calculate scores
        scored_df = self.calculate_taste_scores(filtered_df)
        scored_df = self.calculate_diversity_scores(scored_df)
        scored_df = self.calculate_novelty_scores(scored_df)
        
        # Step 4: Calculate composite scores
        weights = self.rec_config['reranking']
        composite_df = self.calculate_composite_scores(
            scored_df,
            taste_weight=weights['taste_weight'],
            mood_weight=weights['mood_weight'],
            diversity_weight=weights['diversity_weight'],
            target_mood=target_mood
        )
        
        # Step 5: Apply exploration-exploitation
        ee_df = self.apply_exploration_exploitation(composite_df, exploration_ratio)
        
        # Step 6: Ensure diversity
        if ensure_diversity:
            final_df = self.ensure_diversity(ee_df)
        else:
            final_df = ee_df
        
        # Step 7: Select final playlist
        playlist = final_df.head(playlist_size).copy()
        
        # Add playlist metadata
        playlist['playlist_position'] = range(len(playlist))
        playlist['target_mood'] = target_mood
        playlist['generation_timestamp'] = pd.Timestamp.now()
        
        logger.info(f"Generated playlist with {len(playlist)} tracks")
        return playlist
    
    def create_mood_journey_playlist(self, df: pd.DataFrame,
                                   mood_sequence: List[str],
                                   playlist_size: int = 20) -> pd.DataFrame:
        """Create a playlist that transitions through different moods."""
        logger.info(f"Creating mood journey playlist: {mood_sequence}")
        
        if not mood_sequence:
            return self.generate_playlist(df, playlist_size=playlist_size)
        
        # Calculate tracks per mood
        tracks_per_mood = playlist_size // len(mood_sequence)
        remaining_tracks = playlist_size % len(mood_sequence)
        
        journey_tracks = []
        
        for i, mood in enumerate(mood_sequence):
            # Calculate number of tracks for this mood
            mood_track_count = tracks_per_mood
            if i < remaining_tracks:
                mood_track_count += 1
            
            # Generate tracks for this mood
            mood_playlist = self.generate_playlist(
                df, 
                target_mood=mood,
                playlist_size=mood_track_count * 2,  # Get more to ensure variety
                ensure_diversity=True
            )
            
            # Select top tracks for this mood
            mood_tracks = mood_playlist.head(mood_track_count)
            mood_tracks['mood_section'] = mood
            mood_tracks['mood_position'] = i
            
            journey_tracks.append(mood_tracks)
        
        # Combine all mood sections
        if journey_tracks:
            final_playlist = pd.concat(journey_tracks, ignore_index=True)
            final_playlist['playlist_position'] = range(len(final_playlist))
            final_playlist['playlist_type'] = 'mood_journey'
            final_playlist['mood_sequence'] = str(mood_sequence)
            
            return final_playlist
        
        return pd.DataFrame()
    
    def create_similar_tracks_playlist(self, df: pd.DataFrame,
                                     seed_track_id: str,
                                     playlist_size: int = 20) -> pd.DataFrame:
        """Create a playlist of tracks similar to a seed track."""
        logger.info(f"Creating similar tracks playlist for seed: {seed_track_id}")
        
        # Find seed track
        seed_track = df[df['track_id'] == seed_track_id]
        if seed_track.empty:
            logger.error(f"Seed track {seed_track_id} not found")
            return pd.DataFrame()
        
        # Features for similarity calculation
        similarity_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'valence', 'tempo', 'loudness', 'speechiness'
        ]
        
        available_features = [f for f in similarity_features if f in df.columns]
        
        if len(available_features) < 3:
            logger.warning("Not enough features for similarity calculation")
            return self.generate_playlist(df, playlist_size=playlist_size)
        
        # Calculate similarities
        feature_matrix = df[available_features].fillna(0)
        seed_features = seed_track[available_features].fillna(0)
        
        # Normalize features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        seed_features_scaled = self.scaler.transform(seed_features)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(seed_features_scaled, feature_matrix_scaled)[0]
        
        # Add similarity scores
        df_similar = df.copy()
        df_similar['similarity_score'] = similarities
        
        # Remove seed track from results
        df_similar = df_similar[df_similar['track_id'] != seed_track_id]
        
        # Sort by similarity and select top tracks
        similar_playlist = df_similar.sort_values('similarity_score', ascending=False).head(playlist_size)
        
        # Add metadata
        similar_playlist['playlist_position'] = range(len(similar_playlist))
        similar_playlist['playlist_type'] = 'similar_tracks'
        similar_playlist['seed_track_id'] = seed_track_id
        
        return similar_playlist
    
    def get_recommendation_explanation(self, track: pd.Series) -> Dict:
        """Generate explanation for why a track was recommended."""
        explanation = {
            'track_id': track.get('track_id', 'unknown'),
            'track_name': track.get('name', 'Unknown'),
            'reasons': []
        }
        
        # Taste-based reasons
        if 'taste_score' in track and track['taste_score'] > 0.7:
            explanation['reasons'].append(f"High taste match (score: {track['taste_score']:.2f})")
        
        # Mood-based reasons
        if 'primary_mood' in track:
            explanation['reasons'].append(f"Matches {track['primary_mood']} mood")
        
        # Feature-based reasons
        if 'energy' in track and track['energy'] > 0.8:
            explanation['reasons'].append("High energy track")
        elif 'energy' in track and track['energy'] < 0.3:
            explanation['reasons'].append("Low energy, relaxing track")
        
        if 'valence' in track and track['valence'] > 0.8:
            explanation['reasons'].append("Very positive, upbeat track")
        elif 'valence' in track and track['valence'] < 0.3:
            explanation['reasons'].append("Melancholic, emotional track")
        
        if 'danceability' in track and track['danceability'] > 0.8:
            explanation['reasons'].append("Highly danceable")
        
        # Diversity reasons
        if 'selection_strategy' in track:
            if track['selection_strategy'] == 'exploration':
                explanation['reasons'].append("Selected for musical exploration")
            else:
                explanation['reasons'].append("Top recommendation based on your taste")
        
        # Popularity reasons
        if 'popularity' in track:
            if track['popularity'] > 80:
                explanation['reasons'].append("Popular track")
            elif track['popularity'] < 30:
                explanation['reasons'].append("Hidden gem, less mainstream")
        
        return explanation
    
    def get_playlist_statistics(self, playlist: pd.DataFrame) -> Dict:
        """Generate comprehensive playlist statistics."""
        if playlist.empty:
            return {}
        
        stats = {
            'total_tracks': len(playlist),
            'total_duration_minutes': playlist['duration_ms'].sum() / 60000 if 'duration_ms' in playlist.columns else 0,
            'unique_artists': len(set([artist for artists in playlist['artists'] for artist in artists])) if 'artists' in playlist.columns else 0,
            'average_popularity': playlist['popularity'].mean() if 'popularity' in playlist.columns else 0
        }
        
        # Audio feature statistics
        audio_features = ['energy', 'valence', 'danceability', 'acousticness', 'tempo']
        for feature in audio_features:
            if feature in playlist.columns:
                stats[f'avg_{feature}'] = playlist[feature].mean()
                stats[f'std_{feature}'] = playlist[feature].std()
        
        # Mood distribution
        if 'primary_mood' in playlist.columns:
            stats['mood_distribution'] = playlist['primary_mood'].value_counts().to_dict()
        
        # Diversity metrics
        if 'artists' in playlist.columns:
            artist_counts = {}
            for artists in playlist['artists']:
                if isinstance(artists, list):
                    for artist in artists:
                        artist_counts[artist] = artist_counts.get(artist, 0) + 1
            
            stats['artist_diversity'] = len(artist_counts) / len(playlist) if len(playlist) > 0 else 0
            stats['max_tracks_per_artist'] = max(artist_counts.values()) if artist_counts else 0
        
        return stats
