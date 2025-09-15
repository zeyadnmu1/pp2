"""
Recommendation engine that combines taste modeling, mood filtering, and diversity optimization
to generate personalized playlists with exploration-exploitation balance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import random
import yaml

logger = logging.getLogger(__name__)


class PlaylistRecommender:
    """Advanced playlist recommendation engine with mood-aware filtering."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize recommender with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.rec_config = self.config['recommendation']
        self.taste_model = None
        self.mood_analyzer = None
        self.track_features = None
        self.similarity_matrix = None
        
    def set_taste_model(self, taste_model):
        """Set the trained taste model."""
        self.taste_model = taste_model
        
    def set_mood_analyzer(self, mood_analyzer):
        """Set the mood analyzer."""
        self.mood_analyzer = mood_analyzer
        
    def set_track_features(self, df: pd.DataFrame):
        """Set the track features dataset."""
        self.track_features = df.copy()
        self._compute_similarity_matrix()
    
    def _compute_similarity_matrix(self):
        """Compute track similarity matrix for content-based filtering."""
        logger.info("Computing track similarity matrix")
        
        # Select features for similarity computation
        similarity_features = [
            'valence', 'energy', 'danceability', 'acousticness',
            'instrumentalness', 'tempo_normalized', 'loudness_normalized'
        ]
        
        available_features = [f for f in similarity_features 
                            if f in self.track_features.columns]
        
        if len(available_features) < 3:
            logger.warning("Not enough features for similarity computation")
            return
        
        # Prepare feature matrix
        feature_matrix = self.track_features[available_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(feature_matrix_scaled)
        
        logger.info(f"Similarity matrix computed: {self.similarity_matrix.shape}")
    
    def filter_by_mood(self, target_mood: str, 
                      energy_range: Optional[Tuple[float, float]] = None,
                      valence_range: Optional[Tuple[float, float]] = None,
                      tempo_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """Filter tracks by target mood and optional ranges."""
        logger.info(f"Filtering tracks by mood: {target_mood}")
        
        if self.track_features is None:
            raise ValueError("Track features not set. Call set_track_features() first.")
        
        filtered_tracks = self.track_features.copy()
        
        # Mood-based filtering
        if target_mood in ['happy', 'sad', 'energetic', 'calm', 'focus', 'party']:
            if f'{target_mood}_score' in filtered_tracks.columns:
                # Use mood scores if available
                mood_threshold = 0.6
                filtered_tracks = filtered_tracks[
                    filtered_tracks[f'{target_mood}_score'] >= mood_threshold
                ]
            elif 'primary_mood' in filtered_tracks.columns:
                # Use primary mood classification
                filtered_tracks = filtered_tracks[
                    filtered_tracks['primary_mood'] == target_mood
                ]
        
        # Energy range filtering
        if energy_range:
            min_energy, max_energy = energy_range
            filtered_tracks = filtered_tracks[
                (filtered_tracks['energy'] >= min_energy) &
                (filtered_tracks['energy'] <= max_energy)
            ]
        
        # Valence range filtering
        if valence_range:
            min_valence, max_valence = valence_range
            filtered_tracks = filtered_tracks[
                (filtered_tracks['valence'] >= min_valence) &
                (filtered_tracks['valence'] <= max_valence)
            ]
        
        # Tempo range filtering
        if tempo_range:
            min_tempo, max_tempo = tempo_range
            filtered_tracks = filtered_tracks[
                (filtered_tracks['tempo'] >= min_tempo) &
                (filtered_tracks['tempo'] <= max_tempo)
            ]
        
        logger.info(f"Filtered to {len(filtered_tracks)} tracks")
        return filtered_tracks
    
    def get_taste_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get taste preference scores for tracks."""
        if self.taste_model is None:
            logger.warning("Taste model not available. Using default scores.")
            df_scored = df.copy()
            df_scored['taste_score'] = 0.5  # Neutral score
            return df_scored
        
        # Get taste predictions
        df_scored = self.taste_model.predict_preferences(df)
        df_scored['taste_score'] = df_scored['preference_score']
        
        return df_scored
    
    def get_content_based_recommendations(self, seed_tracks: List[str], 
                                        candidate_tracks: pd.DataFrame,
                                        n_recommendations: int = 20) -> pd.DataFrame:
        """Get content-based recommendations using track similarity."""
        if self.similarity_matrix is None:
            logger.warning("Similarity matrix not available. Skipping content-based filtering.")
            return candidate_tracks.head(n_recommendations)
        
        logger.info(f"Getting content-based recommendations from {len(seed_tracks)} seed tracks")
        
        # Get indices of seed tracks
        track_id_to_idx = {track_id: idx for idx, track_id in 
                          enumerate(self.track_features['track_id'])}
        
        seed_indices = []
        for track_id in seed_tracks:
            if track_id in track_id_to_idx:
                seed_indices.append(track_id_to_idx[track_id])
        
        if not seed_indices:
            logger.warning("No seed tracks found in similarity matrix")
            return candidate_tracks.head(n_recommendations)
        
        # Calculate average similarity to seed tracks
        candidate_indices = []
        candidate_track_ids = candidate_tracks['track_id'].tolist()
        
        for track_id in candidate_track_ids:
            if track_id in track_id_to_idx:
                candidate_indices.append(track_id_to_idx[track_id])
        
        if not candidate_indices:
            return candidate_tracks.head(n_recommendations)
        
        # Compute similarity scores
        similarity_scores = []
        for candidate_idx in candidate_indices:
            # Average similarity to all seed tracks
            similarities = [self.similarity_matrix[candidate_idx][seed_idx] 
                          for seed_idx in seed_indices]
            avg_similarity = np.mean(similarities)
            similarity_scores.append(avg_similarity)
        
        # Add similarity scores to candidate tracks
        candidate_tracks_scored = candidate_tracks.copy()
        candidate_tracks_scored['content_similarity'] = 0.0
        
        for i, track_id in enumerate(candidate_track_ids):
            if i < len(similarity_scores):
                mask = candidate_tracks_scored['track_id'] == track_id
                candidate_tracks_scored.loc[mask, 'content_similarity'] = similarity_scores[i]
        
        # Sort by similarity
        candidate_tracks_scored = candidate_tracks_scored.sort_values(
            'content_similarity', ascending=False
        )
        
        return candidate_tracks_scored.head(n_recommendations)
    
    def calculate_diversity_score(self, tracks: pd.DataFrame) -> float:
        """Calculate diversity score for a set of tracks."""
        if len(tracks) < 2:
            return 0.0
        
        # Features for diversity calculation
        diversity_features = ['valence', 'energy', 'danceability', 'acousticness']
        available_features = [f for f in diversity_features if f in tracks.columns]
        
        if not available_features:
            return 0.0
        
        # Calculate pairwise distances
        feature_matrix = tracks[available_features].values
        distances = []
        
        for i in range(len(feature_matrix)):
            for j in range(i + 1, len(feature_matrix)):
                distance = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                distances.append(distance)
        
        # Average distance as diversity score
        diversity = np.mean(distances) if distances else 0.0
        return diversity
    
    def rerank_recommendations(self, candidates: pd.DataFrame,
                             taste_weight: float = 0.6,
                             mood_weight: float = 0.3,
                             diversity_weight: float = 0.1,
                             target_mood: Optional[str] = None) -> pd.DataFrame:
        """Rerank recommendations using multiple criteria."""
        logger.info("Reranking recommendations")
        
        candidates_reranked = candidates.copy()
        
        # Initialize scores
        candidates_reranked['final_score'] = 0.0
        
        # Taste score component
        if 'taste_score' in candidates_reranked.columns:
            taste_scores = candidates_reranked['taste_score']
            candidates_reranked['final_score'] += taste_weight * taste_scores
        
        # Mood score component
        if target_mood and f'{target_mood}_score' in candidates_reranked.columns:
            mood_scores = candidates_reranked[f'{target_mood}_score']
            candidates_reranked['final_score'] += mood_weight * mood_scores
        elif 'mood_confidence' in candidates_reranked.columns:
            mood_scores = candidates_reranked['mood_confidence']
            candidates_reranked['final_score'] += mood_weight * mood_scores
        
        # Content similarity component
        if 'content_similarity' in candidates_reranked.columns:
            similarity_scores = candidates_reranked['content_similarity']
            candidates_reranked['final_score'] += 0.1 * similarity_scores
        
        # Popularity boost (small)
        if 'popularity' in candidates_reranked.columns:
            pop_scores = candidates_reranked['popularity'] / 100
            candidates_reranked['final_score'] += 0.05 * pop_scores
        
        # Sort by final score
        candidates_reranked = candidates_reranked.sort_values(
            'final_score', ascending=False
        )
        
        return candidates_reranked
    
    def apply_exploration_exploitation(self, recommendations: pd.DataFrame,
                                     exploration_ratio: float = 0.2) -> pd.DataFrame:
        """Apply exploration-exploitation strategy."""
        n_total = len(recommendations)
        n_exploitation = int(n_total * (1 - exploration_ratio))
        n_exploration = n_total - n_exploitation
        
        # Exploitation: top-ranked tracks
        exploitation_tracks = recommendations.head(n_exploitation)
        
        # Exploration: random selection from remaining tracks
        remaining_tracks = recommendations.iloc[n_exploitation:]
        if len(remaining_tracks) > 0:
            exploration_tracks = remaining_tracks.sample(
                n=min(n_exploration, len(remaining_tracks)),
                random_state=42
            )
        else:
            exploration_tracks = pd.DataFrame()
        
        # Combine and shuffle
        final_recommendations = pd.concat([exploitation_tracks, exploration_tracks])
        final_recommendations = final_recommendations.sample(frac=1, random_state=42)
        final_recommendations = final_recommendations.reset_index(drop=True)
        
        # Add recommendation metadata
        final_recommendations['recommendation_type'] = 'exploitation'
        if not exploration_tracks.empty:
            exploration_indices = exploration_tracks.index
            mask = final_recommendations.index.isin(exploration_indices)
            final_recommendations.loc[mask, 'recommendation_type'] = 'exploration'
        
        return final_recommendations
    
    def remove_duplicates_and_recently_played(self, 
                                            recommendations: pd.DataFrame,
                                            user_history: Optional[List[str]] = None,
                                            max_duplicates: int = 2) -> pd.DataFrame:
        """Remove duplicates and recently played tracks."""
        cleaned_recs = recommendations.copy()
        
        # Remove exact duplicates
        cleaned_recs = cleaned_recs.drop_duplicates(subset=['track_id'])
        
        # Remove recently played tracks
        if user_history:
            cleaned_recs = cleaned_recs[~cleaned_recs['track_id'].isin(user_history)]
        
        # Limit tracks per artist
        if 'artists' in cleaned_recs.columns:
            # Convert artists list to string for grouping
            cleaned_recs['artists_str'] = cleaned_recs['artists'].astype(str)
            
            # Keep only top tracks per artist
            cleaned_recs = cleaned_recs.groupby('artists_str').head(max_duplicates)
            cleaned_recs = cleaned_recs.drop('artists_str', axis=1)
        
        return cleaned_recs
    
    def generate_playlist(self, 
                         target_mood: str,
                         playlist_size: int = 20,
                         energy_range: Optional[Tuple[float, float]] = None,
                         valence_range: Optional[Tuple[float, float]] = None,
                         tempo_range: Optional[Tuple[int, int]] = None,
                         seed_tracks: Optional[List[str]] = None,
                         user_history: Optional[List[str]] = None,
                         exploration_ratio: float = 0.2) -> Dict:
        """Generate a complete playlist with the specified parameters."""
        logger.info(f"Generating playlist for mood: {target_mood}, size: {playlist_size}")
        
        if self.track_features is None:
            raise ValueError("Track features not set")
        
        # Step 1: Filter by mood and ranges
        candidates = self.filter_by_mood(
            target_mood, energy_range, valence_range, tempo_range
        )
        
        if candidates.empty:
            logger.warning("No candidates found after mood filtering")
            return {
                'playlist': pd.DataFrame(),
                'metadata': {'error': 'No tracks found matching criteria'}
            }
        
        # Step 2: Get taste scores
        candidates = self.get_taste_scores(candidates)
        
        # Step 3: Content-based filtering (if seed tracks provided)
        if seed_tracks:
            candidates = self.get_content_based_recommendations(
                seed_tracks, candidates, n_recommendations=playlist_size * 3
            )
        
        # Step 4: Rerank recommendations
        candidates = self.rerank_recommendations(
            candidates,
            taste_weight=self.rec_config['reranking']['taste_weight'],
            mood_weight=self.rec_config['reranking']['mood_weight'],
            diversity_weight=self.rec_config['reranking']['diversity_weight'],
            target_mood=target_mood
        )
        
        # Step 5: Apply exploration-exploitation
        recommendations = self.apply_exploration_exploitation(
            candidates.head(playlist_size * 2),  # More candidates for better exploration
            exploration_ratio
        )
        
        # Step 6: Remove duplicates and recently played
        final_playlist = self.remove_duplicates_and_recently_played(
            recommendations,
            user_history,
            self.rec_config['filtering']['max_duplicates']
        )
        
        # Step 7: Ensure playlist size
        final_playlist = final_playlist.head(playlist_size)
        
        # Step 8: Calculate playlist metadata
        metadata = self._calculate_playlist_metadata(final_playlist, target_mood)
        
        logger.info(f"Generated playlist with {len(final_playlist)} tracks")
        
        return {
            'playlist': final_playlist,
            'metadata': metadata
        }
    
    def _calculate_playlist_metadata(self, playlist: pd.DataFrame, 
                                   target_mood: str) -> Dict:
        """Calculate metadata for the generated playlist."""
        if playlist.empty:
            return {'track_count': 0}
        
        metadata = {
            'track_count': len(playlist),
            'target_mood': target_mood,
            'avg_valence': playlist['valence'].mean() if 'valence' in playlist.columns else None,
            'avg_energy': playlist['energy'].mean() if 'energy' in playlist.columns else None,
            'avg_danceability': playlist['danceability'].mean() if 'danceability' in playlist.columns else None,
            'avg_popularity': playlist['popularity'].mean() if 'popularity' in playlist.columns else None,
            'total_duration_minutes': playlist['duration_minutes'].sum() if 'duration_minutes' in playlist.columns else None,
            'diversity_score': self.calculate_diversity_score(playlist),
        }
        
        # Mood distribution
        if 'primary_mood' in playlist.columns:
            mood_dist = playlist['primary_mood'].value_counts().to_dict()
            metadata['mood_distribution'] = mood_dist
        
        # Recommendation type distribution
        if 'recommendation_type' in playlist.columns:
            rec_type_dist = playlist['recommendation_type'].value_counts().to_dict()
            metadata['recommendation_type_distribution'] = rec_type_dist
        
        # Taste score statistics
        if 'taste_score' in playlist.columns:
            metadata['avg_taste_score'] = playlist['taste_score'].mean()
            metadata['taste_score_std'] = playlist['taste_score'].std()
        
        # Artist diversity
        if 'artists' in playlist.columns:
            all_artists = [artist for artists_list in playlist['artists'] 
                          for artist in artists_list if isinstance(artists_list, list)]
            unique_artists = len(set(all_artists)) if all_artists else 0
            metadata['unique_artists'] = unique_artists
            metadata['artist_diversity_ratio'] = unique_artists / len(playlist) if len(playlist) > 0 else 0
        
        return metadata
    
    def get_similar_tracks(self, track_id: str, n_similar: int = 10) -> pd.DataFrame:
        """Get tracks similar to a given track."""
        if self.similarity_matrix is None or self.track_features is None:
            logger.warning("Similarity matrix or track features not available")
            return pd.DataFrame()
        
        # Find track index
        track_indices = self.track_features[self.track_features['track_id'] == track_id].index
        
        if len(track_indices) == 0:
            logger.warning(f"Track {track_id} not found in dataset")
            return pd.DataFrame()
        
        track_idx = track_indices[0]
        
        # Get similarity scores
        similarities = self.similarity_matrix[track_idx]
        
        # Get top similar tracks (excluding the track itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        similar_tracks = self.track_features.iloc[similar_indices].copy()
        similar_tracks['similarity_score'] = similarities[similar_indices]
        
        return similar_tracks
    
    def create_mood_transition_playlist(self, 
                                      start_mood: str, 
                                      end_mood: str,
                                      playlist_size: int = 20) -> Dict:
        """Create a playlist that transitions from one mood to another."""
        logger.info(f"Creating transition playlist: {start_mood} -> {end_mood}")
        
        # Calculate transition steps
        n_steps = 5
        step_size = playlist_size // n_steps
        
        # Get mood characteristics
        mood_features = {
            'happy': {'valence': 0.8, 'energy': 0.7},
            'sad': {'valence': 0.2, 'energy': 0.3},
            'energetic': {'valence': 0.6, 'energy': 0.9},
            'calm': {'valence': 0.5, 'energy': 0.2},
            'focus': {'valence': 0.5, 'energy': 0.4},
            'party': {'valence': 0.8, 'energy': 0.8}
        }
        
        start_features = mood_features.get(start_mood, {'valence': 0.5, 'energy': 0.5})
        end_features = mood_features.get(end_mood, {'valence': 0.5, 'energy': 0.5})
        
        # Create transition steps
        transition_tracks = []
        
        for step in range(n_steps):
            # Interpolate between start and end features
            progress = step / (n_steps - 1)
            target_valence = start_features['valence'] + progress * (end_features['valence'] - start_features['valence'])
            target_energy = start_features['energy'] + progress * (end_features['energy'] - start_features['energy'])
            
            # Generate tracks for this step
            step_tracks = self.generate_playlist(
                target_mood=start_mood if step < n_steps // 2 else end_mood,
                playlist_size=step_size,
                valence_range=(target_valence - 0.1, target_valence + 0.1),
                energy_range=(target_energy - 0.1, target_energy + 0.1)
            )
            
            if not step_tracks['playlist'].empty:
                step_playlist = step_tracks['playlist'].copy()
                step_playlist['transition_step'] = step
                step_playlist['target_valence'] = target_valence
                step_playlist['target_energy'] = target_energy
                transition_tracks.append(step_playlist)
        
        # Combine all steps
        if transition_tracks:
            final_playlist = pd.concat(transition_tracks, ignore_index=True)
            final_playlist = final_playlist.head(playlist_size)
            
            metadata = self._calculate_playlist_metadata(final_playlist, f"{start_mood}_to_{end_mood}")
            metadata['transition_type'] = f"{start_mood} -> {end_mood}"
            metadata['n_transition_steps'] = n_steps
            
            return {
                'playlist': final_playlist,
                'metadata': metadata
            }
        else:
            return {
                'playlist': pd.DataFrame(),
                'metadata': {'error': 'Could not create transition playlist'}
            }
