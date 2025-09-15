"""
Mood analysis and classification system for music tracks.
Uses audio features to classify tracks into different mood categories.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import yaml

logger = logging.getLogger(__name__)


class MoodAnalyzer:
    """Advanced mood classification system for music tracks."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize mood analyzer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.mood_thresholds = self.config['features']['mood_thresholds']
        self.audio_features = self.config['features']['audio_features']
        
        # Mood categories and their characteristics
        self.mood_definitions = {
            'happy': {
                'description': 'Upbeat, positive, joyful tracks',
                'primary_features': ['valence', 'energy', 'danceability'],
                'weights': {'valence': 0.4, 'energy': 0.3, 'danceability': 0.3}
            },
            'sad': {
                'description': 'Melancholic, emotional, introspective tracks',
                'primary_features': ['valence', 'energy', 'acousticness'],
                'weights': {'valence': -0.4, 'energy': -0.3, 'acousticness': 0.3}
            },
            'energetic': {
                'description': 'High-energy, intense, powerful tracks',
                'primary_features': ['energy', 'loudness', 'tempo'],
                'weights': {'energy': 0.4, 'loudness': 0.3, 'tempo': 0.3}
            },
            'calm': {
                'description': 'Peaceful, relaxing, soothing tracks',
                'primary_features': ['energy', 'acousticness', 'instrumentalness'],
                'weights': {'energy': -0.3, 'acousticness': 0.4, 'instrumentalness': 0.3}
            },
            'focus': {
                'description': 'Concentration-friendly, minimal vocals',
                'primary_features': ['instrumentalness', 'speechiness', 'energy'],
                'weights': {'instrumentalness': 0.4, 'speechiness': -0.3, 'energy': 0.3}
            },
            'party': {
                'description': 'Danceable, social, high-energy tracks',
                'primary_features': ['danceability', 'energy', 'valence'],
                'weights': {'danceability': 0.4, 'energy': 0.3, 'valence': 0.3}
            }
        }
        
        self.mood_classifier = None
        self.feature_importance = {}
    
    def calculate_mood_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mood scores for each track using weighted features."""
        logger.info("Calculating mood scores for tracks")
        
        # Normalize features to 0-1 scale
        df_normalized = df.copy()
        
        # Normalize tempo (typical range: 60-200 BPM)
        df_normalized['tempo_normalized'] = np.clip((df['tempo'] - 60) / 140, 0, 1)
        
        # Normalize loudness (typical range: -60 to 0 dB)
        df_normalized['loudness_normalized'] = np.clip((df['loudness'] + 60) / 60, 0, 1)
        
        # Calculate mood scores
        for mood, definition in self.mood_definitions.items():
            score = np.zeros(len(df))
            
            for feature, weight in definition['weights'].items():
                if feature == 'tempo':
                    feature_values = df_normalized['tempo_normalized']
                elif feature == 'loudness':
                    feature_values = df_normalized['loudness_normalized']
                else:
                    feature_values = df_normalized[feature]
                
                score += weight * feature_values
            
            # Normalize scores to 0-1 range
            df[f'{mood}_score'] = np.clip(score, 0, 1)
        
        return df
    
    def classify_primary_mood(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify the primary mood for each track."""
        logger.info("Classifying primary moods")
        
        # Get mood score columns
        mood_score_cols = [f'{mood}_score' for mood in self.mood_definitions.keys()]
        
        # Find primary mood (highest score)
        df['primary_mood'] = df[mood_score_cols].idxmax(axis=1).str.replace('_score', '')
        
        # Calculate confidence (difference between top two scores)
        mood_scores = df[mood_score_cols].values
        sorted_scores = np.sort(mood_scores, axis=1)
        df['mood_confidence'] = sorted_scores[:, -1] - sorted_scores[:, -2]
        
        # Secondary mood (second highest score)
        mood_scores_df = df[mood_score_cols]
        df['secondary_mood'] = mood_scores_df.apply(
            lambda row: row.nlargest(2).index[1].replace('_score', ''), axis=1
        )
        
        # Mood distribution
        mood_distribution = df['primary_mood'].value_counts()
        logger.info(f"Primary mood distribution: {mood_distribution.to_dict()}")
        
        return df
    
    def create_mood_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional mood-based features."""
        logger.info("Creating mood-based features")
        
        # Mood intensity (how strongly the track fits its primary mood)
        df['mood_intensity'] = df.apply(
            lambda row: row[f"{row['primary_mood']}_score"], axis=1
        )
        
        # Mood complexity (how many moods the track fits)
        mood_score_cols = [f'{mood}_score' for mood in self.mood_definitions.keys()]
        threshold = 0.6
        df['mood_complexity'] = (df[mood_score_cols] > threshold).sum(axis=1)
        
        # Emotional valence categories
        df['emotional_category'] = pd.cut(
            df['valence'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['negative', 'neutral', 'positive']
        )
        
        # Energy categories
        df['energy_category'] = pd.cut(
            df['energy'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        # Danceability categories
        df['danceability_category'] = pd.cut(
            df['danceability'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['not_danceable', 'somewhat_danceable', 'very_danceable']
        )
        
        # Mood-energy combinations
        df['mood_energy_combo'] = df['primary_mood'] + '_' + df['energy_category'].astype(str)
        
        return df
    
    def train_mood_classifier(self, df: pd.DataFrame, 
                             target_column: str = 'primary_mood') -> Dict:
        """Train a machine learning classifier for mood prediction."""
        logger.info("Training mood classifier")
        
        # Prepare features
        feature_columns = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
            'mood_intensity', 'mood_complexity'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        self.mood_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.mood_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.mood_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        self.feature_importance = dict(zip(
            available_features,
            self.mood_classifier.feature_importances_
        ))
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importance': self.feature_importance,
            'feature_columns': available_features
        }
        
        logger.info(f"Mood classifier trained with accuracy: {accuracy:.3f}")
        return results
    
    def predict_mood(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict mood for new tracks using trained classifier."""
        if self.mood_classifier is None:
            logger.warning("Mood classifier not trained. Using rule-based classification.")
            return self.classify_primary_mood(self.calculate_mood_scores(df))
        
        # Prepare features
        feature_columns = list(self.feature_importance.keys())
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            logger.error("No features available for mood prediction")
            return df
        
        X = df[available_features]
        
        # Predict
        predictions = self.mood_classifier.predict(X)
        probabilities = self.mood_classifier.predict_proba(X)
        
        df['predicted_mood'] = predictions
        df['mood_prediction_confidence'] = np.max(probabilities, axis=1)
        
        # Add probability scores for each mood
        mood_classes = self.mood_classifier.classes_
        for i, mood in enumerate(mood_classes):
            df[f'{mood}_probability'] = probabilities[:, i]
        
        return df
    
    def analyze_mood_transitions(self, df: pd.DataFrame, 
                                playlist_column: str = 'playlist_id') -> Dict:
        """Analyze mood transitions within playlists."""
        logger.info("Analyzing mood transitions in playlists")
        
        transitions = {}
        
        for playlist_id in df[playlist_column].unique():
            playlist_tracks = df[df[playlist_column] == playlist_id].sort_values('track_position')
            
            if len(playlist_tracks) < 2:
                continue
            
            playlist_transitions = []
            moods = playlist_tracks['primary_mood'].tolist()
            
            for i in range(len(moods) - 1):
                transition = f"{moods[i]} -> {moods[i+1]}"
                playlist_transitions.append(transition)
            
            transitions[playlist_id] = playlist_transitions
        
        # Count all transitions
        all_transitions = [t for transitions_list in transitions.values() 
                          for t in transitions_list]
        transition_counts = pd.Series(all_transitions).value_counts()
        
        return {
            'playlist_transitions': transitions,
            'transition_counts': transition_counts.to_dict(),
            'most_common_transitions': transition_counts.head(10).to_dict()
        }
    
    def get_mood_recommendations(self, target_mood: str, 
                                df: pd.DataFrame, 
                                n_recommendations: int = 20) -> pd.DataFrame:
        """Get track recommendations for a specific target mood."""
        logger.info(f"Getting recommendations for mood: {target_mood}")
        
        if target_mood not in self.mood_definitions:
            logger.error(f"Unknown mood: {target_mood}")
            return pd.DataFrame()
        
        # Filter tracks by mood
        mood_tracks = df[df['primary_mood'] == target_mood].copy()
        
        if mood_tracks.empty:
            logger.warning(f"No tracks found for mood: {target_mood}")
            return pd.DataFrame()
        
        # Sort by mood score and confidence
        mood_tracks = mood_tracks.sort_values(
            [f'{target_mood}_score', 'mood_confidence'], 
            ascending=[False, False]
        )
        
        # Add recommendation score
        mood_tracks['recommendation_score'] = (
            mood_tracks[f'{target_mood}_score'] * 0.7 +
            mood_tracks['mood_confidence'] * 0.3
        )
        
        return mood_tracks.head(n_recommendations)
    
    def create_mood_playlist(self, target_moods: List[str], 
                           df: pd.DataFrame,
                           playlist_size: int = 20,
                           mood_distribution: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Create a playlist with specific mood distribution."""
        logger.info(f"Creating mood playlist with moods: {target_moods}")
        
        if mood_distribution is None:
            # Equal distribution
            mood_distribution = {mood: 1.0/len(target_moods) for mood in target_moods}
        
        playlist_tracks = []
        
        for mood, proportion in mood_distribution.items():
            if mood not in target_moods:
                continue
            
            n_tracks = int(playlist_size * proportion)
            mood_recommendations = self.get_mood_recommendations(
                mood, df, n_tracks
            )
            
            playlist_tracks.append(mood_recommendations)
        
        # Combine and shuffle
        if playlist_tracks:
            final_playlist = pd.concat(playlist_tracks, ignore_index=True)
            final_playlist = final_playlist.sample(frac=1).reset_index(drop=True)
            
            # Add playlist metadata
            final_playlist['playlist_position'] = range(len(final_playlist))
            final_playlist['target_moods'] = str(target_moods)
            
            return final_playlist.head(playlist_size)
        
        return pd.DataFrame()
    
    def get_mood_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive mood statistics."""
        stats = {}
        
        # Basic mood distribution
        stats['mood_distribution'] = df['primary_mood'].value_counts().to_dict()
        
        # Average mood scores
        mood_score_cols = [f'{mood}_score' for mood in self.mood_definitions.keys()]
        available_score_cols = [col for col in mood_score_cols if col in df.columns]
        stats['average_mood_scores'] = df[available_score_cols].mean().to_dict()
        
        # Mood confidence statistics
        if 'mood_confidence' in df.columns:
            stats['mood_confidence'] = {
                'mean': df['mood_confidence'].mean(),
                'std': df['mood_confidence'].std(),
                'min': df['mood_confidence'].min(),
                'max': df['mood_confidence'].max()
            }
        
        # Feature correlations with moods
        if available_score_cols:
            correlations = {}
            for feature in self.audio_features:
                if feature in df.columns:
                    feature_correlations = {}
                    for score_col in available_score_cols:
                        mood_name = score_col.replace('_score', '')
                        correlation = df[feature].corr(df[score_col])
                        feature_correlations[mood_name] = correlation
                    correlations[feature] = feature_correlations
            stats['feature_mood_correlations'] = correlations
        
        return stats
