"""
Data processing pipeline for cleaning, transforming, and preparing Spotify data
for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data cleaning, transformation, and preparation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data processor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.audio_features = self.config['features']['audio_features']
        self.mood_thresholds = self.config['features']['mood_thresholds']
        
    def clean_track_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate track data."""
        logger.info(f"Cleaning track data: {len(df)} rows")
        
        # Remove rows with missing essential data
        essential_cols = ['track_id', 'name', 'artists']
        df = df.dropna(subset=essential_cols)
        
        # Handle missing values
        df['popularity'] = df['popularity'].fillna(0)
        df['duration_ms'] = df['duration_ms'].fillna(df['duration_ms'].median())
        df['explicit'] = df['explicit'].fillna(False)
        df['user_preference'] = df['user_preference'].fillna(0.0)
        
        # Convert data types
        df['explicit'] = df['explicit'].astype(bool)
        df['popularity'] = df['popularity'].astype(int)
        df['duration_ms'] = df['duration_ms'].astype(int)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['track_id'])
        
        # Filter out very short or very long tracks
        df = df[
            (df['duration_ms'] >= 30000) &  # At least 30 seconds
            (df['duration_ms'] <= 600000)   # At most 10 minutes
        ]
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    def clean_audio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate audio features data."""
        logger.info(f"Cleaning audio features: {len(df)} rows")
        
        # Remove rows with missing audio features
        df = df.dropna(subset=self.audio_features)
        
        # Validate feature ranges
        feature_ranges = {
            'acousticness': (0, 1),
            'danceability': (0, 1),
            'energy': (0, 1),
            'instrumentalness': (0, 1),
            'liveness': (0, 1),
            'speechiness': (0, 1),
            'valence': (0, 1),
            'loudness': (-60, 0),
            'tempo': (0, 250)
        }
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in df.columns:
                df = df[
                    (df[feature] >= min_val) & 
                    (df[feature] <= max_val)
                ]
        
        logger.info(f"Cleaned audio features: {len(df)} rows remaining")
        return df
    
    def merge_track_and_features(self, tracks_df: pd.DataFrame, 
                                features_df: pd.DataFrame) -> pd.DataFrame:
        """Merge track metadata with audio features."""
        logger.info("Merging track data with audio features")
        
        # Ensure track_id columns match
        if 'id' in features_df.columns:
            features_df = features_df.rename(columns={'id': 'track_id'})
        
        # Merge datasets
        merged_df = pd.merge(
            tracks_df, 
            features_df, 
            on='track_id', 
            how='inner'
        )
        
        logger.info(f"Merged dataset: {len(merged_df)} rows")
        return merged_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for better model performance."""
        logger.info("Engineering additional features")
        
        # Duration-based features
        df['duration_minutes'] = df['duration_ms'] / 60000
        df['is_short_track'] = (df['duration_ms'] < 180000).astype(int)  # < 3 minutes
        df['is_long_track'] = (df['duration_ms'] > 300000).astype(int)   # > 5 minutes
        
        # Tempo bands
        tempo_bands = self.config['features']['engineered_features']['tempo_bands']
        df['tempo_band'] = pd.cut(
            df['tempo'], 
            bins=[0, tempo_bands['slow'][1], tempo_bands['medium'][1], 300],
            labels=['slow', 'medium', 'fast']
        )
        
        # Energy bands
        energy_bands = self.config['features']['engineered_features']['energy_bands']
        df['energy_band'] = pd.cut(
            df['energy'],
            bins=[0, energy_bands['low'][1], energy_bands['medium'][1], 1],
            labels=['low', 'medium', 'high']
        )
        
        # Mood-related features
        df['happiness_score'] = (df['valence'] * 0.6 + df['energy'] * 0.4)
        df['danceability_energy'] = df['danceability'] * df['energy']
        df['acoustic_valence'] = df['acousticness'] * df['valence']
        
        # Popularity bands
        df['popularity_band'] = pd.cut(
            df['popularity'],
            bins=[0, 25, 50, 75, 100],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Loudness normalization
        df['loudness_normalized'] = (df['loudness'] + 60) / 60  # Normalize to 0-1
        
        # Interaction features
        df['valence_energy_interaction'] = df['valence'] * df['energy']
        df['danceability_tempo_interaction'] = df['danceability'] * (df['tempo'] / 200)
        
        # Time-based features (if timestamp available)
        if 'added_at' in df.columns:
            df['added_at'] = pd.to_datetime(df['added_at'])
            df['days_since_added'] = (datetime.now() - df['added_at']).dt.days
            df['is_recent'] = (df['days_since_added'] <= 30).astype(int)
        
        logger.info(f"Engineered features added. Dataset shape: {df.shape}")
        return df
    
    def classify_moods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify tracks into mood categories based on audio features."""
        logger.info("Classifying track moods")
        
        # Initialize mood columns
        mood_columns = ['is_happy', 'is_sad', 'is_energetic', 'is_calm']
        for col in mood_columns:
            df[col] = 0
        
        # Happy classification
        happy_conditions = (
            (df['valence'] >= self.mood_thresholds['happy']['valence_min']) &
            (df['energy'] >= self.mood_thresholds['happy']['energy_min'])
        )
        df.loc[happy_conditions, 'is_happy'] = 1
        
        # Sad classification
        sad_conditions = (
            (df['valence'] <= self.mood_thresholds['sad']['valence_max']) &
            (df['energy'] <= self.mood_thresholds['sad']['energy_max'])
        )
        df.loc[sad_conditions, 'is_sad'] = 1
        
        # Energetic classification
        energetic_conditions = (
            (df['energy'] >= self.mood_thresholds['energetic']['energy_min']) &
            (df['tempo'] >= self.mood_thresholds['energetic']['tempo_min'])
        )
        df.loc[energetic_conditions, 'is_energetic'] = 1
        
        # Calm classification
        calm_conditions = (
            (df['energy'] <= self.mood_thresholds['calm']['energy_max']) &
            (df['valence'] >= self.mood_thresholds['calm']['valence_range'][0]) &
            (df['valence'] <= self.mood_thresholds['calm']['valence_range'][1])
        )
        df.loc[calm_conditions, 'is_calm'] = 1
        
        # Primary mood (most dominant)
        mood_scores = df[['is_happy', 'is_sad', 'is_energetic', 'is_calm']]
        df['primary_mood'] = mood_scores.idxmax(axis=1).str.replace('is_', '')
        
        # Mood confidence (how strongly it fits the mood)
        df['mood_confidence'] = mood_scores.max(axis=1)
        
        mood_counts = df['primary_mood'].value_counts()
        logger.info(f"Mood distribution: {mood_counts.to_dict()}")
        
        return df
    
    def create_training_labels(self, df: pd.DataFrame, 
                              threshold: float = 0.5) -> pd.DataFrame:
        """Create binary labels for taste modeling."""
        logger.info("Creating training labels for taste modeling")
        
        # Binary preference labels
        df['liked'] = (df['user_preference'] >= threshold).astype(int)
        
        # Create negative examples from low-preference tracks
        df['disliked'] = (df['user_preference'] < 0.3).astype(int)
        
        # Label distribution
        liked_count = df['liked'].sum()
        disliked_count = df['disliked'].sum()
        neutral_count = len(df) - liked_count - disliked_count
        
        logger.info(f"Label distribution - Liked: {liked_count}, "
                   f"Disliked: {disliked_count}, Neutral: {neutral_count}")
        
        return df
    
    def prepare_model_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for machine learning models."""
        logger.info("Preparing features for ML models")
        
        # Select numerical features
        numerical_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
            'popularity', 'duration_minutes', 'happiness_score',
            'danceability_energy', 'acoustic_valence', 'loudness_normalized',
            'valence_energy_interaction', 'danceability_tempo_interaction'
        ]
        
        # Select categorical features to encode
        categorical_features = ['tempo_band', 'energy_band', 'popularity_band']
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
        
        # Get all feature columns
        feature_columns = []
        
        # Add numerical features
        for feat in numerical_features:
            if feat in df_encoded.columns:
                feature_columns.append(feat)
        
        # Add encoded categorical features
        for col in df_encoded.columns:
            if any(cat in col for cat in categorical_features):
                feature_columns.append(col)
        
        # Add mood features
        mood_features = ['is_happy', 'is_sad', 'is_energetic', 'is_calm']
        for feat in mood_features:
            if feat in df_encoded.columns:
                feature_columns.append(feat)
        
        # Add boolean features
        boolean_features = ['explicit', 'is_short_track', 'is_long_track']
        for feat in boolean_features:
            if feat in df_encoded.columns:
                df_encoded[feat] = df_encoded[feat].astype(int)
                feature_columns.append(feat)
        
        logger.info(f"Prepared {len(feature_columns)} features for modeling")
        return df_encoded, feature_columns
    
    def split_data(self, df: pd.DataFrame, 
                   test_size: float = 0.2, 
                   val_size: float = 0.2,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        from sklearn.model_selection import train_test_split
        
        logger.info("Splitting data into train/val/test sets")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['liked'] if 'liked' in df.columns else None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state,
            stratify=train_val['liked'] if 'liked' in train_val.columns else None
        )
        
        logger.info(f"Data split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive feature statistics."""
        stats = {
            'total_tracks': len(df),
            'unique_artists': len(set([artist for artists in df['artists'] for artist in artists])),
            'avg_popularity': df['popularity'].mean(),
            'avg_duration_minutes': df['duration_minutes'].mean(),
            'mood_distribution': df['primary_mood'].value_counts().to_dict(),
            'feature_correlations': df[self.audio_features].corr().to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return stats
    
    def process_full_pipeline(self, tracks_df: pd.DataFrame, 
                             features_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Run the complete data processing pipeline."""
        logger.info("Running full data processing pipeline")
        
        # Clean data
        tracks_clean = self.clean_track_data(tracks_df)
        features_clean = self.clean_audio_features(features_df)
        
        # Merge datasets
        merged_df = self.merge_track_and_features(tracks_clean, features_clean)
        
        # Engineer features
        engineered_df = self.engineer_features(merged_df)
        
        # Classify moods
        mood_df = self.classify_moods(engineered_df)
        
        # Create training labels
        labeled_df = self.create_training_labels(mood_df)
        
        # Prepare model features
        final_df, feature_columns = self.prepare_model_features(labeled_df)
        
        # Generate statistics
        stats = self.get_feature_statistics(final_df)
        
        logger.info("Data processing pipeline completed successfully")
        return final_df, feature_columns, stats
