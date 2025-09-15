"""
Basic functionality tests for the Playlist Auto-DJ system.
Tests core components without requiring Spotify API credentials.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.data_processor import DataProcessor
from src.data.cache_manager import CacheManager
from src.features.mood_analyzer import MoodAnalyzer
from src.features.feature_extractor import FeatureExtractor
from src.models.taste_model import TasteModel
from src.models.recommender import PlaylistRecommender
from src.evaluation.metrics import RecommendationEvaluator


class TestDataProcessor:
    """Test data processing functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.processor = DataProcessor()
        
        # Create sample track data
        self.sample_tracks = pd.DataFrame({
            'track_id': ['track1', 'track2', 'track3'],
            'name': ['Song 1', 'Song 2', 'Song 3'],
            'artists': [['Artist 1'], ['Artist 2'], ['Artist 3']],
            'album': ['Album 1', 'Album 2', 'Album 3'],
            'popularity': [80, 60, 40],
            'duration_ms': [180000, 240000, 200000],
            'explicit': [False, True, False],
            'user_preference': [1.0, 0.8, 0.3]
        })
        
        # Create sample audio features
        self.sample_features = pd.DataFrame({
            'track_id': ['track1', 'track2', 'track3'],
            'acousticness': [0.1, 0.8, 0.5],
            'danceability': [0.9, 0.3, 0.7],
            'energy': [0.8, 0.2, 0.6],
            'instrumentalness': [0.0, 0.9, 0.1],
            'liveness': [0.1, 0.1, 0.3],
            'loudness': [-5, -15, -8],
            'speechiness': [0.05, 0.02, 0.08],
            'tempo': [120, 80, 140],
            'valence': [0.8, 0.2, 0.6]
        })
    
    def test_clean_track_data(self):
        """Test track data cleaning."""
        cleaned = self.processor.clean_track_data(self.sample_tracks)
        
        assert len(cleaned) == 3
        assert 'track_id' in cleaned.columns
        assert cleaned['popularity'].dtype == int
        assert cleaned['explicit'].dtype == bool
    
    def test_clean_audio_features(self):
        """Test audio features cleaning."""
        cleaned = self.processor.clean_audio_features(self.sample_features)
        
        assert len(cleaned) == 3
        assert all(0 <= cleaned['valence'].max() <= 1)
        assert all(0 <= cleaned['energy'].max() <= 1)
    
    def test_merge_track_and_features(self):
        """Test merging tracks with features."""
        merged = self.processor.merge_track_and_features(
            self.sample_tracks, self.sample_features
        )
        
        assert len(merged) == 3
        assert 'valence' in merged.columns
        assert 'name' in merged.columns
    
    def test_full_pipeline(self):
        """Test complete processing pipeline."""
        processed_data, feature_columns, stats = self.processor.process_full_pipeline(
            self.sample_tracks, self.sample_features
        )
        
        assert len(processed_data) > 0
        assert len(feature_columns) > 0
        assert 'total_tracks' in stats
        assert stats['total_tracks'] == len(processed_data)


class TestMoodAnalyzer:
    """Test mood analysis functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.analyzer = MoodAnalyzer()
        
        self.sample_data = pd.DataFrame({
            'track_id': ['track1', 'track2', 'track3', 'track4'],
            'valence': [0.9, 0.1, 0.8, 0.3],
            'energy': [0.8, 0.2, 0.9, 0.4],
            'danceability': [0.9, 0.3, 0.8, 0.5],
            'acousticness': [0.1, 0.8, 0.2, 0.6],
            'tempo': [130, 70, 150, 90],
            'loudness': [-5, -15, -3, -10]
        })
    
    def test_calculate_mood_scores(self):
        """Test mood score calculation."""
        scored_data = self.analyzer.calculate_mood_scores(self.sample_data)
        
        # Check that mood scores are added
        mood_cols = [col for col in scored_data.columns if col.endswith('_score')]
        assert len(mood_cols) > 0
        
        # Check score ranges
        for col in mood_cols:
            assert scored_data[col].min() >= 0
            assert scored_data[col].max() <= 1
    
    def test_classify_primary_mood(self):
        """Test primary mood classification."""
        scored_data = self.analyzer.calculate_mood_scores(self.sample_data)
        classified_data = self.analyzer.classify_primary_mood(scored_data)
        
        assert 'primary_mood' in classified_data.columns
        assert 'mood_confidence' in classified_data.columns
        assert len(classified_data['primary_mood'].unique()) > 0
    
    def test_get_mood_recommendations(self):
        """Test mood-based recommendations."""
        scored_data = self.analyzer.calculate_mood_scores(self.sample_data)
        classified_data = self.analyzer.classify_primary_mood(scored_data)
        
        recommendations = self.analyzer.get_mood_recommendations(
            'happy', classified_data, n_recommendations=2
        )
        
        assert len(recommendations) <= 2


class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.extractor = FeatureExtractor()
        
        self.sample_data = pd.DataFrame({
            'track_id': ['track1', 'track2', 'track3'],
            'valence': [0.8, 0.2, 0.6],
            'energy': [0.9, 0.3, 0.7],
            'danceability': [0.8, 0.4, 0.6],
            'acousticness': [0.1, 0.8, 0.3],
            'tempo': [120, 80, 140],
            'loudness': [-5, -15, -8],
            'duration_ms': [180000, 240000, 200000],
            'popularity': [80, 40, 60]
        })
    
    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        features = self.extractor.extract_basic_features(self.sample_data)
        
        assert 'tempo_normalized' in features.columns
        assert 'loudness_normalized' in features.columns
        assert 'duration_minutes' in features.columns
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        features = self.extractor.create_interaction_features(self.sample_data)
        
        assert 'valence_energy' in features.columns
        assert 'dance_energy' in features.columns
    
    def test_create_composite_features(self):
        """Test composite feature creation."""
        basic_features = self.extractor.extract_basic_features(self.sample_data)
        composite_features = self.extractor.create_composite_features(basic_features)
        
        assert 'happiness_index' in composite_features.columns
        assert 'intensity_index' in composite_features.columns
    
    def test_extract_all_features(self):
        """Test complete feature extraction."""
        features, engineered_columns = self.extractor.extract_all_features(self.sample_data)
        
        assert len(engineered_columns) > 0
        assert len(features.columns) > len(self.sample_data.columns)


class TestTasteModel:
    """Test taste modeling functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.model = TasteModel()
        
        # Create sample data with features and labels
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(n_samples)],
            'valence': np.random.random(n_samples),
            'energy': np.random.random(n_samples),
            'danceability': np.random.random(n_samples),
            'acousticness': np.random.random(n_samples),
            'popularity': np.random.randint(0, 100, n_samples),
            'liked': np.random.randint(0, 2, n_samples)  # Binary labels
        })
        
        self.feature_columns = ['valence', 'energy', 'danceability', 'acousticness', 'popularity']
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        X, y = self.model.prepare_training_data(
            self.sample_data, self.feature_columns, target_column='liked'
        )
        
        assert X.shape[0] == len(self.sample_data)
        assert X.shape[1] == len(self.feature_columns)
        assert len(y) == len(self.sample_data)
    
    def test_train_baseline_model(self):
        """Test baseline model training."""
        X, y = self.model.prepare_training_data(
            self.sample_data, self.feature_columns, target_column='liked'
        )
        
        metrics = self.model.train_baseline_model(X, y)
        
        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert self.model.model is not None
    
    def test_predict_preferences(self):
        """Test preference prediction."""
        # Train model first
        X, y = self.model.prepare_training_data(
            self.sample_data, self.feature_columns, target_column='liked'
        )
        self.model.train_baseline_model(X, y)
        
        # Test prediction
        predictions = self.model.predict_preferences(self.sample_data)
        
        assert 'preference_score' in predictions.columns
        assert 'predicted_like' in predictions.columns
        assert len(predictions) == len(self.sample_data)


class TestRecommendationEvaluator:
    """Test evaluation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.evaluator = RecommendationEvaluator()
        
        # Create sample recommendations
        self.sample_recommendations = pd.DataFrame({
            'track_id': ['track1', 'track2', 'track3', 'track4', 'track5'],
            'name': ['Song 1', 'Song 2', 'Song 3', 'Song 4', 'Song 5'],
            'valence': [0.8, 0.6, 0.4, 0.9, 0.3],
            'energy': [0.7, 0.8, 0.5, 0.9, 0.2],
            'popularity': [80, 60, 40, 90, 20],
            'taste_score': [0.9, 0.7, 0.5, 0.8, 0.3]
        })
        
        # Create sample user preferences
        self.user_preferences = {
            'track1': 1.0,
            'track2': 0.8,
            'track3': 0.3,
            'track4': 0.9,
            'track5': 0.1
        }
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        recommended_items = ['track1', 'track2', 'track3', 'track4', 'track5']
        relevant_items = {'track1', 'track2', 'track4'}  # Items with score >= 0.5
        
        precision_5 = self.evaluator.precision_at_k(recommended_items, relevant_items, 5)
        precision_3 = self.evaluator.precision_at_k(recommended_items, relevant_items, 3)
        
        assert 0 <= precision_5 <= 1
        assert 0 <= precision_3 <= 1
    
    def test_diversity_score(self):
        """Test diversity score calculation."""
        diversity = self.evaluator.diversity_score(self.sample_recommendations)
        
        assert diversity >= 0
        assert isinstance(diversity, float)
    
    def test_novelty_score(self):
        """Test novelty score calculation."""
        novelty = self.evaluator.novelty_score(self.sample_recommendations)
        
        assert 0 <= novelty <= 1
        assert isinstance(novelty, float)
    
    def test_evaluate_recommendations(self):
        """Test comprehensive recommendation evaluation."""
        results = self.evaluator.evaluate_recommendations(
            self.sample_recommendations, self.user_preferences
        )
        
        assert 'precision_at_5' in results
        assert 'recall_at_5' in results
        assert 'diversity_score' in results
        assert 'novelty_score' in results


def test_integration():
    """Test basic integration between components."""
    # Create sample data
    tracks_df = pd.DataFrame({
        'track_id': ['track1', 'track2', 'track3'],
        'name': ['Song 1', 'Song 2', 'Song 3'],
        'artists': [['Artist 1'], ['Artist 2'], ['Artist 3']],
        'popularity': [80, 60, 40],
        'duration_ms': [180000, 240000, 200000],
        'user_preference': [1.0, 0.8, 0.3]
    })
    
    features_df = pd.DataFrame({
        'track_id': ['track1', 'track2', 'track3'],
        'valence': [0.8, 0.2, 0.6],
        'energy': [0.9, 0.3, 0.7],
        'danceability': [0.8, 0.4, 0.6],
        'acousticness': [0.1, 0.8, 0.3],
        'tempo': [120, 80, 140],
        'loudness': [-5, -15, -8],
        'instrumentalness': [0.0, 0.9, 0.1],
        'liveness': [0.1, 0.1, 0.3],
        'speechiness': [0.05, 0.02, 0.08]
    })
    
    # Test data processing pipeline
    processor = DataProcessor()
    processed_data, feature_columns, stats = processor.process_full_pipeline(tracks_df, features_df)
    
    assert len(processed_data) == 3
    assert len(feature_columns) > 10  # Should have many engineered features
    assert 'total_tracks' in stats
    
    # Test mood analysis
    mood_analyzer = MoodAnalyzer()
    mood_data = mood_analyzer.calculate_mood_scores(processed_data)
    classified_data = mood_analyzer.classify_primary_mood(mood_data)
    
    assert 'primary_mood' in classified_data.columns
    assert len(classified_data) == 3
    
    print("✅ All integration tests passed!")


if __name__ == "__main__":
    # Run basic integration test
    test_integration()
    print("🎵 Basic functionality tests completed successfully!")
