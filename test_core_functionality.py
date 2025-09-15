"""
Core functionality test that avoids LightGBM dependency issues.
Tests individual components without importing the full package.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_data_processor():
    """Test data processing functionality."""
    print("🧪 Testing DataProcessor...")
    
    from src.data.data_processor import DataProcessor
    
    processor = DataProcessor()
    
    # Create sample data
    tracks_df = pd.DataFrame({
        'track_id': ['track1', 'track2', 'track3'],
        'name': ['Song 1', 'Song 2', 'Song 3'],
        'artists': [['Artist 1'], ['Artist 2'], ['Artist 3']],
        'album': ['Album 1', 'Album 2', 'Album 3'],
        'popularity': [80, 60, 40],
        'duration_ms': [180000, 240000, 200000],
        'explicit': [False, True, False],
        'user_preference': [1.0, 0.8, 0.3]
    })
    
    features_df = pd.DataFrame({
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
    
    # Test data processing pipeline
    processed_data, feature_columns, stats = processor.process_full_pipeline(tracks_df, features_df)
    
    assert len(processed_data) == 3, f"Expected 3 tracks, got {len(processed_data)}"
    assert len(feature_columns) > 10, f"Expected >10 features, got {len(feature_columns)}"
    assert 'total_tracks' in stats, "Missing total_tracks in stats"
    
    print(f"✅ DataProcessor: {len(processed_data)} tracks, {len(feature_columns)} features")
    return True

def test_mood_analyzer():
    """Test mood analysis functionality."""
    print("🧪 Testing MoodAnalyzer...")
    
    from src.features.mood_analyzer import MoodAnalyzer
    
    analyzer = MoodAnalyzer()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'track_id': ['track1', 'track2', 'track3', 'track4'],
        'valence': [0.9, 0.1, 0.8, 0.3],
        'energy': [0.8, 0.2, 0.9, 0.4],
        'danceability': [0.9, 0.3, 0.8, 0.5],
        'acousticness': [0.1, 0.8, 0.2, 0.6],
        'tempo': [130, 70, 150, 90],
        'loudness': [-5, -15, -3, -10]
    })
    
    # Test mood scoring
    scored_data = analyzer.calculate_mood_scores(sample_data)
    mood_cols = [col for col in scored_data.columns if col.endswith('_score')]
    assert len(mood_cols) > 0, "No mood scores generated"
    
    # Test mood classification
    classified_data = analyzer.classify_primary_mood(scored_data)
    assert 'primary_mood' in classified_data.columns, "Missing primary_mood column"
    assert 'mood_confidence' in classified_data.columns, "Missing mood_confidence column"
    
    print(f"✅ MoodAnalyzer: {len(mood_cols)} mood categories classified")
    return True

def test_feature_extractor():
    """Test feature extraction functionality."""
    print("🧪 Testing FeatureExtractor...")
    
    from src.features.feature_extractor import FeatureExtractor
    
    extractor = FeatureExtractor()
    
    # Create sample data
    sample_data = pd.DataFrame({
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
    
    # Test feature extraction
    features, engineered_columns = extractor.extract_all_features(sample_data)
    
    assert len(engineered_columns) > 0, "No engineered features created"
    assert len(features.columns) > len(sample_data.columns), "No new features added"
    
    print(f"✅ FeatureExtractor: {len(engineered_columns)} engineered features")
    return True

def test_cache_manager():
    """Test cache management functionality."""
    print("🧪 Testing CacheManager...")
    
    from src.data.cache_manager import CacheManager
    
    cache_manager = CacheManager()
    
    # Test DataFrame saving/loading
    test_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    
    # Save and load DataFrame
    success = cache_manager.save_dataframe(test_df, 'test_dataframe')
    assert success, "Failed to save DataFrame"
    
    loaded_df = cache_manager.load_dataframe('test_dataframe')
    assert loaded_df is not None, "Failed to load DataFrame"
    assert len(loaded_df) == 3, "Loaded DataFrame has wrong length"
    
    # Test JSON saving/loading
    test_data = {'key1': 'value1', 'key2': [1, 2, 3]}
    success = cache_manager.save_json(test_data, 'test_json')
    assert success, "Failed to save JSON"
    
    loaded_data = cache_manager.load_json('test_json')
    assert loaded_data is not None, "Failed to load JSON"
    assert loaded_data['key1'] == 'value1', "JSON data corrupted"
    
    print("✅ CacheManager: DataFrame and JSON operations working")
    return True

def test_evaluation_metrics():
    """Test evaluation functionality."""
    print("🧪 Testing RecommendationEvaluator...")
    
    from src.evaluation.metrics import RecommendationEvaluator
    
    evaluator = RecommendationEvaluator()
    
    # Test precision@k
    recommended_items = ['track1', 'track2', 'track3', 'track4', 'track5']
    relevant_items = {'track1', 'track2', 'track4'}
    
    precision_5 = evaluator.precision_at_k(recommended_items, relevant_items, 5)
    assert 0 <= precision_5 <= 1, f"Invalid precision score: {precision_5}"
    
    # Test diversity score
    sample_recommendations = pd.DataFrame({
        'track_id': ['track1', 'track2', 'track3'],
        'valence': [0.8, 0.6, 0.4],
        'energy': [0.7, 0.8, 0.5],
        'danceability': [0.9, 0.5, 0.3]
    })
    
    diversity = evaluator.diversity_score(sample_recommendations)
    assert diversity >= 0, f"Invalid diversity score: {diversity}"
    
    print(f"✅ RecommendationEvaluator: Precision@5={precision_5:.3f}, Diversity={diversity:.3f}")
    return True

def test_integration():
    """Test integration between components."""
    print("🧪 Testing Integration...")
    
    # Import components individually to avoid LightGBM
    from src.data.data_processor import DataProcessor
    from src.features.mood_analyzer import MoodAnalyzer
    from src.features.feature_extractor import FeatureExtractor
    
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
    
    # Test full pipeline
    processor = DataProcessor()
    processed_data, feature_columns, stats = processor.process_full_pipeline(tracks_df, features_df)
    
    # Test mood analysis on processed data
    mood_analyzer = MoodAnalyzer()
    mood_data = mood_analyzer.calculate_mood_scores(processed_data)
    classified_data = mood_analyzer.classify_primary_mood(mood_data)
    
    # Test feature extraction
    feature_extractor = FeatureExtractor()
    enhanced_features, engineered_cols = feature_extractor.extract_all_features(classified_data)
    
    assert len(enhanced_features) == 3, "Integration failed: wrong number of tracks"
    assert 'primary_mood' in enhanced_features.columns, "Integration failed: missing mood classification"
    assert len(engineered_cols) > 0, "Integration failed: no engineered features"
    
    print(f"✅ Integration: {len(enhanced_features)} tracks with {len(enhanced_features.columns)} total features")
    return True

def main():
    """Run all core functionality tests."""
    print("🎵 Starting Playlist Auto-DJ Core Functionality Tests\n")
    
    tests = [
        test_data_processor,
        test_mood_analyzer,
        test_feature_extractor,
        test_cache_manager,
        test_evaluation_metrics,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} failed with error: {e}")
        print()
    
    print("=" * 50)
    print(f"🎵 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All core functionality tests passed!")
        print("\n📋 System Status:")
        print("✅ Data processing pipeline working")
        print("✅ Mood analysis and classification working")
        print("✅ Feature engineering working")
        print("✅ Caching system working")
        print("✅ Evaluation metrics working")
        print("✅ Component integration working")
        print("\n🚀 The system is ready for use!")
        print("📝 Note: ML models require LightGBM dependency to be properly installed")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
