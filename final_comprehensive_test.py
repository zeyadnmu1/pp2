"""
Final comprehensive test suite for the Playlist Auto-DJ system.
Tests all components including data processing, ML models, recommendations, and UI.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import traceback

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_processing():
    """Test data processing pipeline."""
    logger.info("🔄 Testing Data Processing Pipeline...")
    
    try:
        from data.data_processor import DataProcessor
        from data.cache_manager import CacheManager
        
        # Create test data
        test_tracks = pd.DataFrame({
            'track_id': ['track_1', 'track_2', 'track_3'],
            'name': ['Song 1', 'Song 2', 'Song 3'],
            'artists': [['Artist 1'], ['Artist 2'], ['Artist 3']],
            'album': ['Album 1', 'Album 2', 'Album 3'],
            'popularity': [80, 60, 90],
            'duration_ms': [180000, 210000, 195000],
            'explicit': [False, True, False],
            'user_preference': [1.0, 0.0, 0.8]
        })
        
        test_features = pd.DataFrame({
            'track_id': ['track_1', 'track_2', 'track_3'],
            'acousticness': [0.2, 0.8, 0.3],
            'danceability': [0.7, 0.4, 0.9],
            'energy': [0.8, 0.3, 0.9],
            'instrumentalness': [0.1, 0.9, 0.0],
            'liveness': [0.2, 0.1, 0.3],
            'loudness': [-5, -12, -4],
            'speechiness': [0.1, 0.05, 0.15],
            'tempo': [120, 80, 140],
            'valence': [0.8, 0.2, 0.9]
        })
        
        # Test data processor
        processor = DataProcessor()
        final_df, feature_columns, stats = processor.process_full_pipeline(test_tracks, test_features)
        
        # Test cache manager
        cache_manager = CacheManager()
        cache_manager.save_dataframe(final_df, "test_processed_data")
        loaded_df = cache_manager.load_dataframe("test_processed_data")
        
        assert len(final_df) == 3, "Data processing failed: wrong number of rows"
        assert len(feature_columns) > 10, "Feature engineering failed: too few features"
        assert loaded_df is not None, "Cache manager failed"
        
        logger.info(f"✅ Data Processing: {len(final_df)} tracks, {len(feature_columns)} features")
        return True, final_df, feature_columns
        
    except Exception as e:
        logger.error(f"❌ Data Processing failed: {e}")
        return False, None, None

def test_mood_analysis():
    """Test mood analysis system."""
    logger.info("🎭 Testing Mood Analysis...")
    
    try:
        from features.mood_analyzer import MoodAnalyzer
        
        # Create test data with audio features
        test_data = pd.DataFrame({
            'track_id': ['track_1', 'track_2', 'track_3', 'track_4'],
            'acousticness': [0.2, 0.8, 0.1, 0.5],
            'danceability': [0.9, 0.3, 0.8, 0.6],
            'energy': [0.9, 0.2, 0.8, 0.4],
            'instrumentalness': [0.0, 0.9, 0.1, 0.3],
            'liveness': [0.2, 0.1, 0.3, 0.2],
            'loudness': [-3, -15, -5, -8],
            'speechiness': [0.1, 0.05, 0.15, 0.08],
            'tempo': [140, 70, 130, 100],
            'valence': [0.9, 0.1, 0.8, 0.5]
        })
        
        analyzer = MoodAnalyzer()
        
        # Test mood score calculation
        mood_data = analyzer.calculate_mood_scores(test_data)
        mood_classified = analyzer.classify_primary_mood(mood_data)
        
        # Test mood features
        mood_features = analyzer.create_mood_features(mood_classified)
        
        # Test mood statistics
        stats = analyzer.get_mood_statistics(mood_features)
        
        assert 'primary_mood' in mood_classified.columns, "Mood classification failed"
        assert 'mood_confidence' in mood_classified.columns, "Mood confidence calculation failed"
        assert len(stats['mood_distribution']) > 0, "Mood statistics failed"
        
        mood_counts = mood_classified['primary_mood'].value_counts()
        logger.info(f"✅ Mood Analysis: {len(mood_counts)} mood categories detected")
        return True, mood_classified
        
    except Exception as e:
        logger.error(f"❌ Mood Analysis failed: {e}")
        return False, None

def test_feature_extraction():
    """Test feature extraction system."""
    logger.info("🔧 Testing Feature Extraction...")
    
    try:
        from features.feature_extractor import FeatureExtractor
        
        # Create test data
        test_data = pd.DataFrame({
            'track_id': ['track_1', 'track_2', 'track_3'],
            'duration_ms': [180000, 210000, 195000],
            'popularity': [80, 60, 90],
            'acousticness': [0.2, 0.8, 0.3],
            'danceability': [0.7, 0.4, 0.9],
            'energy': [0.8, 0.3, 0.9],
            'instrumentalness': [0.1, 0.9, 0.0],
            'liveness': [0.2, 0.1, 0.3],
            'loudness': [-5, -12, -4],
            'speechiness': [0.1, 0.05, 0.15],
            'tempo': [120, 80, 140],
            'valence': [0.8, 0.2, 0.9],
            'artists': [['Artist 1'], ['Artist 2'], ['Artist 3']]
        })
        
        extractor = FeatureExtractor()
        
        # Test feature extraction
        engineered_df, engineered_columns = extractor.extract_all_features(test_data)
        
        assert len(engineered_columns) > 20, "Feature extraction failed: too few engineered features"
        assert 'happiness_index' in engineered_df.columns, "Composite features failed"
        assert 'tempo_normalized' in engineered_df.columns, "Basic features failed"
        
        logger.info(f"✅ Feature Extraction: {len(engineered_columns)} engineered features")
        return True, engineered_df, engineered_columns
        
    except Exception as e:
        logger.error(f"❌ Feature Extraction failed: {e}")
        return False, None, None

def test_taste_modeling():
    """Test taste modeling system."""
    logger.info("🤖 Testing Taste Modeling...")
    
    try:
        from models.taste_model import TasteModel
        
        # Create test data with features and labels
        np.random.seed(42)
        n_samples = 100
        
        test_data = pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(n_samples)],
            'acousticness': np.random.random(n_samples),
            'danceability': np.random.random(n_samples),
            'energy': np.random.random(n_samples),
            'valence': np.random.random(n_samples),
            'tempo': np.random.uniform(60, 200, n_samples),
            'loudness': np.random.uniform(-60, 0, n_samples),
            'popularity': np.random.randint(0, 100, n_samples),
            'liked': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        })
        
        feature_columns = ['acousticness', 'danceability', 'energy', 'valence', 'tempo', 'loudness', 'popularity']
        
        taste_model = TasteModel()
        
        # Test baseline model training
        X, y = taste_model.prepare_training_data(test_data, feature_columns, 'liked')
        metrics = taste_model.train_baseline_model(X, y)
        
        # Test predictions
        predictions = taste_model.predict_preferences(test_data)
        
        # Test model summary
        summary = taste_model.get_model_summary()
        
        assert metrics['accuracy'] > 0.5, "Model training failed: poor accuracy"
        assert 'preference_score' in predictions.columns, "Prediction failed"
        assert summary['status'] == 'trained', "Model summary failed"
        
        logger.info(f"✅ Taste Modeling: {metrics['accuracy']:.3f} accuracy, {summary['feature_count']} features")
        return True, taste_model, predictions
        
    except Exception as e:
        logger.error(f"❌ Taste Modeling failed: {e}")
        return False, None, None

def test_recommendation_engine():
    """Test recommendation engine."""
    logger.info("🎵 Testing Recommendation Engine...")
    
    try:
        from models.recommender import PlaylistRecommender
        
        # Create test data
        np.random.seed(42)
        n_tracks = 200
        
        test_data = pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(n_tracks)],
            'name': [f'Song {i}' for i in range(n_tracks)],
            'artists': [[f'Artist {i}'] for i in range(n_tracks)],
            'acousticness': np.random.random(n_tracks),
            'danceability': np.random.random(n_tracks),
            'energy': np.random.random(n_tracks),
            'valence': np.random.random(n_tracks),
            'tempo': np.random.uniform(60, 200, n_tracks),
            'popularity': np.random.randint(0, 100, n_tracks),
            'primary_mood': np.random.choice(['happy', 'sad', 'energetic', 'calm'], n_tracks),
            'preference_score': np.random.random(n_tracks)
        })
        
        recommender = PlaylistRecommender()
        
        # Test mood-based recommendations
        mood_recs = recommender.get_mood_based_recommendations(
            test_data, target_mood='happy', n_recommendations=20
        )
        
        # Test playlist generation
        playlist = recommender.generate_playlist(
            test_data, 
            target_mood='energetic',
            energy_range=(0.6, 1.0),
            playlist_size=15
        )
        
        # Test diversity calculation
        diversity_score = recommender.calculate_playlist_diversity(playlist)
        
        assert len(mood_recs) <= 20, "Mood recommendations failed"
        assert len(playlist) <= 15, "Playlist generation failed"
        assert 0 <= diversity_score <= 1, "Diversity calculation failed"
        
        logger.info(f"✅ Recommendation Engine: {len(playlist)} tracks, {diversity_score:.3f} diversity")
        return True, playlist, diversity_score
        
    except Exception as e:
        logger.error(f"❌ Recommendation Engine failed: {e}")
        return False, None, None

def test_evaluation_system():
    """Test evaluation system."""
    logger.info("📊 Testing Evaluation System...")
    
    try:
        from evaluation.metrics import RecommendationEvaluator
        
        # Create test data
        np.random.seed(42)
        n_tracks = 50
        
        test_recommendations = pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(n_tracks)],
            'preference_score': np.random.random(n_tracks),
            'artists': [[f'Artist {i % 10}'] for i in range(n_tracks)],  # 10 unique artists
            'primary_mood': np.random.choice(['happy', 'sad', 'energetic'], n_tracks)
        })
        
        test_ground_truth = pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(n_tracks)],
            'actual_preference': np.random.choice([0, 1], n_tracks, p=[0.4, 0.6])
        })
        
        evaluator = RecommendationEvaluator()
        
        # Test precision@k
        precision_5 = evaluator.precision_at_k(test_recommendations, test_ground_truth, k=5)
        precision_10 = evaluator.precision_at_k(test_recommendations, test_ground_truth, k=10)
        
        # Test diversity metrics
        artist_diversity = evaluator.calculate_artist_diversity(test_recommendations)
        mood_diversity = evaluator.calculate_mood_diversity(test_recommendations)
        
        # Test comprehensive evaluation
        eval_results = evaluator.evaluate_recommendations(
            test_recommendations, test_ground_truth, k_values=[5, 10]
        )
        
        assert 0 <= precision_5 <= 1, "Precision@5 calculation failed"
        assert 0 <= artist_diversity <= 1, "Artist diversity calculation failed"
        assert 'precision_at_5' in eval_results, "Comprehensive evaluation failed"
        
        logger.info(f"✅ Evaluation System: P@5={precision_5:.3f}, Diversity={artist_diversity:.3f}")
        return True, eval_results
        
    except Exception as e:
        logger.error(f"❌ Evaluation System failed: {e}")
        return False, None

def test_system_integration():
    """Test end-to-end system integration."""
    logger.info("🔗 Testing System Integration...")
    
    try:
        # Create comprehensive test dataset
        np.random.seed(42)
        n_tracks = 100
        
        # Simulate realistic music data
        test_tracks = pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(n_tracks)],
            'name': [f'Song {i}' for i in range(n_tracks)],
            'artists': [[f'Artist {i % 20}'] for i in range(n_tracks)],  # 20 unique artists
            'album': [f'Album {i % 30}' for i in range(n_tracks)],  # 30 unique albums
            'popularity': np.random.randint(10, 100, n_tracks),
            'duration_ms': np.random.randint(120000, 300000, n_tracks),
            'explicit': np.random.choice([True, False], n_tracks, p=[0.2, 0.8]),
            'user_preference': np.random.random(n_tracks)
        })
        
        test_features = pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(n_tracks)],
            'acousticness': np.random.random(n_tracks),
            'danceability': np.random.random(n_tracks),
            'energy': np.random.random(n_tracks),
            'instrumentalness': np.random.random(n_tracks),
            'liveness': np.random.random(n_tracks),
            'loudness': np.random.uniform(-60, 0, n_tracks),
            'speechiness': np.random.random(n_tracks),
            'tempo': np.random.uniform(60, 200, n_tracks),
            'valence': np.random.random(n_tracks)
        })
        
        # Test complete pipeline
        from data.data_processor import DataProcessor
        from features.mood_analyzer import MoodAnalyzer
        from models.recommender import PlaylistRecommender
        from evaluation.metrics import RecommendationEvaluator
        
        # 1. Process data
        processor = DataProcessor()
        processed_df, feature_columns, stats = processor.process_full_pipeline(test_tracks, test_features)
        
        # 2. Analyze moods
        mood_analyzer = MoodAnalyzer()
        mood_df = mood_analyzer.calculate_mood_scores(processed_df)
        mood_df = mood_analyzer.classify_primary_mood(mood_df)
        
        # 3. Generate recommendations
        recommender = PlaylistRecommender()
        playlist = recommender.generate_playlist(
            mood_df,
            target_mood='happy',
            energy_range=(0.5, 1.0),
            valence_range=(0.6, 1.0),
            playlist_size=20
        )
        
        # 4. Evaluate results
        evaluator = RecommendationEvaluator()
        diversity = evaluator.calculate_artist_diversity(playlist)
        
        # Validate integration
        assert len(processed_df) == n_tracks, "Data processing integration failed"
        assert 'primary_mood' in mood_df.columns, "Mood analysis integration failed"
        assert len(playlist) <= 20, "Recommendation integration failed"
        assert 0 <= diversity <= 1, "Evaluation integration failed"
        
        integration_results = {
            'total_tracks': len(processed_df),
            'features_engineered': len(feature_columns),
            'playlist_size': len(playlist),
            'diversity_score': diversity,
            'mood_distribution': mood_df['primary_mood'].value_counts().to_dict()
        }
        
        logger.info(f"✅ System Integration: {len(processed_df)} tracks → {len(playlist)} playlist")
        return True, integration_results
        
    except Exception as e:
        logger.error(f"❌ System Integration failed: {e}")
        return False, None

def run_comprehensive_test():
    """Run all tests and generate comprehensive report."""
    logger.info("🚀 Starting Comprehensive Test Suite...")
    logger.info("=" * 60)
    
    test_results = {}
    start_time = datetime.now()
    
    # Run all tests
    tests = [
        ("Data Processing", test_data_processing),
        ("Mood Analysis", test_mood_analysis),
        ("Feature Extraction", test_feature_extraction),
        ("Taste Modeling", test_taste_modeling),
        ("Recommendation Engine", test_recommendation_engine),
        ("Evaluation System", test_evaluation_system),
        ("System Integration", test_system_integration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                success = result[0]
                test_results[test_name] = {'status': 'PASSED' if success else 'FAILED', 'data': result[1:]}
            else:
                success = result
                test_results[test_name] = {'status': 'PASSED' if success else 'FAILED', 'data': None}
            
            if success:
                passed_tests += 1
                
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            test_results[test_name] = {'status': 'CRASHED', 'error': str(e)}
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Generate final report
    logger.info("=" * 60)
    logger.info("📋 COMPREHENSIVE TEST REPORT")
    logger.info("=" * 60)
    logger.info(f"⏱️  Test Duration: {duration:.2f} seconds")
    logger.info(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    logger.info(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    logger.info("")
    
    for test_name, result in test_results.items():
        status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
        logger.info(f"{status_emoji} {test_name}: {result['status']}")
    
    logger.info("=" * 60)
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! System is fully functional.")
    else:
        logger.info(f"⚠️  {total_tests - passed_tests} tests failed. Review issues above.")
    
    logger.info("=" * 60)
    
    return test_results, passed_tests, total_tests

if __name__ == "__main__":
    # Ensure we're in the right directory
    if not os.path.exists('src'):
        logger.error("❌ Please run this script from the playlist-auto-dj directory")
        sys.exit(1)
    
    # Run comprehensive tests
    results, passed, total = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)
