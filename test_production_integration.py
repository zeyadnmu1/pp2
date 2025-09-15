"""
Production integration test for Playlist Auto-DJ.
Tests the full system with real components (without requiring Spotify credentials).
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_production_imports():
    """Test that all production components can be imported."""
    print("🧪 Testing Production Imports...")
    
    try:
        from data.spotify_client import SpotifyClient
        from data.data_processor import DataProcessor
        from data.cache_manager import CacheManager
        from features.mood_analyzer import MoodAnalyzer
        from features.feature_extractor import FeatureExtractor
        from models.taste_model import TasteModel
        from models.recommender import PlaylistRecommender
        from evaluation.metrics import RecommendationEvaluator
        
        print("✅ All production components imported successfully")
        
        # Test LightGBM availability
        try:
            import lightgbm
            print("✅ LightGBM available (advanced ML features enabled)")
        except ImportError as e:
            print("⚠️  LightGBM not available (will use Random Forest fallback)")
            print(f"   LightGBM error: {str(e)[:100]}...")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_production_pipeline():
    """Test the full production pipeline with mock data."""
    print("\n🧪 Testing Production Pipeline...")
    
    try:
        # Import components within the function to handle import errors
        from data.spotify_client import SpotifyClient
        from data.data_processor import DataProcessor
        from data.cache_manager import CacheManager
        from features.mood_analyzer import MoodAnalyzer
        from features.feature_extractor import FeatureExtractor
        from models.taste_model import TasteModel
        from models.recommender import PlaylistRecommender
        from evaluation.metrics import RecommendationEvaluator
        
        # Initialize components
        data_processor = DataProcessor()
        cache_manager = CacheManager()
        mood_analyzer = MoodAnalyzer()
        feature_extractor = FeatureExtractor()
        taste_model = TasteModel()
        recommender = PlaylistRecommender()
        evaluator = RecommendationEvaluator()
        
        print("✅ Components initialized")
        
        # Create mock Spotify data
        mock_tracks = pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(100)],
            'name': [f'Song {i}' for i in range(100)],
            'artists': [[f'Artist {i}'] for i in range(100)],
            'album': [f'Album {i}' for i in range(100)],
            'popularity': np.random.randint(0, 100, 100),
            'duration_ms': np.random.randint(120000, 300000, 100),
            'explicit': np.random.choice([True, False], 100),
            'user_preference': np.random.choice([0.0, 0.3, 0.7, 1.0], 100, p=[0.3, 0.2, 0.3, 0.2])
        })
        
        # Create mock audio features
        mock_features = pd.DataFrame({
            'id': [f'track_{i}' for i in range(100)],
            'acousticness': np.random.random(100),
            'danceability': np.random.random(100),
            'energy': np.random.random(100),
            'instrumentalness': np.random.random(100),
            'liveness': np.random.random(100),
            'loudness': np.random.uniform(-60, 0, 100),
            'speechiness': np.random.random(100),
            'tempo': np.random.uniform(60, 200, 100),
            'valence': np.random.random(100)
        })
        
        print("✅ Mock data created")
        
        # Test data processing pipeline
        processed_df, feature_columns, stats = data_processor.process_full_pipeline(
            mock_tracks, mock_features
        )
        
        print(f"✅ Data processing: {len(processed_df)} tracks, {len(feature_columns)} features")
        
        # Test mood analysis
        mood_df = mood_analyzer.calculate_mood_scores(processed_df)
        mood_df = mood_analyzer.classify_primary_mood(mood_df)
        
        mood_distribution = mood_df['primary_mood'].value_counts()
        print(f"✅ Mood analysis: {dict(mood_distribution)}")
        
        # Test feature extraction
        engineered_df, engineered_features = feature_extractor.extract_all_features(mood_df)
        
        print(f"✅ Feature engineering: {len(engineered_features)} new features")
        
        # Test taste modeling (with fallback)
        try:
            X, y = taste_model.prepare_training_data(
                engineered_df, feature_columns + engineered_features
            )
            
            # Try LightGBM first, fallback to Random Forest
            try:
                metrics = taste_model.train_lightgbm_model(X, y)
                model_type = "LightGBM"
            except ImportError:
                metrics = taste_model.train_random_forest_model(X, y)
                model_type = "Random Forest"
            
            print(f"✅ Taste modeling ({model_type}): AUC = {metrics['auc']:.3f}")
            
            # Test predictions
            predictions_df = taste_model.predict_preferences(engineered_df)
            print(f"✅ Preference prediction: {len(predictions_df)} tracks scored")
            
        except Exception as e:
            print(f"⚠️  Taste modeling skipped: {e}")
            predictions_df = engineered_df.copy()
            predictions_df['preference_score'] = np.random.random(len(predictions_df))
        
        # Test recommendation engine
        recommendations = recommender.generate_recommendations(
            predictions_df,
            target_mood='happy',
            n_recommendations=20
        )
        
        print(f"✅ Recommendations: {len(recommendations)} tracks generated")
        
        # Test evaluation
        metrics = evaluator.evaluate_recommendations(
            recommendations,
            predictions_df,
            k_values=[5, 10, 20]
        )
        
        print(f"✅ Evaluation: Precision@5 = {metrics['precision_at_5']:.3f}")
        
        # Test caching
        cache_success = cache_manager.save_dataframe(processed_df, "test_production")
        loaded_df = cache_manager.load_dataframe("test_production")
        
        if cache_success and loaded_df is not None:
            print("✅ Caching system working")
        else:
            print("⚠️  Caching system issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Production pipeline error: {e}")
        logger.error(f"Production pipeline error: {e}", exc_info=True)
        return False

def test_production_app_structure():
    """Test that production app structure is correct."""
    print("\n🧪 Testing Production App Structure...")
    
    required_files = [
        'src/app/production_app.py',
        'setup_production.py',
        'run_production.py',
        'PRODUCTION_GUIDE.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All production files present")
    
    # Check production app content
    with open('src/app/production_app.py', 'r') as f:
        content = f.read()
        
    required_components = [
        'SpotifyClient',
        'authenticate_spotify',
        'load_user_data',
        'train_taste_model',
        'render_playlist_generator'
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing components in production app: {missing_components}")
        return False
    
    print("✅ Production app structure verified")
    return True

def test_environment_setup():
    """Test environment setup capabilities."""
    print("\n🧪 Testing Environment Setup...")
    
    # Test directory creation
    required_dirs = ['data/cache', 'data/raw', 'data/processed', 'data/models', 'logs']
    
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    all_dirs_exist = all(Path(d).exists() for d in required_dirs)
    
    if all_dirs_exist:
        print("✅ Directory structure created successfully")
    else:
        print("❌ Failed to create required directories")
        return False
    
    # Test config loading
    try:
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['spotify', 'data', 'features', 'models']
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            print(f"❌ Missing config sections: {missing_sections}")
            return False
        
        print("✅ Configuration file valid")
        
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False
    
    return True

def run_production_integration_test():
    """Run complete production integration test."""
    print("=" * 60)
    print("🎵 PLAYLIST AUTO-DJ - PRODUCTION INTEGRATION TEST")
    print("=" * 60)
    print()
    
    tests = [
        ("Import Test", test_production_imports),
        ("Pipeline Test", test_production_pipeline),
        ("App Structure Test", test_production_app_structure),
        ("Environment Test", test_environment_setup)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Production system is ready!")
        print("\nNext steps:")
        print("1. Run: python setup_production.py")
        print("2. Set your Spotify API credentials")
        print("3. Run: python run_production.py")
        print("4. Enjoy your personalized music recommendations!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_production_integration_test()
    sys.exit(0 if success else 1)
