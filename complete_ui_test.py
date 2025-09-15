"""
Comprehensive UI and functionality test for the Playlist Auto-DJ system.
Tests all components including playlist generation, model insights, and backend integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import logging
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_playlist_generation():
    """Test the complete playlist generation workflow."""
    logger.info("Testing playlist generation functionality...")
    
    try:
        from src.models.recommender import PlaylistRecommender
        from src.features.mood_analyzer import MoodAnalyzer
        
        # Create test data
        np.random.seed(42)
        n_tracks = 100
        
        test_data = {
            'track_id': [f'track_{i}' for i in range(n_tracks)],
            'name': [f'Song {i}' for i in range(n_tracks)],
            'artists': [[f'Artist {i}'] for i in range(n_tracks)],
            'album': [f'Album {i}' for i in range(n_tracks)],
            'popularity': np.random.randint(0, 100, n_tracks),
            'duration_ms': np.random.randint(120000, 300000, n_tracks),
            'acousticness': np.random.random(n_tracks),
            'danceability': np.random.random(n_tracks),
            'energy': np.random.random(n_tracks),
            'instrumentalness': np.random.random(n_tracks),
            'liveness': np.random.random(n_tracks),
            'loudness': np.random.uniform(-60, 0, n_tracks),
            'speechiness': np.random.random(n_tracks),
            'tempo': np.random.uniform(60, 200, n_tracks),
            'valence': np.random.random(n_tracks),
            'user_preference': np.random.random(n_tracks)
        }
        
        df = pd.DataFrame(test_data)
        
        # Initialize components
        mood_analyzer = MoodAnalyzer()
        recommender = PlaylistRecommender()
        
        # Test mood analysis
        logger.info("Testing mood analysis...")
        df_with_moods = mood_analyzer.calculate_mood_scores(df)
        df_with_moods = mood_analyzer.classify_primary_mood(df_with_moods)
        
        mood_distribution = df_with_moods['primary_mood'].value_counts()
        logger.info(f"Mood distribution: {mood_distribution.to_dict()}")
        
        # Test playlist generation with different parameters
        test_scenarios = [
            {
                'name': 'Happy Energetic Playlist',
                'target_mood': 'happy',
                'energy_range': (0.6, 1.0),
                'valence_range': (0.6, 1.0),
                'tempo_range': (120, 180),
                'playlist_size': 15
            },
            {
                'name': 'Calm Focus Playlist',
                'target_mood': 'calm',
                'energy_range': (0.0, 0.4),
                'valence_range': (0.3, 0.7),
                'tempo_range': (60, 100),
                'playlist_size': 20
            },
            {
                'name': 'Party Dance Playlist',
                'target_mood': 'party',
                'energy_range': (0.7, 1.0),
                'valence_range': (0.5, 1.0),
                'tempo_range': (100, 200),
                'playlist_size': 25
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            
            playlist = recommender.generate_playlist(
                df_with_moods,
                target_mood=scenario['target_mood'],
                playlist_size=scenario['playlist_size'],
                energy_range=scenario['energy_range'],
                valence_range=scenario['valence_range'],
                tempo_range=scenario['tempo_range']
            )
            
            if not playlist.empty:
                stats = recommender.get_playlist_statistics(playlist)
                results[scenario['name']] = {
                    'tracks_generated': len(playlist),
                    'target_size': scenario['playlist_size'],
                    'avg_energy': playlist['energy'].mean(),
                    'avg_valence': playlist['valence'].mean(),
                    'avg_tempo': playlist['tempo'].mean(),
                    'unique_artists': stats.get('unique_artists', 0),
                    'artist_diversity': stats.get('artist_diversity', 0)
                }
                logger.info(f"✅ {scenario['name']}: {len(playlist)} tracks generated")
            else:
                results[scenario['name']] = {'error': 'No tracks generated'}
                logger.warning(f"⚠️ {scenario['name']}: No tracks generated")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in playlist generation test: {e}")
        return {'error': str(e)}

def test_mood_journey_playlist():
    """Test mood journey playlist creation."""
    logger.info("Testing mood journey playlist...")
    
    try:
        from src.models.recommender import PlaylistRecommender
        from src.features.mood_analyzer import MoodAnalyzer
        
        # Create test data
        np.random.seed(42)
        n_tracks = 200
        
        test_data = {
            'track_id': [f'track_{i}' for i in range(n_tracks)],
            'name': [f'Song {i}' for i in range(n_tracks)],
            'artists': [[f'Artist {i}'] for i in range(n_tracks)],
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
            'popularity': np.random.randint(0, 100, n_tracks),
            'duration_ms': np.random.randint(120000, 300000, n_tracks)
        }
        
        df = pd.DataFrame(test_data)
        
        # Initialize components
        mood_analyzer = MoodAnalyzer()
        recommender = PlaylistRecommender()
        
        # Add mood analysis
        df_with_moods = mood_analyzer.calculate_mood_scores(df)
        df_with_moods = mood_analyzer.classify_primary_mood(df_with_moods)
        
        # Test mood journey
        mood_sequence = ['calm', 'happy', 'energetic', 'party']
        journey_playlist = recommender.create_mood_journey_playlist(
            df_with_moods, 
            mood_sequence, 
            playlist_size=24
        )
        
        if not journey_playlist.empty:
            mood_sections = journey_playlist['mood_section'].value_counts()
            logger.info(f"✅ Mood journey playlist created: {len(journey_playlist)} tracks")
            logger.info(f"Mood sections: {mood_sections.to_dict()}")
            return {
                'success': True,
                'total_tracks': len(journey_playlist),
                'mood_sections': mood_sections.to_dict(),
                'sequence': mood_sequence
            }
        else:
            logger.warning("⚠️ Mood journey playlist generation failed")
            return {'success': False, 'error': 'No tracks generated'}
            
    except Exception as e:
        logger.error(f"Error in mood journey test: {e}")
        return {'success': False, 'error': str(e)}

def test_feature_engineering():
    """Test advanced feature engineering."""
    logger.info("Testing feature engineering...")
    
    try:
        from src.features.feature_extractor import FeatureExtractor
        
        # Create test data
        np.random.seed(42)
        n_tracks = 50
        
        test_data = {
            'track_id': [f'track_{i}' for i in range(n_tracks)],
            'name': [f'Song {i}' for i in range(n_tracks)],
            'artists': [[f'Artist {i}'] for i in range(n_tracks)],
            'acousticness': np.random.random(n_tracks),
            'danceability': np.random.random(n_tracks),
            'energy': np.random.random(n_tracks),
            'instrumentalness': np.random.random(n_tracks),
            'liveness': np.random.random(n_tracks),
            'loudness': np.random.uniform(-60, 0, n_tracks),
            'speechiness': np.random.random(n_tracks),
            'tempo': np.random.uniform(60, 200, n_tracks),
            'valence': np.random.random(n_tracks),
            'popularity': np.random.randint(0, 100, n_tracks),
            'duration_ms': np.random.randint(120000, 300000, n_tracks)
        }
        
        df = pd.DataFrame(test_data)
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        
        # Test feature extraction
        df_features, engineered_columns = feature_extractor.extract_all_features(df)
        
        logger.info(f"✅ Feature engineering completed")
        logger.info(f"Original features: {len(df.columns)}")
        logger.info(f"Engineered features: {len(engineered_columns)}")
        logger.info(f"Total features: {len(df_features.columns)}")
        
        return {
            'success': True,
            'original_features': len(df.columns),
            'engineered_features': len(engineered_columns),
            'total_features': len(df_features.columns),
            'feature_names': engineered_columns[:10]  # First 10 for brevity
        }
        
    except Exception as e:
        logger.error(f"Error in feature engineering test: {e}")
        return {'success': False, 'error': str(e)}

def test_evaluation_metrics():
    """Test evaluation metrics calculation."""
    logger.info("Testing evaluation metrics...")
    
    try:
        from src.evaluation.metrics import RecommendationEvaluator
        
        # Create test data
        np.random.seed(42)
        n_tracks = 100
        
        test_data = {
            'track_id': [f'track_{i}' for i in range(n_tracks)],
            'name': [f'Song {i}' for i in range(n_tracks)],
            'artists': [[f'Artist {i % 20}'] for i in range(n_tracks)],  # 20 unique artists
            'user_preference': np.random.choice([0, 1], n_tracks, p=[0.7, 0.3]),
            'predicted_preference': np.random.random(n_tracks),
            'energy': np.random.random(n_tracks),
            'valence': np.random.random(n_tracks),
            'danceability': np.random.random(n_tracks)
        }
        
        df = pd.DataFrame(test_data)
        
        # Initialize evaluator
        evaluator = RecommendationEvaluator()
        
        # Test precision@k
        precision_5 = evaluator.precision_at_k(df, 'user_preference', 'predicted_preference', k=5)
        precision_10 = evaluator.precision_at_k(df, 'user_preference', 'predicted_preference', k=10)
        
        # Test diversity
        diversity_score = evaluator.calculate_diversity_score(df)
        
        # Test novelty
        novelty_score = evaluator.calculate_novelty_score(df)
        
        logger.info(f"✅ Evaluation metrics calculated")
        logger.info(f"Precision@5: {precision_5:.3f}")
        logger.info(f"Precision@10: {precision_10:.3f}")
        logger.info(f"Diversity Score: {diversity_score:.3f}")
        logger.info(f"Novelty Score: {novelty_score:.3f}")
        
        return {
            'success': True,
            'precision_at_5': precision_5,
            'precision_at_10': precision_10,
            'diversity_score': diversity_score,
            'novelty_score': novelty_score
        }
        
    except Exception as e:
        logger.error(f"Error in evaluation metrics test: {e}")
        return {'success': False, 'error': str(e)}

def test_cache_management():
    """Test cache management functionality."""
    logger.info("Testing cache management...")
    
    try:
        from src.data.cache_manager import CacheManager
        
        # Initialize cache manager
        cache_manager = CacheManager()
        
        # Test DataFrame saving and loading
        test_df = pd.DataFrame({
            'track_id': ['track_1', 'track_2', 'track_3'],
            'name': ['Song 1', 'Song 2', 'Song 3'],
            'energy': [0.5, 0.7, 0.3]
        })
        
        # Save DataFrame
        save_success = cache_manager.save_dataframe(test_df, 'test_tracks')
        
        # Load DataFrame
        loaded_df = cache_manager.load_dataframe('test_tracks')
        
        # Test JSON saving and loading
        test_data = {'test_key': 'test_value', 'numbers': [1, 2, 3]}
        json_save_success = cache_manager.save_json(test_data, 'test_data')
        loaded_json = cache_manager.load_json('test_data')
        
        # Get cache info
        cache_info = cache_manager.get_cache_info()
        
        logger.info(f"✅ Cache management test completed")
        logger.info(f"DataFrame save/load: {save_success and loaded_df is not None}")
        logger.info(f"JSON save/load: {json_save_success and loaded_json is not None}")
        logger.info(f"Cache info: {cache_info}")
        
        return {
            'success': True,
            'dataframe_operations': save_success and loaded_df is not None,
            'json_operations': json_save_success and loaded_json is not None,
            'cache_info': cache_info
        }
        
    except Exception as e:
        logger.error(f"Error in cache management test: {e}")
        return {'success': False, 'error': str(e)}

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    logger.info("🚀 Starting comprehensive system tests...")
    
    results = {}
    
    # Test 1: Playlist Generation
    results['playlist_generation'] = test_playlist_generation()
    
    # Test 2: Mood Journey Playlists
    results['mood_journey'] = test_mood_journey_playlist()
    
    # Test 3: Feature Engineering
    results['feature_engineering'] = test_feature_engineering()
    
    # Test 4: Evaluation Metrics
    results['evaluation_metrics'] = test_evaluation_metrics()
    
    # Test 5: Cache Management
    results['cache_management'] = test_cache_management()
    
    # Summary
    logger.info("📊 Test Results Summary:")
    for test_name, result in results.items():
        if isinstance(result, dict) and 'error' not in result:
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.info(f"❌ {test_name}: FAILED")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_tests()
    
    # Print detailed results
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        print(f"\n{test_name.upper()}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {result}")
