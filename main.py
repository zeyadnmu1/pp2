"""
Main entry point for the Playlist Auto-DJ system.
Provides command-line interface for various operations.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.spotify_client import SpotifyClient
from src.data.data_processor import DataProcessor
from src.data.cache_manager import CacheManager
from src.features.mood_analyzer import MoodAnalyzer
from src.features.feature_extractor import FeatureExtractor
from src.models.taste_model import TasteModel
from src.models.recommender import PlaylistRecommender
from src.evaluation.metrics import RecommendationEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_data_pipeline():
    """Set up and run the complete data processing pipeline."""
    logger.info("Setting up data pipeline...")
    
    try:
        # Initialize components
        spotify_client = SpotifyClient()
        data_processor = DataProcessor()
        cache_manager = CacheManager()
        
        # Fetch user data
        logger.info("Fetching user data from Spotify...")
        user_data = spotify_client.get_comprehensive_dataset()
        
        if user_data.empty:
            logger.error("No user data found. Please check your Spotify connection.")
            return False
        
        # Get audio features
        logger.info("Fetching audio features...")
        track_ids = user_data['track_id'].tolist()
        audio_features = spotify_client.get_audio_features(track_ids)
        features_df = pd.DataFrame(audio_features)
        
        # Process data
        logger.info("Processing data...")
        processed_data, feature_columns, stats = data_processor.process_full_pipeline(
            user_data, features_df
        )
        
        # Save processed data
        logger.info("Saving processed data...")
        cache_manager.save_processed_dataset(processed_data, feature_columns, stats)
        
        logger.info(f"Data pipeline completed successfully!")
        logger.info(f"Processed {len(processed_data)} tracks with {len(feature_columns)} features")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data pipeline: {e}")
        return False


def train_models():
    """Train the taste model and set up the recommendation system."""
    logger.info("Training models...")
    
    try:
        # Load processed data
        cache_manager = CacheManager()
        data_result = cache_manager.load_processed_dataset()
        
        if data_result is None:
            logger.error("No processed data found. Please run data pipeline first.")
            return False
        
        processed_data, feature_columns, stats = data_result
        
        # Initialize and train taste model
        taste_model = TasteModel()
        
        # Prepare training data
        X, y = taste_model.prepare_training_data(processed_data, feature_columns)
        
        # Train LightGBM model
        logger.info("Training LightGBM taste model...")
        metrics = taste_model.train_lightgbm_model(X, y)
        
        # Save model
        model_path = "data/models/taste_model_lightgbm.pkl"
        taste_model.save_model(model_path, metadata={"training_metrics": metrics})
        
        logger.info(f"Model training completed!")
        logger.info(f"Model AUC: {metrics['auc']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return False


def generate_sample_playlist():
    """Generate a sample playlist to test the system."""
    logger.info("Generating sample playlist...")
    
    try:
        # Load components
        cache_manager = CacheManager()
        data_result = cache_manager.load_processed_dataset()
        
        if data_result is None:
            logger.error("No processed data found. Please run data pipeline first.")
            return False
        
        processed_data, feature_columns, stats = data_result
        
        # Load taste model
        taste_model = TasteModel()
        model_path = "data/models/taste_model_lightgbm.pkl"
        
        if not taste_model.load_model(model_path):
            logger.error("Could not load taste model. Please train models first.")
            return False
        
        # Set up recommender
        mood_analyzer = MoodAnalyzer()
        recommender = PlaylistRecommender()
        recommender.set_taste_model(taste_model)
        recommender.set_mood_analyzer(mood_analyzer)
        recommender.set_track_features(processed_data)
        
        # Generate playlist
        logger.info("Generating happy playlist...")
        result = recommender.generate_playlist(
            target_mood="happy",
            playlist_size=20,
            energy_range=(0.5, 1.0),
            valence_range=(0.6, 1.0)
        )
        
        if not result['playlist'].empty:
            playlist = result['playlist']
            metadata = result['metadata']
            
            logger.info(f"Generated playlist with {len(playlist)} tracks")
            logger.info(f"Average valence: {metadata.get('avg_valence', 0):.2f}")
            logger.info(f"Average energy: {metadata.get('avg_energy', 0):.2f}")
            logger.info(f"Diversity score: {metadata.get('diversity_score', 0):.2f}")
            
            # Save playlist
            playlist_path = "data/sample_playlist.csv"
            playlist[['name', 'artists', 'valence', 'energy', 'danceability']].to_csv(
                playlist_path, index=False
            )
            logger.info(f"Playlist saved to {playlist_path}")
            
            return True
        else:
            logger.warning("No tracks found for playlist generation")
            return False
        
    except Exception as e:
        logger.error(f"Error generating playlist: {e}")
        return False


def run_evaluation():
    """Run evaluation on the recommendation system."""
    logger.info("Running evaluation...")
    
    try:
        # Load data and models (similar to generate_sample_playlist)
        cache_manager = CacheManager()
        data_result = cache_manager.load_processed_dataset()
        
        if data_result is None:
            logger.error("No processed data found.")
            return False
        
        processed_data, feature_columns, stats = data_result
        
        # Create sample user preferences
        user_preferences = {}
        for _, track in processed_data.head(100).iterrows():
            # Use user_preference as ground truth
            user_preferences[track['track_id']] = track.get('user_preference', 0.5)
        
        # Generate recommendations
        taste_model = TasteModel()
        if taste_model.load_model("data/models/taste_model_lightgbm.pkl"):
            recommender = PlaylistRecommender()
            recommender.set_taste_model(taste_model)
            recommender.set_track_features(processed_data)
            
            result = recommender.generate_playlist(
                target_mood="happy",
                playlist_size=20
            )
            
            if not result['playlist'].empty:
                # Evaluate recommendations
                evaluator = RecommendationEvaluator()
                evaluation_results = evaluator.evaluate_recommendations(
                    result['playlist'], user_preferences
                )
                
                # Print results
                logger.info("Evaluation Results:")
                logger.info(f"Precision@10: {evaluation_results.get('precision_at_10', 0):.3f}")
                logger.info(f"Recall@10: {evaluation_results.get('recall_at_10', 0):.3f}")
                logger.info(f"Diversity Score: {evaluation_results.get('diversity_score', 0):.3f}")
                logger.info(f"Novelty Score: {evaluation_results.get('novelty_score', 0):.3f}")
                
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return False


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Playlist Auto-DJ - AI-powered music recommendations")
    
    parser.add_argument(
        "command",
        choices=["setup", "train", "playlist", "evaluate", "web"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    success = False
    
    if args.command == "setup":
        success = setup_data_pipeline()
    elif args.command == "train":
        success = train_models()
    elif args.command == "playlist":
        success = generate_sample_playlist()
    elif args.command == "evaluate":
        success = run_evaluation()
    elif args.command == "web":
        logger.info("Starting web application...")
        os.system("streamlit run src/app/streamlit_app.py")
        success = True
    
    if success:
        logger.info(f"Command '{args.command}' completed successfully!")
        sys.exit(0)
    else:
        logger.error(f"Command '{args.command}' failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
