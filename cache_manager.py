"""
Cache management system for efficient data storage and retrieval.
Handles caching of Spotify data, processed features, and model artifacts.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
import hashlib
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for Spotify data and processed features."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize cache manager with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.cache_dir = Path(self.config['data']['cache_dir'])
        self.raw_data_dir = Path(self.config['data']['raw_data_dir'])
        self.processed_data_dir = Path(self.config['data']['processed_data_dir'])
        self.models_dir = Path(self.config['data']['models_dir'])
        self.cache_expiry_hours = self.config['data']['cache_expiry_hours']
        
        # Create directories if they don't exist
        for directory in [self.cache_dir, self.raw_data_dir, 
                         self.processed_data_dir, self.models_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(self, data: Union[str, Dict, List]) -> str:
        """Generate a unique cache key from data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, filepath: Path) -> bool:
        """Check if cached file is still valid based on expiry time."""
        if not filepath.exists():
            return False
        
        file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
        
        return file_time > expiry_time
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, 
                      directory: str = "processed") -> bool:
        """Save DataFrame to parquet format with compression."""
        try:
            if directory == "raw":
                filepath = self.raw_data_dir / f"{filename}.parquet"
            elif directory == "processed":
                filepath = self.processed_data_dir / f"{filename}.parquet"
            else:
                filepath = self.cache_dir / f"{filename}.parquet"
            
            df.to_parquet(filepath, compression='snappy', index=False)
            logger.info(f"Saved DataFrame to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to {filename}: {e}")
            return False
    
    def load_dataframe(self, filename: str, 
                      directory: str = "processed") -> Optional[pd.DataFrame]:
        """Load DataFrame from parquet format."""
        try:
            if directory == "raw":
                filepath = self.raw_data_dir / f"{filename}.parquet"
            elif directory == "processed":
                filepath = self.processed_data_dir / f"{filename}.parquet"
            else:
                filepath = self.cache_dir / f"{filename}.parquet"
            
            if not self._is_cache_valid(filepath):
                logger.info(f"Cache expired for {filename}")
                return None
            
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded DataFrame from {filepath}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from {filename}: {e}")
            return None
    
    def save_json(self, data: Dict, filename: str, 
                 directory: str = "cache") -> bool:
        """Save dictionary data as JSON."""
        try:
            if directory == "cache":
                filepath = self.cache_dir / f"{filename}.json"
            elif directory == "processed":
                filepath = self.processed_data_dir / f"{filename}.json"
            else:
                filepath = self.raw_data_dir / f"{filename}.json"
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved JSON data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON to {filename}: {e}")
            return False
    
    def load_json(self, filename: str, 
                 directory: str = "cache") -> Optional[Dict]:
        """Load dictionary data from JSON."""
        try:
            if directory == "cache":
                filepath = self.cache_dir / f"{filename}.json"
            elif directory == "processed":
                filepath = self.processed_data_dir / f"{filename}.json"
            else:
                filepath = self.raw_data_dir / f"{filename}.json"
            
            if not self._is_cache_valid(filepath):
                logger.info(f"Cache expired for {filename}")
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded JSON data from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON from {filename}: {e}")
            return None
    
    def save_model(self, model: Any, filename: str, 
                  metadata: Optional[Dict] = None) -> bool:
        """Save machine learning model with metadata."""
        try:
            model_path = self.models_dir / f"{filename}.pkl"
            metadata_path = self.models_dir / f"{filename}_metadata.json"
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'saved_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'file_size': os.path.getsize(model_path)
            })
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved model to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {filename}: {e}")
            return False
    
    def load_model(self, filename: str) -> Optional[tuple]:
        """Load machine learning model with metadata."""
        try:
            model_path = self.models_dir / f"{filename}.pkl"
            metadata_path = self.models_dir / f"{filename}_metadata.json"
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return None
            
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"Loaded model from {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {filename}: {e}")
            return None
    
    def cache_spotify_data(self, user_id: str, data_type: str, 
                          data: Union[List, Dict]) -> bool:
        """Cache Spotify API data with user-specific keys."""
        cache_key = f"{user_id}_{data_type}_{self._generate_cache_key(str(data))}"
        return self.save_json(data, cache_key, directory="cache")
    
    def get_cached_spotify_data(self, user_id: str, 
                               data_type: str) -> Optional[Union[List, Dict]]:
        """Retrieve cached Spotify data for a user."""
        # Look for any cached files matching the pattern
        pattern = f"{user_id}_{data_type}_*"
        cache_files = list(self.cache_dir.glob(f"{pattern}.json"))
        
        if not cache_files:
            return None
        
        # Get the most recent cache file
        latest_file = max(cache_files, key=lambda x: x.stat().st_mtime)
        filename = latest_file.stem
        
        return self.load_json(filename, directory="cache")
    
    def cache_audio_features(self, track_ids: List[str], 
                           features: List[Dict]) -> bool:
        """Cache audio features for specific tracks."""
        cache_data = {
            'track_ids': track_ids,
            'features': features,
            'cached_at': datetime.now().isoformat()
        }
        
        cache_key = f"audio_features_{self._generate_cache_key(str(track_ids))}"
        return self.save_json(cache_data, cache_key, directory="cache")
    
    def get_cached_audio_features(self, track_ids: List[str]) -> Optional[List[Dict]]:
        """Retrieve cached audio features for tracks."""
        cache_key = f"audio_features_{self._generate_cache_key(str(track_ids))}"
        cached_data = self.load_json(cache_key, directory="cache")
        
        if cached_data and cached_data.get('track_ids') == track_ids:
            return cached_data.get('features')
        
        return None
    
    def save_processed_dataset(self, df: pd.DataFrame, feature_columns: List[str], 
                              stats: Dict, version: str = "latest") -> bool:
        """Save complete processed dataset with metadata."""
        try:
            # Save main dataset
            dataset_saved = self.save_dataframe(df, f"processed_dataset_{version}")
            
            # Save feature columns
            features_saved = self.save_json(
                {'feature_columns': feature_columns}, 
                f"feature_columns_{version}",
                directory="processed"
            )
            
            # Save statistics
            stats_saved = self.save_json(
                stats, 
                f"dataset_stats_{version}",
                directory="processed"
            )
            
            return dataset_saved and features_saved and stats_saved
            
        except Exception as e:
            logger.error(f"Error saving processed dataset: {e}")
            return False
    
    def load_processed_dataset(self, version: str = "latest") -> Optional[tuple]:
        """Load complete processed dataset with metadata."""
        try:
            # Load main dataset
            df = self.load_dataframe(f"processed_dataset_{version}")
            if df is None:
                return None
            
            # Load feature columns
            features_data = self.load_json(f"feature_columns_{version}", directory="processed")
            feature_columns = features_data.get('feature_columns', []) if features_data else []
            
            # Load statistics
            stats = self.load_json(f"dataset_stats_{version}", directory="processed")
            if stats is None:
                stats = {}
            
            logger.info(f"Loaded processed dataset version {version}")
            return df, feature_columns, stats
            
        except Exception as e:
            logger.error(f"Error loading processed dataset: {e}")
            return None
    
    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """Clear cache files, optionally only those older than specified hours."""
        cleared_count = 0
        
        if older_than_hours is None:
            older_than_hours = self.cache_expiry_hours
        
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        cache_file.unlink()
                        cleared_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {cleared_count} cache files")
        return cleared_count
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        info = {
            'cache_dir': str(self.cache_dir),
            'raw_data_dir': str(self.raw_data_dir),
            'processed_data_dir': str(self.processed_data_dir),
            'models_dir': str(self.models_dir),
            'cache_expiry_hours': self.cache_expiry_hours
        }
        
        # Count files in each directory
        for dir_name, dir_path in [
            ('cache_files', self.cache_dir),
            ('raw_files', self.raw_data_dir),
            ('processed_files', self.processed_data_dir),
            ('model_files', self.models_dir)
        ]:
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                info[f'{dir_name}_count'] = len([f for f in files if f.is_file()])
                info[f'{dir_name}_size_mb'] = sum(
                    f.stat().st_size for f in files if f.is_file()
                ) / (1024 * 1024)
            else:
                info[f'{dir_name}_count'] = 0
                info[f'{dir_name}_size_mb'] = 0
        
        return info
    
    def backup_data(self, backup_dir: str) -> bool:
        """Create backup of all cached data."""
        try:
            import shutil
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup each directory
            for source_dir, dir_name in [
                (self.cache_dir, 'cache'),
                (self.raw_data_dir, 'raw'),
                (self.processed_data_dir, 'processed'),
                (self.models_dir, 'models')
            ]:
                if source_dir.exists():
                    backup_subdir = backup_path / f"{dir_name}_{timestamp}"
                    shutil.copytree(source_dir, backup_subdir)
            
            logger.info(f"Backup created at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
