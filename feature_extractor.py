"""
Feature extraction and engineering for music recommendation system.
Creates advanced features from Spotify audio features and metadata.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import yaml

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Advanced feature extraction and engineering for music tracks."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature extractor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.audio_features = self.config['features']['audio_features']
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        
        # Feature engineering parameters
        self.tempo_bands = self.config['features']['engineered_features']['tempo_bands']
        self.energy_bands = self.config['features']['engineered_features']['energy_bands']
    
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and normalize basic audio features."""
        logger.info("Extracting basic audio features")
        
        df_features = df.copy()
        
        # Normalize tempo to 0-1 scale (typical range: 60-200 BPM)
        df_features['tempo_normalized'] = np.clip((df['tempo'] - 60) / 140, 0, 1)
        
        # Normalize loudness to 0-1 scale (typical range: -60 to 0 dB)
        df_features['loudness_normalized'] = np.clip((df['loudness'] + 60) / 60, 0, 1)
        
        # Duration in minutes
        df_features['duration_minutes'] = df['duration_ms'] / 60000
        
        # Popularity score (already 0-100, normalize to 0-1)
        df_features['popularity_normalized'] = df['popularity'] / 100
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between audio characteristics."""
        logger.info("Creating interaction features")
        
        df_interactions = df.copy()
        
        # Valence-Energy interaction (mood quadrants)
        df_interactions['valence_energy'] = df['valence'] * df['energy']
        df_interactions['valence_energy_diff'] = df['valence'] - df['energy']
        
        # Danceability interactions
        df_interactions['dance_energy'] = df['danceability'] * df['energy']
        df_interactions['dance_tempo'] = df['danceability'] * df['tempo_normalized']
        df_interactions['dance_valence'] = df['danceability'] * df['valence']
        
        # Acoustic interactions
        df_interactions['acoustic_valence'] = df['acousticness'] * df['valence']
        df_interactions['acoustic_energy'] = df['acousticness'] * df['energy']
        df_interactions['acoustic_instrumental'] = df['acousticness'] * df['instrumentalness']
        
        # Loudness interactions
        df_interactions['loudness_energy'] = df['loudness_normalized'] * df['energy']
        df_interactions['loudness_valence'] = df['loudness_normalized'] * df['valence']
        
        # Speech interactions
        df_interactions['speech_energy'] = df['speechiness'] * df['energy']
        df_interactions['speech_valence'] = df['speechiness'] * df['valence']
        
        # Liveness interactions
        df_interactions['live_energy'] = df['liveness'] * df['energy']
        df_interactions['live_dance'] = df['liveness'] * df['danceability']
        
        return df_interactions
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features."""
        logger.info("Creating ratio features")
        
        df_ratios = df.copy()
        
        # Avoid division by zero
        epsilon = 1e-8
        
        # Energy to acousticness ratio
        df_ratios['energy_acoustic_ratio'] = df['energy'] / (df['acousticness'] + epsilon)
        
        # Valence to energy ratio
        df_ratios['valence_energy_ratio'] = df['valence'] / (df['energy'] + epsilon)
        
        # Danceability to tempo ratio
        df_ratios['dance_tempo_ratio'] = df['danceability'] / (df['tempo_normalized'] + epsilon)
        
        # Instrumentalness to speechiness ratio
        df_ratios['instrumental_speech_ratio'] = df['instrumentalness'] / (df['speechiness'] + epsilon)
        
        # Loudness to energy ratio
        df_ratios['loudness_energy_ratio'] = df['loudness_normalized'] / (df['energy'] + epsilon)
        
        return df_ratios
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical features from continuous variables."""
        logger.info("Creating categorical features")
        
        df_categorical = df.copy()
        
        # Tempo categories
        df_categorical['tempo_category'] = pd.cut(
            df['tempo'],
            bins=[0, self.tempo_bands['slow'][1], self.tempo_bands['medium'][1], 300],
            labels=['slow', 'medium', 'fast']
        )
        
        # Energy categories
        df_categorical['energy_category'] = pd.cut(
            df['energy'],
            bins=[0, self.energy_bands['low'][1], self.energy_bands['medium'][1], 1],
            labels=['low', 'medium', 'high']
        )
        
        # Valence categories (emotional tone)
        df_categorical['valence_category'] = pd.cut(
            df['valence'],
            bins=[0, 0.33, 0.66, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        # Danceability categories
        df_categorical['danceability_category'] = pd.cut(
            df['danceability'],
            bins=[0, 0.33, 0.66, 1],
            labels=['not_danceable', 'somewhat_danceable', 'very_danceable']
        )
        
        # Popularity categories
        df_categorical['popularity_category'] = pd.cut(
            df['popularity'],
            bins=[0, 25, 50, 75, 100],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Duration categories
        df_categorical['duration_category'] = pd.cut(
            df['duration_minutes'],
            bins=[0, 2, 4, 6, float('inf')],
            labels=['very_short', 'short', 'medium', 'long']
        )
        
        # Acousticness categories
        df_categorical['acousticness_category'] = pd.cut(
            df['acousticness'],
            bins=[0, 0.2, 0.8, 1],
            labels=['electric', 'mixed', 'acoustic']
        )
        
        # Instrumentalness categories
        df_categorical['instrumentalness_category'] = pd.cut(
            df['instrumentalness'],
            bins=[0, 0.1, 0.5, 1],
            labels=['vocal', 'mixed', 'instrumental']
        )
        
        return df_categorical
    
    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features representing complex musical concepts."""
        logger.info("Creating composite features")
        
        df_composite = df.copy()
        
        # Mood indicators
        df_composite['happiness_index'] = (
            df['valence'] * 0.5 + 
            df['energy'] * 0.3 + 
            df['danceability'] * 0.2
        )
        
        df_composite['melancholy_index'] = (
            (1 - df['valence']) * 0.4 + 
            (1 - df['energy']) * 0.3 + 
            df['acousticness'] * 0.3
        )
        
        df_composite['intensity_index'] = (
            df['energy'] * 0.4 + 
            df['loudness_normalized'] * 0.3 + 
            df['tempo_normalized'] * 0.3
        )
        
        df_composite['calmness_index'] = (
            (1 - df['energy']) * 0.4 + 
            df['acousticness'] * 0.3 + 
            (1 - df['loudness_normalized']) * 0.3
        )
        
        # Danceability composite
        df_composite['party_index'] = (
            df['danceability'] * 0.4 + 
            df['energy'] * 0.3 + 
            df['valence'] * 0.3
        )
        
        # Focus/concentration index
        df_composite['focus_index'] = (
            df['instrumentalness'] * 0.4 + 
            (1 - df['speechiness']) * 0.3 + 
            df['acousticness'] * 0.3
        )
        
        # Workout index
        df_composite['workout_index'] = (
            df['energy'] * 0.4 + 
            df['tempo_normalized'] * 0.3 + 
            df['danceability'] * 0.3
        )
        
        # Relaxation index
        df_composite['relaxation_index'] = (
            (1 - df['energy']) * 0.3 + 
            df['acousticness'] * 0.3 + 
            (1 - df['tempo_normalized']) * 0.2 + 
            (1 - df['loudness_normalized']) * 0.2
        )
        
        return df_composite
    
    def create_statistical_features(self, df: pd.DataFrame, 
                                   groupby_column: str = 'artists') -> pd.DataFrame:
        """Create statistical features based on artist/album patterns."""
        logger.info(f"Creating statistical features grouped by {groupby_column}")
        
        df_stats = df.copy()
        
        if groupby_column not in df.columns:
            logger.warning(f"Column {groupby_column} not found. Skipping statistical features.")
            return df_stats
        
        # Artist-level statistics
        if groupby_column == 'artists':
            # Convert list of artists to string for grouping
            df_stats['artists_str'] = df_stats['artists'].astype(str)
            group_col = 'artists_str'
        else:
            group_col = groupby_column
        
        # Calculate group statistics for key features
        key_features = ['valence', 'energy', 'danceability', 'acousticness', 'popularity']
        
        for feature in key_features:
            if feature in df_stats.columns:
                group_stats = df_stats.groupby(group_col)[feature].agg(['mean', 'std', 'count'])
                
                # Merge back to main dataframe
                df_stats = df_stats.merge(
                    group_stats.rename(columns={
                        'mean': f'{group_col}_{feature}_mean',
                        'std': f'{group_col}_{feature}_std',
                        'count': f'{group_col}_{feature}_count'
                    }),
                    left_on=group_col,
                    right_index=True,
                    how='left'
                )
                
                # Feature deviation from group mean
                df_stats[f'{feature}_deviation_from_{group_col}'] = (
                    df_stats[feature] - df_stats[f'{group_col}_{feature}_mean']
                )
        
        return df_stats
    
    def create_clustering_features(self, df: pd.DataFrame, 
                                  n_clusters: int = 8) -> pd.DataFrame:
        """Create clustering-based features for music similarity."""
        logger.info(f"Creating clustering features with {n_clusters} clusters")
        
        df_clustering = df.copy()
        
        # Select features for clustering
        clustering_features = [
            'valence', 'energy', 'danceability', 'acousticness',
            'instrumentalness', 'tempo_normalized', 'loudness_normalized'
        ]
        
        available_features = [f for f in clustering_features if f in df.columns]
        
        if len(available_features) < 3:
            logger.warning("Not enough features for clustering. Skipping clustering features.")
            return df_clustering
        
        # Prepare data for clustering
        X_cluster = df[available_features].fillna(0)
        
        # Standardize features
        X_scaled = StandardScaler().fit_transform(X_cluster)
        
        # Perform K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        df_clustering['music_cluster'] = cluster_labels
        
        # Distance to cluster centers
        cluster_distances = self.kmeans.transform(X_scaled)
        df_clustering['distance_to_cluster_center'] = np.min(cluster_distances, axis=1)
        
        # Cluster-based features
        for i in range(n_clusters):
            df_clustering[f'cluster_{i}_distance'] = cluster_distances[:, i]
        
        # Cluster statistics
        cluster_stats = df_clustering.groupby('music_cluster').agg({
            'valence': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'popularity': 'mean'
        }).add_suffix('_cluster_mean')
        
        df_clustering = df_clustering.merge(
            cluster_stats, 
            left_on='music_cluster', 
            right_index=True, 
            how='left'
        )
        
        logger.info(f"Created clustering features. Cluster distribution: "
                   f"{df_clustering['music_cluster'].value_counts().to_dict()}")
        
        return df_clustering
    
    def create_pca_features(self, df: pd.DataFrame, 
                           n_components: int = 5) -> pd.DataFrame:
        """Create PCA features for dimensionality reduction."""
        logger.info(f"Creating PCA features with {n_components} components")
        
        df_pca = df.copy()
        
        # Select numerical features for PCA
        pca_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness_normalized', 'speechiness', 
            'tempo_normalized', 'valence', 'popularity_normalized'
        ]
        
        available_features = [f for f in pca_features if f in df.columns]
        
        if len(available_features) < n_components:
            logger.warning("Not enough features for PCA. Skipping PCA features.")
            return df_pca
        
        # Prepare data for PCA
        X_pca = df[available_features].fillna(0)
        
        # Standardize features
        X_scaled = StandardScaler().fit_transform(X_pca)
        
        # Apply PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        pca_components = self.pca.fit_transform(X_scaled)
        
        # Add PCA components as features
        for i in range(n_components):
            df_pca[f'pca_component_{i}'] = pca_components[:, i]
        
        # Explained variance ratio
        explained_variance = self.pca.explained_variance_ratio_
        logger.info(f"PCA explained variance ratio: {explained_variance}")
        
        return df_pca
    
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features if timestamp data is available."""
        logger.info("Creating time-based features")
        
        df_time = df.copy()
        
        # Check for timestamp columns
        timestamp_columns = ['added_at', 'release_date', 'created_at']
        available_timestamps = [col for col in timestamp_columns if col in df.columns]
        
        if not available_timestamps:
            logger.info("No timestamp columns found. Skipping time-based features.")
            return df_time
        
        for timestamp_col in available_timestamps:
            # Convert to datetime
            df_time[timestamp_col] = pd.to_datetime(df_time[timestamp_col], errors='coerce')
            
            # Extract time components
            df_time[f'{timestamp_col}_year'] = df_time[timestamp_col].dt.year
            df_time[f'{timestamp_col}_month'] = df_time[timestamp_col].dt.month
            df_time[f'{timestamp_col}_day_of_week'] = df_time[timestamp_col].dt.dayofweek
            df_time[f'{timestamp_col}_hour'] = df_time[timestamp_col].dt.hour
            
            # Days since timestamp
            df_time[f'days_since_{timestamp_col}'] = (
                pd.Timestamp.now() - df_time[timestamp_col]
            ).dt.days
            
            # Recency features
            df_time[f'is_recent_{timestamp_col}'] = (
                df_time[f'days_since_{timestamp_col}'] <= 30
            ).astype(int)
            
            # Season features
            df_time[f'{timestamp_col}_season'] = df_time[f'{timestamp_col}_month'].map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            })
        
        return df_time
    
    def extract_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Extract all engineered features."""
        logger.info("Extracting all engineered features")
        
        # Start with basic features
        df_features = self.extract_basic_features(df)
        
        # Add interaction features
        df_features = self.create_interaction_features(df_features)
        
        # Add ratio features
        df_features = self.create_ratio_features(df_features)
        
        # Add categorical features
        df_features = self.create_categorical_features(df_features)
        
        # Add composite features
        df_features = self.create_composite_features(df_features)
        
        # Add clustering features
        df_features = self.create_clustering_features(df_features)
        
        # Add PCA features
        df_features = self.create_pca_features(df_features)
        
        # Add time-based features
        df_features = self.create_time_based_features(df_features)
        
        # Add statistical features (if possible)
        if 'artists' in df_features.columns:
            df_features = self.create_statistical_features(df_features, 'artists')
        
        # Get list of all engineered feature columns
        original_columns = set(df.columns)
        engineered_columns = [col for col in df_features.columns 
                            if col not in original_columns]
        
        logger.info(f"Created {len(engineered_columns)} engineered features")
        return df_features, engineered_columns
    
    def get_feature_importance_analysis(self, df: pd.DataFrame, 
                                      target_column: str) -> Dict:
        """Analyze feature importance for a target variable."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        
        logger.info(f"Analyzing feature importance for {target_column}")
        
        # Select numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features 
                            if col != target_column and col in df.columns]
        
        if len(numerical_features) < 2:
            logger.warning("Not enough numerical features for importance analysis")
            return {}
        
        X = df[numerical_features].fillna(0)
        y = df[target_column].fillna(0)
        
        # Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = dict(zip(numerical_features, rf.feature_importances_))
        
        # Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = dict(zip(numerical_features, mi_scores))
        
        # Correlation with target
        correlations = {}
        for feature in numerical_features:
            correlations[feature] = df[feature].corr(df[target_column])
        
        return {
            'random_forest_importance': rf_importance,
            'mutual_information': mi_importance,
            'correlations': correlations,
            'top_rf_features': sorted(rf_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:10],
            'top_mi_features': sorted(mi_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
        }
