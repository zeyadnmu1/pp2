"""
Evaluation metrics and feedback processing for the recommendation system.
Includes offline metrics (precision@K, recall@K, diversity) and online feedback handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import yaml

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Comprehensive evaluation system for recommendation quality."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.eval_config = self.config['evaluation']
        self.k_values = self.eval_config['k_values']
        self.feedback_history = []
        
    def precision_at_k(self, recommended_items: List[str], 
                      relevant_items: Set[str], k: int) -> float:
        """Calculate Precision@K metric."""
        if k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len([item for item in recommended_k if item in relevant_items])
        
        return relevant_recommended / min(k, len(recommended_k))
    
    def recall_at_k(self, recommended_items: List[str], 
                   relevant_items: Set[str], k: int) -> float:
        """Calculate Recall@K metric."""
        if not relevant_items or k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len([item for item in recommended_k if item in relevant_items])
        
        return relevant_recommended / len(relevant_items)
    
    def f1_at_k(self, recommended_items: List[str], 
               relevant_items: Set[str], k: int) -> float:
        """Calculate F1@K metric."""
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_average_precision(self, recommended_items: List[str], 
                              relevant_items: Set[str]) -> float:
        """Calculate Mean Average Precision (MAP)."""
        if not relevant_items:
            return 0.0
        
        average_precisions = []
        relevant_count = 0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                average_precisions.append(precision_at_i)
        
        if not average_precisions:
            return 0.0
        
        return np.mean(average_precisions)
    
    def normalized_discounted_cumulative_gain(self, recommended_items: List[str],
                                            relevance_scores: Dict[str, float],
                                            k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
        if k <= 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            relevance = relevance_scores.get(item, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        sorted_relevances = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevances[:k]):
            idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def diversity_score(self, recommended_tracks: pd.DataFrame,
                       feature_columns: List[str] = None) -> float:
        """Calculate diversity score based on feature differences."""
        if len(recommended_tracks) < 2:
            return 0.0
        
        if feature_columns is None:
            feature_columns = ['valence', 'energy', 'danceability', 'acousticness']
        
        available_features = [col for col in feature_columns 
                            if col in recommended_tracks.columns]
        
        if not available_features:
            return 0.0
        
        # Calculate pairwise distances
        feature_matrix = recommended_tracks[available_features].values
        distances = []
        
        for i in range(len(feature_matrix)):
            for j in range(i + 1, len(feature_matrix)):
                distance = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                distances.append(distance)
        
        # Average distance as diversity score
        return np.mean(distances) if distances else 0.0
    
    def novelty_score(self, recommended_tracks: pd.DataFrame,
                     popularity_column: str = 'popularity') -> float:
        """Calculate novelty score based on track popularity."""
        if popularity_column not in recommended_tracks.columns:
            return 0.0
        
        # Novelty is inverse of popularity (normalized)
        popularities = recommended_tracks[popularity_column].values
        max_popularity = 100.0  # Spotify popularity scale
        
        # Convert to novelty scores (higher for less popular tracks)
        novelty_scores = (max_popularity - popularities) / max_popularity
        
        return np.mean(novelty_scores)
    
    def coverage_score(self, recommended_items: List[str], 
                      catalog_items: Set[str]) -> float:
        """Calculate catalog coverage score."""
        if not catalog_items:
            return 0.0
        
        unique_recommended = set(recommended_items)
        covered_items = unique_recommended.intersection(catalog_items)
        
        return len(covered_items) / len(catalog_items)
    
    def intra_list_diversity(self, recommended_tracks: pd.DataFrame,
                           similarity_matrix: Optional[np.ndarray] = None) -> float:
        """Calculate intra-list diversity using similarity matrix."""
        if similarity_matrix is None or len(recommended_tracks) < 2:
            return self.diversity_score(recommended_tracks)
        
        n_tracks = len(recommended_tracks)
        similarities = []
        
        for i in range(n_tracks):
            for j in range(i + 1, n_tracks):
                if i < len(similarity_matrix) and j < len(similarity_matrix[0]):
                    similarities.append(similarity_matrix[i][j])
        
        if not similarities:
            return 0.0
        
        # Diversity is inverse of similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def evaluate_recommendations(self, 
                               recommended_tracks: pd.DataFrame,
                               user_preferences: Dict[str, float],
                               catalog_tracks: Optional[Set[str]] = None) -> Dict:
        """Comprehensive evaluation of recommendations."""
        logger.info("Evaluating recommendations")
        
        if recommended_tracks.empty:
            return {'error': 'No recommendations to evaluate'}
        
        recommended_ids = recommended_tracks['track_id'].tolist()
        relevant_items = set([track_id for track_id, score in user_preferences.items() 
                            if score >= 0.5])
        
        results = {}
        
        # Precision, Recall, F1 at different K values
        for k in self.k_values:
            results[f'precision_at_{k}'] = self.precision_at_k(recommended_ids, relevant_items, k)
            results[f'recall_at_{k}'] = self.recall_at_k(recommended_ids, relevant_items, k)
            results[f'f1_at_{k}'] = self.f1_at_k(recommended_ids, relevant_items, k)
        
        # Mean Average Precision
        results['mean_average_precision'] = self.mean_average_precision(
            recommended_ids, relevant_items
        )
        
        # NDCG
        for k in self.k_values:
            results[f'ndcg_at_{k}'] = self.normalized_discounted_cumulative_gain(
                recommended_ids, user_preferences, k
            )
        
        # Diversity metrics
        results['diversity_score'] = self.diversity_score(recommended_tracks)
        results['novelty_score'] = self.novelty_score(recommended_tracks)
        
        # Coverage (if catalog provided)
        if catalog_tracks:
            results['coverage_score'] = self.coverage_score(recommended_ids, catalog_tracks)
        
        # Additional metrics
        results['recommendation_count'] = len(recommended_tracks)
        
        if 'popularity' in recommended_tracks.columns:
            results['avg_popularity'] = recommended_tracks['popularity'].mean()
            results['popularity_std'] = recommended_tracks['popularity'].std()
        
        if 'taste_score' in recommended_tracks.columns:
            results['avg_taste_score'] = recommended_tracks['taste_score'].mean()
            results['taste_score_std'] = recommended_tracks['taste_score'].std()
        
        logger.info(f"Evaluation completed. Precision@10: {results.get('precision_at_10', 0):.3f}")
        return results
    
    def process_user_feedback(self, track_id: str, feedback: str, 
                            user_id: str, timestamp: Optional[str] = None) -> bool:
        """Process user feedback (thumbs up/down, skip, etc.)."""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
        
        feedback_entry = {
            'user_id': user_id,
            'track_id': track_id,
            'feedback': feedback,
            'timestamp': timestamp
        }
        
        self.feedback_history.append(feedback_entry)
        logger.info(f"Processed feedback: {feedback} for track {track_id}")
        
        return True
    
    def get_feedback_statistics(self, user_id: Optional[str] = None) -> Dict:
        """Get statistics from user feedback."""
        feedback_data = self.feedback_history
        
        if user_id:
            feedback_data = [f for f in feedback_data if f['user_id'] == user_id]
        
        if not feedback_data:
            return {'total_feedback': 0}
        
        feedback_df = pd.DataFrame(feedback_data)
        
        stats = {
            'total_feedback': len(feedback_df),
            'feedback_distribution': feedback_df['feedback'].value_counts().to_dict(),
            'unique_users': feedback_df['user_id'].nunique(),
            'unique_tracks': feedback_df['track_id'].nunique()
        }
        
        # Calculate feedback rates
        if 'like' in stats['feedback_distribution'] and 'dislike' in stats['feedback_distribution']:
            likes = stats['feedback_distribution']['like']
            dislikes = stats['feedback_distribution']['dislike']
            total_explicit = likes + dislikes
            
            if total_explicit > 0:
                stats['like_rate'] = likes / total_explicit
                stats['dislike_rate'] = dislikes / total_explicit
        
        return stats
    
    def create_feedback_based_preferences(self, user_id: str) -> Dict[str, float]:
        """Create preference scores based on user feedback."""
        user_feedback = [f for f in self.feedback_history if f['user_id'] == user_id]
        
        preferences = {}
        
        for feedback in user_feedback:
            track_id = feedback['track_id']
            feedback_type = feedback['feedback']
            
            # Convert feedback to preference score
            if feedback_type == 'like':
                preferences[track_id] = 1.0
            elif feedback_type == 'dislike':
                preferences[track_id] = 0.0
            elif feedback_type == 'skip':
                preferences[track_id] = 0.2
            elif feedback_type == 'play_complete':
                preferences[track_id] = 0.8
            elif feedback_type == 'replay':
                preferences[track_id] = 1.0
        
        return preferences
    
    def evaluate_playlist_quality(self, playlist: pd.DataFrame) -> Dict:
        """Evaluate overall playlist quality."""
        if playlist.empty:
            return {'error': 'Empty playlist'}
        
        quality_metrics = {}
        
        # Basic statistics
        quality_metrics['track_count'] = len(playlist)
        
        # Feature statistics
        audio_features = ['valence', 'energy', 'danceability', 'acousticness', 
                         'instrumentalness', 'speechiness', 'tempo']
        
        for feature in audio_features:
            if feature in playlist.columns:
                values = playlist[feature]
                quality_metrics[f'{feature}_mean'] = values.mean()
                quality_metrics[f'{feature}_std'] = values.std()
                quality_metrics[f'{feature}_range'] = values.max() - values.min()
        
        # Diversity metrics
        quality_metrics['diversity_score'] = self.diversity_score(playlist)
        quality_metrics['novelty_score'] = self.novelty_score(playlist)
        
        # Duration analysis
        if 'duration_minutes' in playlist.columns:
            quality_metrics['total_duration'] = playlist['duration_minutes'].sum()
            quality_metrics['avg_track_duration'] = playlist['duration_minutes'].mean()
        
        # Artist diversity
        if 'artists' in playlist.columns:
            all_artists = []
            for artists_list in playlist['artists']:
                if isinstance(artists_list, list):
                    all_artists.extend(artists_list)
            
            unique_artists = len(set(all_artists))
            quality_metrics['unique_artists'] = unique_artists
            quality_metrics['artist_diversity_ratio'] = unique_artists / len(playlist)
        
        # Mood consistency
        if 'primary_mood' in playlist.columns:
            mood_distribution = playlist['primary_mood'].value_counts()
            dominant_mood_ratio = mood_distribution.iloc[0] / len(playlist)
            quality_metrics['mood_consistency'] = dominant_mood_ratio
            quality_metrics['mood_diversity'] = len(mood_distribution)
        
        # Popularity distribution
        if 'popularity' in playlist.columns:
            quality_metrics['avg_popularity'] = playlist['popularity'].mean()
            quality_metrics['popularity_std'] = playlist['popularity'].std()
            
            # Categorize by popularity
            high_pop = (playlist['popularity'] >= 70).sum()
            med_pop = ((playlist['popularity'] >= 30) & (playlist['popularity'] < 70)).sum()
            low_pop = (playlist['popularity'] < 30).sum()
            
            quality_metrics['high_popularity_ratio'] = high_pop / len(playlist)
            quality_metrics['medium_popularity_ratio'] = med_pop / len(playlist)
            quality_metrics['low_popularity_ratio'] = low_pop / len(playlist)
        
        return quality_metrics
    
    def compare_recommendation_strategies(self, 
                                        strategies_results: Dict[str, pd.DataFrame],
                                        user_preferences: Dict[str, float]) -> Dict:
        """Compare different recommendation strategies."""
        logger.info("Comparing recommendation strategies")
        
        comparison_results = {}
        
        for strategy_name, recommendations in strategies_results.items():
            strategy_metrics = self.evaluate_recommendations(
                recommendations, user_preferences
            )
            comparison_results[strategy_name] = strategy_metrics
        
        # Find best strategy for each metric
        best_strategies = {}
        
        if comparison_results:
            # Get all metric names
            all_metrics = set()
            for metrics in comparison_results.values():
                all_metrics.update(metrics.keys())
            
            for metric in all_metrics:
                if metric == 'error':
                    continue
                
                metric_values = {}
                for strategy, metrics in comparison_results.items():
                    if metric in metrics:
                        metric_values[strategy] = metrics[metric]
                
                if metric_values:
                    best_strategy = max(metric_values.items(), key=lambda x: x[1])
                    best_strategies[metric] = {
                        'strategy': best_strategy[0],
                        'value': best_strategy[1]
                    }
        
        return {
            'individual_results': comparison_results,
            'best_strategies': best_strategies,
            'strategy_count': len(strategies_results)
        }
    
    def generate_evaluation_report(self, evaluation_results: Dict) -> str:
        """Generate a human-readable evaluation report."""
        report = ["=== Recommendation Evaluation Report ===\n"]
        
        if 'error' in evaluation_results:
            report.append(f"Error: {evaluation_results['error']}")
            return "\n".join(report)
        
        # Basic metrics
        report.append("## Accuracy Metrics")
        for k in self.k_values:
            precision = evaluation_results.get(f'precision_at_{k}', 0)
            recall = evaluation_results.get(f'recall_at_{k}', 0)
            f1 = evaluation_results.get(f'f1_at_{k}', 0)
            
            report.append(f"K={k}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # MAP and NDCG
        map_score = evaluation_results.get('mean_average_precision', 0)
        report.append(f"\nMean Average Precision: {map_score:.3f}")
        
        report.append("\n## NDCG Scores")
        for k in self.k_values:
            ndcg = evaluation_results.get(f'ndcg_at_{k}', 0)
            report.append(f"NDCG@{k}: {ndcg:.3f}")
        
        # Diversity metrics
        diversity = evaluation_results.get('diversity_score', 0)
        novelty = evaluation_results.get('novelty_score', 0)
        
        report.append(f"\n## Diversity & Novelty")
        report.append(f"Diversity Score: {diversity:.3f}")
        report.append(f"Novelty Score: {novelty:.3f}")
        
        # Additional metrics
        if 'avg_popularity' in evaluation_results:
            avg_pop = evaluation_results['avg_popularity']
            report.append(f"\nAverage Popularity: {avg_pop:.1f}")
        
        if 'avg_taste_score' in evaluation_results:
            avg_taste = evaluation_results['avg_taste_score']
            report.append(f"Average Taste Score: {avg_taste:.3f}")
        
        return "\n".join(report)
    
    def save_evaluation_results(self, results: Dict, filepath: str) -> bool:
        """Save evaluation results to file."""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            return False
