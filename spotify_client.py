"""
Spotify API client for fetching user data, playlists, and track features.
Handles authentication, rate limiting, and data extraction.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Tuple
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from tqdm import tqdm
import yaml




logger = logging.getLogger(__name__)


class SpotifyClient:
    """Enhanced Spotify client with caching and rate limiting."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Spotify client with configuration."""
        self.config = self._load_config(config_path)
        self.sp = self._authenticate()
        self.user_id = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Replace environment variables
        spotify_config = config['spotify']
        spotify_config['client_id'] = os.getenv('SPOTIFY_CLIENT_ID', spotify_config['client_id'])
        spotify_config['client_secret'] = os.getenv('SPOTIFY_CLIENT_SECRET', spotify_config['client_secret'])
        
        return config
    
    def _authenticate(self) -> spotipy.Spotify:
        """Authenticate with Spotify API."""
        try:
            auth_manager = SpotifyOAuth(
                client_id=self.config['spotify']['client_id'],
                client_secret=self.config['spotify']['client_secret'],
                redirect_uri=self.config['spotify']['redirect_uri'],
                scope=self.config['spotify']['scope'],
                cache_path=".spotify_cache"
            )
            
            sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Test authentication
            user_info = sp.current_user()
            self.user_id = user_info['id']
            logger.info(f"Successfully authenticated as user: {self.user_id}")
            
            return sp
            
        except Exception as e:
            logger.error(f"Spotify authentication failed: {e}")
            raise
    
    def get_user_playlists(self, limit: int = 50) -> List[Dict]:
        """Fetch user's playlists with metadata."""
        playlists = []
        offset = 0
        
        while True:
            try:
                results = self.sp.current_user_playlists(limit=limit, offset=offset)
                
                for playlist in results['items']:
                    if playlist['owner']['id'] == self.user_id:  # Only user's own playlists
                        playlists.append({
                            'id': playlist['id'],
                            'name': playlist['name'],
                            'description': playlist['description'],
                            'track_count': playlist['tracks']['total'],
                            'public': playlist['public'],
                            'collaborative': playlist['collaborative']
                        })
                
                if not results['next']:
                    break
                    
                offset += limit
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching playlists: {e}")
                break
        
        logger.info(f"Fetched {len(playlists)} user playlists")
        return playlists
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        """Fetch all tracks from a specific playlist."""
        tracks = []
        offset = 0
        limit = self.config['data']['max_tracks_per_request']
        
        while True:
            try:
                results = self.sp.playlist_tracks(
                    playlist_id, 
                    offset=offset, 
                    limit=limit,
                    fields="items(track(id,name,artists,album,popularity,duration_ms,explicit))"
                )
                
                for item in results['items']:
                    track = item['track']
                    if track and track['id']:  # Skip None tracks
                        tracks.append({
                            'track_id': track['id'],
                            'name': track['name'],
                            'artists': [artist['name'] for artist in track['artists']],
                            'album': track['album']['name'],
                            'popularity': track['popularity'],
                            'duration_ms': track['duration_ms'],
                            'explicit': track['explicit'],
                            'playlist_id': playlist_id
                        })
                
                if not results['next']:
                    break
                    
                offset += limit
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching tracks for playlist {playlist_id}: {e}")
                break
        
        return tracks
    
    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """Fetch audio features for multiple tracks."""
        features = []
        batch_size = 100  # Spotify API limit
        
        for i in tqdm(range(0, len(track_ids), batch_size), desc="Fetching audio features"):
            batch = track_ids[i:i + batch_size]
            
            try:
                batch_features = self.sp.audio_features(batch)
                
                for feature in batch_features:
                    if feature:  # Skip None results
                        features.append(feature)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching audio features for batch: {e}")
                continue
        
        logger.info(f"Fetched audio features for {len(features)} tracks")
        return features
    
    def get_user_top_tracks(self, time_range: str = 'medium_term', limit: int = 50) -> List[Dict]:
        """Fetch user's top tracks for taste modeling."""
        try:
            results = self.sp.current_user_top_tracks(
                time_range=time_range, 
                limit=limit
            )
            
            tracks = []
            for track in results['items']:
                tracks.append({
                    'track_id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms'],
                    'explicit': track['explicit'],
                    'user_preference': 1.0  # Implicit positive feedback
                })
            
            logger.info(f"Fetched {len(tracks)} top tracks for {time_range}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error fetching top tracks: {e}")
            return []
    
    def get_saved_tracks(self, limit: int = 50) -> List[Dict]:
        """Fetch user's saved/liked tracks."""
        tracks = []
        offset = 0
        
        while True:
            try:
                results = self.sp.current_user_saved_tracks(limit=limit, offset=offset)
                
                for item in results['items']:
                    track = item['track']
                    tracks.append({
                        'track_id': track['id'],
                        'name': track['name'],
                        'artists': [artist['name'] for artist in track['artists']],
                        'album': track['album']['name'],
                        'popularity': track['popularity'],
                        'duration_ms': track['duration_ms'],
                        'explicit': track['explicit'],
                        'added_at': item['added_at'],
                        'user_preference': 1.0  # Explicit positive feedback
                    })
                
                if not results['next']:
                    break
                    
                offset += limit
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching saved tracks: {e}")
                break
        
        logger.info(f"Fetched {len(tracks)} saved tracks")
        return tracks
    
    def search_tracks(self, query: str, limit: int = 50, market: str = 'US') -> List[Dict]:
        """Search for tracks to expand the dataset."""
        try:
            results = self.sp.search(q=query, type='track', limit=limit, market=market)
            
            tracks = []
            for track in results['tracks']['items']:
                tracks.append({
                    'track_id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms'],
                    'explicit': track['explicit'],
                    'user_preference': 0.0  # Unknown preference
                })
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return []
    
    def create_playlist(self, name: str, description: str, track_ids: List[str], 
                       public: bool = False) -> Optional[str]:
        """Create a new playlist with given tracks."""
        try:
            # Create playlist
            playlist = self.sp.user_playlist_create(
                user=self.user_id,
                name=name,
                public=public,
                description=description
            )
            
            playlist_id = playlist['id']
            
            # Add tracks in batches
            batch_size = 100
            for i in range(0, len(track_ids), batch_size):
                batch = track_ids[i:i + batch_size]
                track_uris = [f"spotify:track:{track_id}" for track_id in batch]
                self.sp.playlist_add_items(playlist_id, track_uris)
                time.sleep(0.1)
            
            logger.info(f"Created playlist '{name}' with {len(track_ids)} tracks")
            return playlist_id
            
        except Exception as e:
            logger.error(f"Error creating playlist: {e}")
            return None
    
    def get_comprehensive_dataset(self) -> pd.DataFrame:
        """Fetch comprehensive dataset for training."""
        all_tracks = []
        
        # Get user's positive examples
        logger.info("Fetching user's liked tracks...")
        saved_tracks = self.get_saved_tracks()
        all_tracks.extend(saved_tracks)
        
        # Get top tracks
        logger.info("Fetching user's top tracks...")
        for time_range in ['short_term', 'medium_term', 'long_term']:
            top_tracks = self.get_user_top_tracks(time_range=time_range)
            all_tracks.extend(top_tracks)
        
        # Get playlist tracks
        logger.info("Fetching playlist tracks...")
        playlists = self.get_user_playlists()
        for playlist in playlists[:10]:  # Limit to avoid rate limits
            playlist_tracks = self.get_playlist_tracks(playlist['id'])
            # Mark as positive examples with lower confidence
            for track in playlist_tracks:
                track['user_preference'] = 0.7
            all_tracks.extend(playlist_tracks)
        
        # Convert to DataFrame and remove duplicates
        df = pd.DataFrame(all_tracks)
        if not df.empty:
            df = df.drop_duplicates(subset=['track_id'])
            df = df.groupby('track_id').agg({
                'name': 'first',
                'artists': 'first',
                'album': 'first',
                'popularity': 'first',
                'duration_ms': 'first',
                'explicit': 'first',
                'user_preference': 'max'  # Take highest preference score
            }).reset_index()
        
        logger.info(f"Compiled dataset with {len(df)} unique tracks")
        return df
