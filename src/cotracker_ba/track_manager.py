"""Track Manager Module

This module handles merging tracks from overlapping windows and preparing them for bundle adjustment.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist

from .feature_extractor import Track

logger = logging.getLogger(__name__)


@dataclass
class MergedTrack:
    """Represents a merged track across the entire video"""
    track_id: int
    observations: Dict[int, np.ndarray]  # frame_id -> (x, y) coordinates
    visibility: Dict[int, float]  # frame_id -> visibility score
    confidence: Dict[int, float]  # frame_id -> confidence score


class TrackManager:
    """Manages and merges tracks from sliding windows"""
    
    def __init__(self, 
                 overlap_threshold: float = 10.0,
                 min_track_length: int = 10,
                 visibility_threshold: float = 0.5):
        """
        Initialize Track Manager
        
        Args:
            overlap_threshold: Maximum pixel distance to consider tracks as overlapping
            min_track_length: Minimum number of frames for a valid track
            visibility_threshold: Minimum visibility score to consider a point visible
        """
        self.overlap_threshold = overlap_threshold
        self.min_track_length = min_track_length
        self.visibility_threshold = visibility_threshold
        
        logger.info(f"Initialized TrackManager with overlap_threshold={overlap_threshold}, "
                   f"min_track_length={min_track_length}")
    
    def find_overlapping_tracks(self, 
                               track1: Track, 
                               track2: Track) -> Optional[Tuple[int, int, float]]:
        """Find overlapping frames between two tracks and compute similarity
        
        Args:
            track1: First track
            track2: Second track
            
        Returns:
            Tuple of (overlap_start, overlap_end, avg_distance) if tracks overlap, None otherwise
        """
        # Find frame overlap
        overlap_start = max(track1.start_frame, track2.start_frame)
        overlap_end = min(track1.end_frame, track2.end_frame)
        
        if overlap_start > overlap_end:
            return None
        
        # Extract overlapping points
        idx1_start = overlap_start - track1.start_frame
        idx1_end = idx1_start + (overlap_end - overlap_start + 1)
        
        idx2_start = overlap_start - track2.start_frame
        idx2_end = idx2_start + (overlap_end - overlap_start + 1)
        
        points1 = track1.points[idx1_start:idx1_end]
        points2 = track2.points[idx2_start:idx2_end]
        
        vis1 = track1.visibility[idx1_start:idx1_end]
        vis2 = track2.visibility[idx2_start:idx2_end]
        
        # Only compare visible points
        visible_mask = (vis1 > self.visibility_threshold) & (vis2 > self.visibility_threshold)
        
        if np.sum(visible_mask) < 3:  # Need at least 3 visible points
            return None
        
        # Compute average distance
        distances = np.linalg.norm(points1[visible_mask] - points2[visible_mask], axis=1)
        avg_distance = np.mean(distances)
        
        return overlap_start, overlap_end, avg_distance
    
    def merge_two_tracks(self, track1: Track, track2: Track) -> MergedTrack:
        """Merge two overlapping tracks into one
        
        Args:
            track1: First track
            track2: Second track
            
        Returns:
            Merged track combining both inputs
        """
        merged = MergedTrack(
            track_id=-1,  # Will be assigned later
            observations={},
            visibility={},
            confidence={}
        )
        
        # Add all observations from track1
        for i in range(len(track1.points)):
            frame_id = track1.start_frame + i
            merged.observations[frame_id] = track1.points[i]
            merged.visibility[frame_id] = track1.visibility[i]
            merged.confidence[frame_id] = track1.confidence[i]
        
        # Add non-overlapping observations from track2
        overlap_info = self.find_overlapping_tracks(track1, track2)
        if overlap_info:
            overlap_start, overlap_end, _ = overlap_info
            
            # Add observations before overlap
            for i in range(track2.start_frame - track2.start_frame, 
                          overlap_start - track2.start_frame):
                frame_id = track2.start_frame + i
                merged.observations[frame_id] = track2.points[i]
                merged.visibility[frame_id] = track2.visibility[i]
                merged.confidence[frame_id] = track2.confidence[i]
            
            # Average overlapping observations
            for frame_id in range(overlap_start, overlap_end + 1):
                idx1 = frame_id - track1.start_frame
                idx2 = frame_id - track2.start_frame
                
                # Average positions if both visible
                if (track1.visibility[idx1] > self.visibility_threshold and 
                    track2.visibility[idx2] > self.visibility_threshold):
                    merged.observations[frame_id] = (track1.points[idx1] + track2.points[idx2]) / 2
                    merged.visibility[frame_id] = (track1.visibility[idx1] + track2.visibility[idx2]) / 2
                    merged.confidence[frame_id] = (track1.confidence[idx1] + track2.confidence[idx2]) / 2
            
            # Add observations after overlap
            for i in range(overlap_end - track2.start_frame + 1, 
                          len(track2.points)):
                frame_id = track2.start_frame + i
                merged.observations[frame_id] = track2.points[i]
                merged.visibility[frame_id] = track2.visibility[i]
                merged.confidence[frame_id] = track2.confidence[i]
        else:
            # No overlap, just add all from track2
            for i in range(len(track2.points)):
                frame_id = track2.start_frame + i
                merged.observations[frame_id] = track2.points[i]
                merged.visibility[frame_id] = track2.visibility[i]
                merged.confidence[frame_id] = track2.confidence[i]
        
        return merged
    
    def merge_all_tracks(self, tracks: List[Track]) -> List[MergedTrack]:
        """Merge all tracks from sliding windows
        
        Args:
            tracks: List of tracks from all windows
            
        Returns:
            List of merged tracks
        """
        if not tracks:
            return []
        
        # Convert initial tracks to MergedTrack format
        merged_tracks = []
        for track in tracks:
            merged = MergedTrack(
                track_id=len(merged_tracks),
                observations={},
                visibility={},
                confidence={}
            )
            for i in range(len(track.points)):
                frame_id = track.start_frame + i
                merged.observations[frame_id] = track.points[i]
                merged.visibility[frame_id] = track.visibility[i]
                merged.confidence[frame_id] = track.confidence[i]
            merged_tracks.append(merged)
        
        # Iteratively merge overlapping tracks
        changed = True
        iteration = 0
        while changed and iteration < 10:  # Max 10 iterations
            changed = False
            new_merged = []
            used = set()
            
            for i in range(len(merged_tracks)):
                if i in used:
                    continue
                
                current = merged_tracks[i]
                merged_with_any = False
                
                # Try to merge with remaining tracks
                for j in range(i + 1, len(merged_tracks)):
                    if j in used:
                        continue
                    
                    # Check if tracks can be merged
                    can_merge = self.can_merge_tracks(current, merged_tracks[j])
                    
                    if can_merge:
                        # Merge tracks
                        merged = self.merge_merged_tracks(current, merged_tracks[j])
                        new_merged.append(merged)
                        used.add(i)
                        used.add(j)
                        merged_with_any = True
                        changed = True
                        break
                
                if not merged_with_any:
                    new_merged.append(current)
                    used.add(i)
            
            merged_tracks = new_merged
            iteration += 1
            
            logger.info(f"Merge iteration {iteration}: {len(merged_tracks)} tracks remaining")
        
        # Filter by minimum track length
        filtered_tracks = []
        for track in merged_tracks:
            visible_frames = sum(1 for v in track.visibility.values() 
                               if v > self.visibility_threshold)
            if visible_frames >= self.min_track_length:
                track.track_id = len(filtered_tracks)
                filtered_tracks.append(track)
        
        logger.info(f"Merged {len(tracks)} raw tracks into {len(filtered_tracks)} final tracks")
        return filtered_tracks
    
    def can_merge_tracks(self, track1: MergedTrack, track2: MergedTrack) -> bool:
        """Check if two merged tracks can be merged
        
        Args:
            track1: First merged track
            track2: Second merged track
            
        Returns:
            True if tracks can be merged
        """
        # Find overlapping frames
        common_frames = set(track1.observations.keys()) & set(track2.observations.keys())
        
        if len(common_frames) < 3:
            return False
        
        # Compute average distance in overlapping frames
        distances = []
        for frame_id in common_frames:
            if (track1.visibility[frame_id] > self.visibility_threshold and
                track2.visibility[frame_id] > self.visibility_threshold):
                dist = np.linalg.norm(track1.observations[frame_id] - track2.observations[frame_id])
                distances.append(dist)
        
        if len(distances) < 3:
            return False
        
        avg_distance = np.mean(distances)
        return avg_distance < self.overlap_threshold
    
    def merge_merged_tracks(self, track1: MergedTrack, track2: MergedTrack) -> MergedTrack:
        """Merge two already merged tracks
        
        Args:
            track1: First merged track
            track2: Second merged track
            
        Returns:
            Combined merged track
        """
        merged = MergedTrack(
            track_id=-1,
            observations={},
            visibility={},
            confidence={}
        )
        
        # Combine all frames
        all_frames = set(track1.observations.keys()) | set(track2.observations.keys())
        
        for frame_id in all_frames:
            if frame_id in track1.observations and frame_id in track2.observations:
                # Average if both have observations
                if (track1.visibility[frame_id] > self.visibility_threshold and
                    track2.visibility[frame_id] > self.visibility_threshold):
                    merged.observations[frame_id] = (track1.observations[frame_id] + 
                                                   track2.observations[frame_id]) / 2
                    merged.visibility[frame_id] = (track1.visibility[frame_id] + 
                                                  track2.visibility[frame_id]) / 2
                    merged.confidence[frame_id] = (track1.confidence[frame_id] + 
                                                  track2.confidence[frame_id]) / 2
                elif track1.visibility[frame_id] > track2.visibility[frame_id]:
                    merged.observations[frame_id] = track1.observations[frame_id]
                    merged.visibility[frame_id] = track1.visibility[frame_id]
                    merged.confidence[frame_id] = track1.confidence[frame_id]
                else:
                    merged.observations[frame_id] = track2.observations[frame_id]
                    merged.visibility[frame_id] = track2.visibility[frame_id]
                    merged.confidence[frame_id] = track2.confidence[frame_id]
            elif frame_id in track1.observations:
                merged.observations[frame_id] = track1.observations[frame_id]
                merged.visibility[frame_id] = track1.visibility[frame_id]
                merged.confidence[frame_id] = track1.confidence[frame_id]
            else:
                merged.observations[frame_id] = track2.observations[frame_id]
                merged.visibility[frame_id] = track2.visibility[frame_id]
                merged.confidence[frame_id] = track2.confidence[frame_id]
        
        return merged
    
    def convert_to_bundle_adjustment_format(self, 
                                          merged_tracks: List[MergedTrack],
                                          n_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert merged tracks to format suitable for bundle adjustment
        
        Args:
            merged_tracks: List of merged tracks
            n_frames: Total number of frames in video
            
        Returns:
            Tuple of:
                - observations: Array of shape (n_observations, 4) with (track_id, frame_id, x, y)
                - visibility_mask: Binary array of shape (n_tracks, n_frames)
                - track_lengths: Array of shape (n_tracks,) with number of visible frames per track
        """
        observations = []
        n_tracks = len(merged_tracks)
        
        # Create visibility mask
        visibility_mask = np.zeros((n_tracks, n_frames), dtype=bool)
        
        for track in merged_tracks:
            for frame_id, point in track.observations.items():
                if track.visibility[frame_id] > self.visibility_threshold:
                    observations.append([track.track_id, frame_id, point[0], point[1]])
                    visibility_mask[track.track_id, frame_id] = True
        
        observations = np.array(observations)
        track_lengths = np.sum(visibility_mask, axis=1)
        
        logger.info(f"Converted {len(merged_tracks)} tracks with {len(observations)} observations")
        return observations, visibility_mask, track_lengths