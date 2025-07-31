"""Label-based tracker for objects we know are unique in the scene."""

import logging
from typing import Dict, Set

logger = logging.getLogger('mast3r_slam.label_based_tracker')

class LabelBasedTracker:
    """Tracks objects that we know are unique in the scene based on their labels."""
    
    def __init__(self, config: Dict):
        # Objects that we know have only one instance in typical rooms
        # Note: tables and sofas can have multiple instances, so removed from list
        self.unique_objects = config.get('unique_objects', [
            'a floor', 'a ceiling', 'a door',
            'a rug', 'a cabinet', 'a plant stand', 'a vase', 'a blanket'
        ])
        
        # Track assignments for unique objects
        self.unique_label_tracks: Dict[str, int] = {}
        
        logger.info(f"Initialized LabelBasedTracker with {len(self.unique_objects)} unique objects")
    
    def is_unique_object(self, label: str) -> bool:
        """Check if an object label represents a unique object in the scene."""
        return label in self.unique_objects
    
    def get_track_for_unique_object(self, label: str) -> int:
        """Get the track ID for a unique object, creating one if needed."""
        if label in self.unique_label_tracks:
            return self.unique_label_tracks[label]
        return None
    
    def assign_track_to_unique_object(self, label: str, track_id: int):
        """Assign a track ID to a unique object."""
        if label in self.unique_objects:
            self.unique_label_tracks[label] = track_id
            logger.info(f"Assigned track {track_id} to unique object '{label}'")
    
    def should_force_match(self, label: str, existing_track_label: str) -> bool:
        """Check if we should force a match between detection and track."""
        # If both are the same unique object type, force the match
        if label == existing_track_label and label in self.unique_objects:
            return True
        return False
    
    def get_unique_object_stats(self) -> Dict:
        """Get statistics about unique object tracking."""
        return {
            'tracked_unique_objects': len(self.unique_label_tracks),
            'unique_object_tracks': dict(self.unique_label_tracks)
        }