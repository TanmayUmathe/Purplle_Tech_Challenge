"""
pipeline/tracker.py

Multi-Object Tracker and Cross-Camera Re-ID engine.
Why: Face blur makes face-feature tracking impossible. Torso-based color histogram 
     matching combined with spatial/temporal gating provides a robust, CPU-friendly 
     tracking and Re-ID mechanism. Incorporates a 60s lost track buffer to solve 
     the re-entry inflation problem.
"""

import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

# Re-entry window in seconds
# Why: 60s allows customers to take a call or step out briefly without starting a new session.
REENTRY_WINDOW_SECONDS = 60.0

# Cosine similarity threshold for re-entry linking
# Why: 0.72 is calibrated for matching the same individual across exits on the entry camera.
REENTRY_SIM_THRESHOLD = 0.72

# Plausible spatial shift limit for re-entry in normalized coordinates [0.0 - 1.0]
# Why: 0.4 represents the physical constraint of moving through the threshold gate region.
REENTRY_SPATIAL_GATE = 0.4

# Max seconds a track can be missing before being declared lost/archived
# Why: 3.0 seconds handles brief occlusion behind displays or pillars.
MAX_LOST_TIME_SECONDS = 3.0

# Detection confidence score threshold for high vs low quality
# Why: Below 0.4, detections are prone to occlusion; we propagate rather than drop.
LOW_CONF_LIMIT = 0.4


class Track:
    """
    Represents an active or lost person track within the store.
    Stores trajectory coordinates, zone history, and torso color signatures.
    """

    def __init__(
        self, 
        track_id: int, 
        visitor_id: str, 
        bbox: Tuple[int, int, int, int], 
        timestamp_str: str, 
        torso_hist: Optional[np.ndarray] = None
    ):
        self.track_id = track_id
        self.visitor_id = visitor_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.first_seen = timestamp_str
        self.last_seen = timestamp_str
        self.torso_hist = torso_hist
        self.had_exit = False
        self.low_confidence_frames = 0
        self.visible_frames = 1
        self.zones_visited: List[str] = []

    def get_centroid_normalized(self, frame_w: int, frame_h: int) -> Tuple[float, float]:
        """
        Computes the normalized (0-1) coordinates of the track's bounding box centroid.
        Why: Normalization ensures spatial metrics remain identical across camera resolutions.
        """
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0 / frame_w
        cy = (y1 + y2) / 2.0 / frame_h
        return cx, cy

    def update(
        self, 
        bbox: Tuple[int, int, int, int], 
        timestamp_str: str, 
        confidence: float, 
        torso_hist: Optional[np.ndarray] = None
    ) -> None:
        """
        Updates the track's boundary box, timestamp, confidence logs, and appearance signature.
        Why: Integrates confidence checks (Gap 7) to track low-confidence periods without dropping.
        """
        self.bbox = bbox
        self.last_seen = timestamp_str
        self.visible_frames += 1
        
        if confidence < LOW_CONF_LIMIT:
            self.low_confidence_frames += 1

        # Exponential moving average for appearance signature to handle profile transitions
        if torso_hist is not None:
            if self.torso_hist is None:
                self.torso_hist = torso_hist
            else:
                # 0.8 EMA weighting preserves long-term color identity while adjusting for skew
                self.torso_hist = 0.8 * self.torso_hist + 0.2 * torso_hist
                norm = np.linalg.norm(self.torso_hist)
                if norm > 0:
                    self.torso_hist = self.torso_hist / norm


class MultiObjectTracker:
    """
    Manages active and lost tracks across the camera networks.
    Enforces re-entry checking (Gap 2) and cross-camera Re-ID linking.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.active_tracks: Dict[int, Track] = {}
        # Buffer of lost tracks: {track_id: (Track, lost_timestamp_str)}
        self._lost_buffer: Dict[int, Tuple[Track, str]] = {}
        self._next_track_id = 1

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """
        Parses UTC ISO timestamp strings into datetime objects for calculation.
        """
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

    def update_tracks(
        self, 
        detections: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray]]], 
        timestamp_str: str, 
        frame_w: int, 
        frame_h: int
    ) -> List[Tuple[Track, str]]:
        """
        Associates incoming frame detections with existing active tracks.
        Handles track initiation, loss buffering, and re-entry checks on entry cameras.
        Returns: List of (Track, event_type) to emit.
        """
        emitted_events: List[Tuple[Track, str]] = []
        matched_detections: Set[int] = set()
        matched_tracks: Set[int] = set()

        # Step 1: Attempt matching with active tracks using IoU + Centroid Distance
        for t_id, track in list(self.active_tracks.items()):
            best_score = -1.0
            best_det_idx = -1
            
            # Compute normalized centroid of track
            tx, ty = track.get_centroid_normalized(frame_w, frame_h)

            for d_idx, (d_bbox, d_conf, d_hist) in enumerate(detections):
                if d_idx in matched_detections:
                    continue
                
                # Compute centroid of detection
                dx1, dy1, dx2, dy2 = d_bbox
                dcx = (dx1 + dx2) / 2.0 / frame_w
                dcy = (dy1 + dy2) / 2.0 / frame_h
                
                # Centroid distance cost (normalized)
                dist = np.sqrt((tx - dcx) ** 2 + (ty - dcy) ** 2)
                
                # Check spatial closeness
                # Why: 0.25 spatial threshold limits frame-to-frame matching to plausible human speeds.
                if dist < 0.25:
                    score = 1.0 - dist
                    # Include appearance check if histograms exist
                    if track.torso_hist is not None and d_hist is not None:
                        sim = np.dot(track.torso_hist, d_hist)
                        score += sim
                    
                    if score > best_score:
                        best_score = score
                        best_det_idx = d_idx

            if best_det_idx >= 0:
                d_bbox, d_conf, d_hist = detections[best_det_idx]
                track.update(d_bbox, timestamp_str, d_conf, d_hist)
                matched_tracks.add(t_id)
                matched_detections.add(best_det_idx)

        # Step 2: Declare unmatched active tracks as lost if they exceed the time limit
        current_time = self._parse_timestamp(timestamp_str)
        for t_id, track in list(self.active_tracks.items()):
            if t_id not in matched_tracks:
                last_seen_time = self._parse_timestamp(track.last_seen)
                time_lost = (current_time - last_seen_time).total_seconds()
                
                if time_lost > MAX_LOST_TIME_SECONDS:
                    # Move track to lost buffer
                    self._lost_buffer[t_id] = (track, track.last_seen)
                    # Remove from active list
                    del self.active_tracks[t_id]
                    # Emit EXIT event
                    emitted_events.append((track, "EXIT"))

        # Step 3: Handle unmatched detections (New Tracks or Re-entries)
        for d_idx, (d_bbox, d_conf, d_hist) in enumerate(detections):
            if d_idx in matched_detections:
                continue

            # Compute normalized centroid of new detection
            dx1, dy1, dx2, dy2 = d_bbox
            dcx = (dx1 + dx2) / 2.0 / frame_w
            dcy = (dy1 + dy2) / 2.0 / frame_h

            is_reentry_match = False
            matched_lost_id = -1

            # Re-entry check is exclusive to CAM_3 (Entry/Exit camera)
            if self.camera_id == "CAM_3" and d_hist is not None:
                # Scan lost buffer for match within the last 60 seconds
                for lost_id, (lost_track, lost_timestamp_str) in list(self._lost_buffer.items()):
                    lost_time = self._parse_timestamp(lost_timestamp_str)
                    sec_since_lost = (current_time - lost_time).total_seconds()

                    # Filter tracks in buffer outside the 60-second window
                    if sec_since_lost > REENTRY_WINDOW_SECONDS:
                        del self._lost_buffer[lost_id]
                        continue

                    # Compute appearance cosine similarity
                    if lost_track.torso_hist is not None:
                        sim = np.dot(d_hist, lost_track.torso_hist)
                        
                        # Compute spatial distance to the last known position
                        lx, ly = lost_track.get_centroid_normalized(frame_w, frame_h)
                        s_dist = np.sqrt((dcx - lx) ** 2 + (dcy - ly) ** 2)

                        if sim > REENTRY_SIM_THRESHOLD and s_dist < REENTRY_SPATIAL_GATE:
                            is_reentry_match = True
                            matched_lost_id = lost_id
                            break

            if is_reentry_match and matched_lost_id >= 0:
                # Reactivate track with same visitor_id
                reactivated_track, _ = self._lost_buffer[matched_lost_id]
                reactivated_track.update(d_bbox, timestamp_str, d_conf, d_hist)
                reactivated_track.had_exit = True
                
                # Move back to active list and remove from buffer
                self.active_tracks[matched_lost_id] = reactivated_track
                del self._lost_buffer[matched_lost_id]
                
                # Emit REENTRY event
                emitted_events.append((reactivated_track, "REENTRY"))
            else:
                # Create a completely new track
                track_id = self._next_track_id
                self._next_track_id += 1
                
                # Generate unique visitor_id
                visitor_id = f"VIS_{uuid.uuid4().hex[:6]}"
                new_track = Track(track_id, visitor_id, d_bbox, timestamp_str, d_hist)
                
                self.active_tracks[track_id] = new_track
                
                # Emit ENTRY event
                emitted_events.append((new_track, "ENTRY"))

        return emitted_events
