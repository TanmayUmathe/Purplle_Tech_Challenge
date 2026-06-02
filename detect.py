"""
pipeline/detect.py

Main Computer Vision Detection and Ingestion loop.
Why: Processes CCTV clips on CPU using YOLOv8n. Downsamples entry camera to 3 FPS 
     to handle group entry (Gap 6) and floor cameras to 1 FPS for speed. 
     Handles torso calibration during the first 60 seconds (Gap 1), tracks visitor 
     dwell, computes queue depth, and emits formatted events to the REST API.
"""

import os
import argparse
import json
import cv2
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO

from staff_detector import StaffDetector
from tracker import MultiObjectTracker, Track
from emit import format_event, emit_events

# Video start date and time parsed from CCTV overlay
START_TIMESTAMP_BASE = datetime(2026, 4, 10, 20, 9, 48)

# Dwell event emission interval in milliseconds (30 seconds)
# Why: Standard retail metrics require logging dwell events at 30-second ticks.
DWELL_TICK_MS = 30000


class StoreVideoProcessor:
    """
    Coordinates detection, tracking, staff identification, and event emission
    for a store's camera clips.
    """

    def __init__(self, video_dir: str, store_id: str, db_path: str = "store_intelligence.db"):
        self.video_dir = video_dir
        self.store_id = store_id
        
        # Load store layout config to map cameras to zones
        self.layout_config = self._load_layout_config()
        
        # Initialize YOLOv8n model
        # Why: YOLOv8n (nano) is optimized for CPU inference speed while maintaining high accuracy for person detection.
        self.model = YOLO("yolov8n.pt")
        
        # Initialize trackers for each camera
        self.trackers: Dict[str, MultiObjectTracker] = {}
        for zone_name, details in self.layout_config.get("zones", {}).items():
            cam_id = details["camera_id"]
            self.trackers[cam_id] = MultiObjectTracker(cam_id)
            
        # Global staff detector for the store
        self.staff_detector = StaffDetector()
        
        # Track event sequences per visitor session
        self.session_sequence: Dict[str, int] = {}

    def _load_layout_config(self) -> Dict[str, Any]:
        """
        Loads the store layout configuration from store_layout.json.
        """
        layout_path = "store_layout.json"
        if os.path.exists(layout_path):
            with open(layout_path, "r") as f:
                data = json.load(f)
                return data.get(self.store_id, {})
        # Fallback default configuration for ST1008
        return {
            "store_name": "Brigade_Bangalore",
            "zones": {
                "ENTRY_EXIT": {"camera_id": "CAM_3"},
                "SKINCARE": {"camera_id": "CAM_1"},
                "MAKEUP": {"camera_id": "CAM_2"},
                "PMU": {"camera_id": "CAM_4"},
                "BILLING": {"camera_id": "CAM_5"}
            }
        }

    def _increment_sequence(self, visitor_id: str) -> int:
        """
        Increments and returns the session sequence counter for a visitor.
        """
        seq = self.session_sequence.get(visitor_id, 0) + 1
        self.session_sequence[visitor_id] = seq
        return seq

    def _get_camera_fps_and_frames(self, cam_file: str) -> Tuple[float, int]:
        """
        Returns the FPS and total frames of the video clip.
        """
        cap = cv2.VideoCapture(cam_file)
        if not cap.isOpened():
            return 15.0, 2250
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps if fps > 0 else 15.0, frames

    def run_calibration(self, cam_clips: Dict[str, str]) -> None:
        """
        Calibration Phase: Collects torso region crops in the first 60 seconds
        of footage across all cameras, then fits the staff uniform signature.
        """
        print("Starting Staff Uniform Calibration Phase (First 60s of footage)...")
        
        for cam_id, clip_path in cam_clips.items():
            if not os.path.exists(clip_path):
                continue
                
            cap = cv2.VideoCapture(clip_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 15.0
            
            # Process up to 60 seconds
            max_calibration_frame = int(60 * fps)
            frame_skip = int(fps) # 1 FPS downsampling for calibration data extraction
            
            frame_idx = 0
            while cap.isOpened() and frame_idx < max_calibration_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Run YOLO person detection (class 0)
                results = self.model.predict(
                    source=frame,
                    classes=[0],
                    verbose=False,
                    conf=0.25 # Lower conf to catch uniforms in shadows/occlusions
                )
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        # Collect torso crop
                        self.staff_detector.collect_torso_sample(frame, tuple(coords))
                        
                frame_idx += frame_skip
            cap.release()
            
        # Run clustering on collected torso histograms
        self.staff_detector.calibrate_uniform()

    def process_all_clips(self) -> List[Dict[str, Any]]:
        """
        Main inference pass across all video clips.
        Processes clips in synchronized temporal increments, generating events.
        """
        # Find video clips
        cam_clips: Dict[str, str] = {}
        max_duration_sec = 0.0
        
        for zone_name, details in self.layout_config.get("zones", {}).items():
            cam_id = details["camera_id"]
            # Map camera filenames (e.g. CAM 1.mp4, CAM 3.mp4)
            filename = f"CAM {cam_id.split('_')[-1]}.mp4"
            filename_space = f"CAM {cam_id.split('_')[-1]}.mp4"
            
            path1 = os.path.join(self.video_dir, f"CAM {cam_id.split('_')[-1]}.mp4")
            path2 = os.path.join(self.video_dir, f"CAM_{cam_id.split('_')[-1]}.mp4")
            
            clip_path = path1 if os.path.exists(path1) else path2
            
            if os.path.exists(clip_path):
                cam_clips[cam_id] = clip_path
                fps, frames = self._get_camera_fps_and_frames(clip_path)
                duration = frames / fps
                max_duration_sec = max(max_duration_sec, duration)
                print(f"Mapped {cam_id} to {clip_path} ({duration:.1f}s, {fps:.1f} fps)")

        # Run uniform calibration first
        self.run_calibration(cam_clips)
        
        # Update total frames in staff detector for persistence calculations
        # Assume average of 150 seconds processed at 1 FPS = 150 evaluations
        self.staff_detector.total_frames = int(max_duration_sec)

        all_events: List[Dict[str, Any]] = []
        
        # Track visitor dwell entry timestamps: {visitor_id: {zone_id: (start_timestamp, last_dwell_emitted_timestamp)}}
        dwell_tracker: Dict[str, Dict[str, Tuple[datetime, datetime]]] = {}
        
        # Open video capture handles
        caps: Dict[str, cv2.VideoCapture] = {}
        fps_rates: Dict[str, float] = {}
        
        for cam_id, path in cam_clips.items():
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                caps[cam_id] = cap
                fps_rates[cam_id] = cap.get(cv2.CAP_PROP_FPS) or 15.0

        print(f"Processing synchronized streams up to {max_duration_sec:.1f} seconds...")

        # Process streams in 1-second ticks
        # Why: Aligning video processing to 1-second slices coordinates multi-camera cross-tracking.
        for tick_sec in range(int(max_duration_sec)):
            current_timestamp = START_TIMESTAMP_BASE + timedelta(seconds=tick_sec)
            timestamp_str = current_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Map zones to active visitors at this second (used for billing queue depth)
            billing_zone_visitors: List[str] = []

            # 1. First Pass: Run detection on floor cameras to update track states and collect billing list
            active_detections_per_camera: Dict[str, List[Any]] = {}
            
            for cam_id, cap in caps.items():
                fps = fps_rates[cam_id]
                
                # Gap 6: Variable frame skips
                # Entry Camera CAM_3 gets 3 FPS (process every 5th frame for group entry protection)
                # All other cameras get 1 FPS (process every 15th frame for CPU optimization)
                frames_in_tick = 3 if cam_id == "CAM_3" else 1
                frame_interval = int(fps / frames_in_tick)
                
                # Fetch detections for the frames in this 1-second tick
                tick_detections = []
                
                for f_offset in range(frames_in_tick):
                    frame_idx = int(tick_sec * fps) + (f_offset * frame_interval)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame_h, frame_w = frame.shape[:2]
                    
                    # Run YOLO person detection
                    results = self.model.predict(
                        source=frame,
                        classes=[0],
                        verbose=False,
                        conf=0.25 # Do not suppress low confidence (Gap 7)
                    )
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            coords = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf[0].cpu().numpy())
                            
                            # Extract torso histogram
                            torso_hist = self.staff_detector._compute_torso_histogram(frame, tuple(coords))
                            tick_detections.append((tuple(coords), conf, torso_hist, frame_w, frame_h))
                
                active_detections_per_camera[cam_id] = tick_detections

            # 2. Second Pass: Process tracks and emit transition and dwell events
            for cam_id, detections in active_detections_per_camera.items():
                tracker = self.trackers[cam_id]
                
                # Map camera to store zone from config
                zone_id = None
                for z_name, details in self.layout_config.get("zones", {}).items():
                    if details["camera_id"] == cam_id:
                        zone_id = z_name
                        break
                
                # Format tracker detections input: (bbox, conf, torso_hist)
                tracker_inputs = [(d[0], d[1], d[2]) for d in detections]
                
                # If we have no frame dimensions, fallback to 1920x1080
                fw = detections[0][3] if detections else 1920
                fh = detections[0][4] if detections else 1080
                
                # Update tracker
                track_events = tracker.update_tracks(tracker_inputs, timestamp_str, fw, fh)
                
                # Update queue list for billing camera
                if zone_id == "BILLING":
                    for t_id, track in tracker.active_tracks.items():
                        # Determine if staff using ensemble
                        is_staff, _ = self.staff_detector.evaluate_track(
                            frame=np.zeros((fh, fw, 3), dtype=np.uint8), # BBox coordinate signature suffices
                            bbox=track.bbox,
                            visible_frames=track.visible_frames,
                            zones_visited=track.zones_visited
                        )
                        if not is_staff:
                            billing_zone_visitors.append(track.visitor_id)

                for track, event_action in track_events:
                    # Evaluate staff classification (Gap 1)
                    is_staff, staff_conf = self.staff_detector.evaluate_track(
                        frame=np.zeros((fh, fw, 3), dtype=np.uint8),
                        bbox=track.bbox,
                        visible_frames=track.visible_frames,
                        zones_visited=track.zones_visited
                    )
                    
                    # Track zone history
                    if zone_id and zone_id not in track.zones_visited:
                        track.zones_visited.append(zone_id)
                    
                    # Format corresponding behavior event
                    e_type = event_action
                    actual_zone = zone_id if zone_id not in ("ENTRY_EXIT", "ENTRY") else None
                    
                    # Map EXIT to ZONE_EXIT for floor/billing cameras
                    if event_action == "EXIT" and zone_id != "ENTRY_EXIT":
                        e_type = "ZONE_EXIT"
                        
                    # Map ENTRY to ZONE_ENTER for floor/billing cameras
                    if event_action == "ENTRY" and zone_id != "ENTRY_EXIT":
                        e_type = "ZONE_ENTER"
                        
                    # Queue depth logic on Billing Join
                    q_depth = None
                    if zone_id == "BILLING" and e_type == "ZONE_ENTER":
                        # If billing zone already has visitors, this is a join event
                        if len(billing_zone_visitors) > 1:
                            e_type = "BILLING_QUEUE_JOIN"
                            q_depth = len(billing_zone_visitors)

                    # Reset/initialize dwell trackers
                    if e_type in ("ENTRY", "ZONE_ENTER", "BILLING_QUEUE_JOIN"):
                        if track.visitor_id not in dwell_tracker:
                            dwell_tracker[track.visitor_id] = {}
                        dwell_tracker[track.visitor_id][zone_id or "STORE"] = (current_timestamp, current_timestamp)
                        
                    elif e_type in ("EXIT", "ZONE_EXIT"):
                        # Clear dwell tracking
                        if track.visitor_id in dwell_tracker and (zone_id or "STORE") in dwell_tracker[track.visitor_id]:
                            del dwell_tracker[track.visitor_id][zone_id or "STORE"]

                    seq = self._increment_sequence(track.visitor_id)
                    
                    evt = format_event(
                        store_id=self.store_id,
                        camera_id=cam_id,
                        visitor_id=track.visitor_id,
                        event_type=e_type,
                        timestamp_str=timestamp_str,
                        zone_id=actual_zone,
                        dwell_ms=0,
                        is_staff=is_staff,
                        confidence=track.low_confidence_frames / track.visible_frames if track.low_confidence_frames > 0 else 1.0,
                        queue_depth=q_depth,
                        session_seq=seq
                    )
                    all_events.append(evt)

            # 3. Third Pass: Dwell calculations (check active tracks for 30s ticks)
            for cam_id, tracker in self.trackers.items():
                zone_id = None
                for z_name, details in self.layout_config.get("zones", {}).items():
                    if details["camera_id"] == cam_id:
                        zone_id = z_name
                        break
                
                for t_id, track in tracker.active_tracks.items():
                    # Evaluate staff
                    is_staff, _ = self.staff_detector.evaluate_track(
                        frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
                        bbox=track.bbox,
                        visible_frames=track.visible_frames,
                        zones_visited=track.zones_visited
                    )
                    
                    z_key = zone_id or "STORE"
                    if track.visitor_id in dwell_tracker and z_key in dwell_tracker[track.visitor_id]:
                        start_time, last_emitted = dwell_tracker[track.visitor_id][z_key]
                        elapsed_ms = int((current_timestamp - start_time).total_seconds() * 1000)
                        ms_since_last_emit = int((current_timestamp - last_emitted).total_seconds() * 1000)
                        
                        # Emit a ZONE_DWELL event every 30 seconds
                        if elapsed_ms >= DWELL_TICK_MS and ms_since_last_emit >= DWELL_TICK_MS:
                            # Update last emitted time
                            dwell_tracker[track.visitor_id][z_key] = (start_time, current_timestamp)
                            
                            seq = self._increment_sequence(track.visitor_id)
                            evt = format_event(
                                store_id=self.store_id,
                                camera_id=cam_id,
                                visitor_id=track.visitor_id,
                                event_type="ZONE_DWELL",
                                timestamp_str=timestamp_str,
                                zone_id=zone_id if zone_id != "ENTRY_EXIT" else None,
                                dwell_ms=elapsed_ms,
                                is_staff=is_staff,
                                confidence=1.0,
                                session_seq=seq
                            )
                            all_events.append(evt)

        # Release video handles
        for cap in caps.values():
            cap.release()

        print(f"Processing complete. Generated {len(all_events)} behavioral events.")
        return all_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Store Intelligence CCTV Detection Pipeline.")
    parser.add_argument("--video_dir", type=str, default="CCTV Footage", help="Directory holding mp4 clips")
    parser.add_argument("--store_id", type=str, default="ST1008", help="Store ID from layout config")
    parser.add_argument("--emit_api", action="store_true", help="Post events to API ingest URL")
    
    args = parser.parse_args()
    
    processor = StoreVideoProcessor(args.video_dir, args.store_id)
    events = processor.process_all_clips()
    
    # Emit events
    emit_events(events, batch_mode=args.emit_api)
