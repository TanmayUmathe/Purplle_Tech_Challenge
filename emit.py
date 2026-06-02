"""
pipeline/emit.py

Event emission module for the Store Intelligence System.
Why: Converts tracker trajectories into standardized business events.
     Translates frame offsets into ISO-8601 UTC timestamps, propagates raw detection
     confidence levels, and pushes batches to the REST API ingest endpoint.
"""

import os
import uuid
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# API endpoint for event ingestion
# Why: Allows uploading to live Render instances during deployment by setting INGEST_URL env var
INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8000/events/ingest")



def format_event(
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp_str: str,
    zone_id: Optional[str] = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = 1.0,
    queue_depth: Optional[int] = None,
    sku_zone: Optional[str] = None,
    session_seq: Optional[int] = None
) -> Dict[str, Any]:
    """
    Formats observation data into the required JSON event schema.
    Why: Enforces that confidence is a float, generates globally unique UUID event IDs,
         and formats metadata attributes (Gap 7).
    """
    event = {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp_str,
        "zone_id": zone_id,
        "dwell_ms": int(dwell_ms),
        "is_staff": is_staff,
        "confidence": float(confidence),
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": sku_zone,
            "session_seq": session_seq
        }
    }
    return event


def emit_events(events: List[Dict[str, Any]], batch_mode: bool = True) -> bool:
    """
    Emits events to the local file logs and sends them to the API endpoint.
    Why: Batch mode limits network overhead by sending events in groups of up to 500.
    """
    if not events:
        return True

    # Log events to local JSONL for verification and recovery
    log_file = "emitted_events.jsonl"
    with open(log_file, "a") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    # If API is running, post events
    if batch_mode:
        # Cap batches at 500 events
        for i in range(0, len(events), 500):
            batch = events[i:i + 500]
            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                    INGEST_URL, 
                    json={"events": batch}, 
                    headers=headers,
                    timeout=5.0
                )
                if response.status_code in (200, 201, 202):
                    print(f"Emitted batch of {len(batch)} events to API successfully.")
                else:
                    print(f"API rejected batch with status {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                # Fallback to silent logging on local file if database/API is offline
                print(f"API ingest offline. Logged {len(batch)} events locally. Error: {e}")
                return False
    return True
