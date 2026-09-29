"""
tracker.py - Multi-Object Tracker with 3D Metric State Estimation,
Velocity Filtering, and Collision Risk Assessment.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class TrackedObject:
    def __init__(
        self,
        track_id: int,
        box_2d: List[float],
        class_name: str,
        confidence: float,
        pos_3d: Tuple[float, float],
        corners_3d: np.ndarray
    ):
        self.track_id = track_id
        self.box_2d = list(box_2d)
        self.class_name = class_name
        self.confidence = confidence
        self.X, self.Z = pos_3d
        self.corners_3d = corners_3d

        # Kinematic state
        self.vx = 0.0 # m/s
        self.vz = 0.0 # m/s
        self.speed_kmh = 0.0
        self.ttc: Optional[float] = None
        self.in_ego_lane: bool = False
        self.warning_level: str = 'normal' # 'normal', 'caution', 'warning', 'critical'

        # Trajectory trail (last N ground coordinates)
        self.trail: List[Tuple[float, float]] = [(self.X, self.Z)]
        self.max_trail = 20

        # Lifecyle
        self.hits = 1
        self.age = 1
        self.missed_frames = 0

    def update(
        self,
        box_2d: List[float],
        confidence: float,
        pos_3d: Tuple[float, float],
        corners_3d: np.ndarray,
        dt: float
    ):
        new_x, new_z = pos_3d
        if dt > 0.001:
            raw_vx = (new_x - self.X) / dt
            raw_vz = (new_z - self.Z) / dt
            # Alpha filter for velocity smoothing
            alpha = 0.4
            self.vx = alpha * raw_vx + (1.0 - alpha) * self.vx
            self.vz = alpha * raw_vz + (1.0 - alpha) * self.vz
            # Relative speed magnitude
            rel_speed_mps = np.sqrt(self.vx**2 + self.vz**2)
            self.speed_kmh = float(rel_speed_mps * 3.6)

        self.box_2d = list(box_2d)
        self.confidence = confidence
        self.X, self.Z = new_x, new_z
        self.corners_3d = corners_3d

        self.trail.append((self.X, self.Z))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

        self.hits += 1
        self.age += 1
        self.missed_frames = 0

    def predict_gap(self):
        """Called on frame where object wasn't matched."""
        self.age += 1
        self.missed_frames += 1


class CarVisionTracker:
    """
    Hungarian-like Euclidean and IoU association tracker for autonomous driving perception.
    """

    def __init__(self, max_missed: int = 6, metric_dist_thresh: float = 4.0):
        self.next_id = 1
        self.tracks: Dict[int, TrackedObject] = {}
        self.max_missed = max_missed
        self.dist_thresh = metric_dist_thresh # Max spatial jump in meters between frames

    def update(
        self,
        detections: List[Dict[str, Any]],
        dt: float = 0.05
    ) -> List[TrackedObject]:
        """
        Args:
            detections: List of dicts with:
                - 'box_2d': [x1, y1, x2, y2]
                - 'class_name': str
                - 'confidence': float
                - 'pos_3d': (X, Z) in meters
                - 'corners_3d': np.ndarray (8, 3)
            dt: Time delta between frames in seconds
        """
        active_track_ids = list(self.tracks.keys())
        det_count = len(detections)
        trk_count = len(active_track_ids)

        matched_tracks = set()
        matched_dets = set()

        if trk_count > 0 and det_count > 0:
            # Build distance cost matrix based on 3D metric ground distance
            cost_matrix = np.zeros((trk_count, det_count))
            for i, tid in enumerate(active_track_ids):
                trk = self.tracks[tid]
                for j, det in enumerate(detections):
                    dx = trk.X - det['pos_3d'][0]
                    dz = trk.Z - det['pos_3d'][1]
                    dist = np.sqrt(dx**2 + dz**2)
                    # Penalize class mismatch
                    if trk.class_name != det['class_name']:
                        dist += 10.0
                    cost_matrix[i, j] = dist

            # Greedy matching for minimal cost
            while True:
                min_val = cost_matrix.min()
                if min_val > self.dist_thresh:
                    break
                i, j = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
                if i in matched_tracks or j in matched_dets:
                    cost_matrix[i, j] = 999.0
                    continue

                tid = active_track_ids[i]
                det = detections[j]
                self.tracks[tid].update(
                    box_2d=det['box_2d'],
                    confidence=det['confidence'],
                    pos_3d=det['pos_3d'],
                    corners_3d=det['corners_3d'],
                    dt=dt
                )
                matched_tracks.add(i)
                matched_dets.add(j)
                cost_matrix[i, :] = 999.0
                cost_matrix[:, j] = 999.0

        # Unmatched existing tracks
        for i, tid in enumerate(active_track_ids):
            if i not in matched_tracks:
                self.tracks[tid].predict_gap()

        # Unmatched detections -> spawn new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_track = TrackedObject(
                    track_id=self.next_id,
                    box_2d=det['box_2d'],
                    class_name=det['class_name'],
                    confidence=det['confidence'],
                    pos_3d=det['pos_3d'],
                    corners_3d=det['corners_3d']
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Prune stale tracks
        dead_ids = [tid for tid, trk in self.tracks.items() if trk.missed_frames > self.max_missed]
        for tid in dead_ids:
            del self.tracks[tid]

        # Assess risk & lane status
        lead_vehicle = None
        min_lead_z = 999.0

        for trk in self.tracks.values():
            trk.in_ego_lane = abs(trk.X) < 1.85
            # TTC computation
            if trk.vz < -0.2: # Closing in
                trk.ttc = float(trk.Z / max(-trk.vz, 0.1))
            else:
                trk.ttc = None

            # Warning level
            if trk.class_name in ('person', 'bicycle'):
                if abs(trk.X) < 2.5 and trk.Z < 25.0:
                    trk.warning_level = 'critical'
                elif abs(trk.X) < 4.5 and trk.Z < 40.0:
                    trk.warning_level = 'caution'
                else:
                    trk.warning_level = 'normal'
            else:
                if trk.in_ego_lane and trk.ttc is not None and trk.ttc < 2.2:
                    trk.warning_level = 'critical'
                elif trk.in_ego_lane and trk.ttc is not None and trk.ttc < 3.5:
                    trk.warning_level = 'warning'
                elif trk.in_ego_lane and trk.Z < 15.0:
                    trk.warning_level = 'caution'
                else:
                    trk.warning_level = 'normal'

            if trk.in_ego_lane and trk.Z > 1.0 and trk.Z < min_lead_z:
                min_lead_z = trk.Z
                lead_vehicle = trk

        return list(self.tracks.values())
