"""
tracker.py - High-Performance 3D Multi-Object Tracker for Autonomous Driving.
Combines 3D Kalman Filtering, One-Euro Jitter Filtering, Trajectory Prediction,
and Collision Risk Assessment (TTC / FCW).
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import time
import numpy as np


class OneEuroFilter:
    """
    1€ Filter: Adaptive low-pass filter designed specifically to eliminate high-frequency
    human/camera tracking jitter while avoiding phase lag during rapid velocity changes.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def filter(self, x: float, t: Optional[float] = None) -> float:
        now = t if t is not None else time.time()
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = now
            return x

        dt = max(1e-4, now - (self.t_prev if self.t_prev is not None else now - 0.033))
        self.t_prev = now

        # Filter derivative
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        edx = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        self.dx_prev = edx

        # Dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(dt, cutoff)
        filtered = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = filtered
        return float(filtered)

    def _alpha(self, dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / dt)


class KalmanFilter3D:
    """
    Constant Velocity 3D Ground Kinematic Kalman Filter.
    State vector: [X, Z, vx, vz]^T
    """

    def __init__(self, init_x: float, init_z: float):
        # State [X, Z, vx, vz]
        self.x = np.array([init_x, init_z, 0.0, 0.0], dtype=np.float64)

        # State covariance P
        self.P = np.diag([0.5, 2.0, 9.0, 16.0]).astype(np.float64)

        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

    def predict(self, dt: float):
        dt = max(0.005, min(dt, 0.2))

        # State transition F
        F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Process noise Q (continuous white noise acceleration model)
        q_acc = 3.0 # m/s^2
        q1 = (dt**3) / 3.0 * q_acc
        q2 = (dt**2) / 2.0 * q_acc
        q3 = dt * q_acc
        Q = np.array([
            [q1, 0.0, q2, 0.0],
            [0.0, q1, 0.0, q2],
            [q2, 0.0, q3, 0.0],
            [0.0, q2, 0.0, q3]
        ], dtype=np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas_x: float, meas_z: float):
        z = np.array([meas_x, meas_z], dtype=np.float64)

        # Measurement noise R: distance-dependent uncertainty
        r_x = 0.35**2
        r_z = max(0.8, 0.045 * meas_z)**2
        R = np.diag([r_x, r_z]).astype(np.float64)

        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P


class TrackedObject:
    """
    Represents an actively tracked road actor (vehicle, pedestrian, two-wheeler)
    with 3D kinematic filtering, visual smoothing, and safety metrics.
    """

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

        # 3D Kalman Filter
        self.kf = KalmanFilter3D(self.X, self.Z)

        # 1€ Filters for ultra-smooth display coordinates
        self.filter_x = OneEuroFilter(min_cutoff=1.5, beta=0.08)
        self.filter_z = OneEuroFilter(min_cutoff=1.5, beta=0.08)

        # 1€ Filters for 2D bounding box
        self.box_filters = [OneEuroFilter(min_cutoff=2.0, beta=0.05) for _ in range(4)]

        # Kinematic state
        self.vx = 0.0 # m/s
        self.vz = 0.0 # m/s
        self.yaw = 0.0 # radians (0 = forward along +Z)
        self.speed_kmh = 0.0
        self.ttc: Optional[float] = None
        self.in_ego_lane: bool = False
        self.warning_level: str = 'normal' # 'normal', 'caution', 'critical'

        # Trajectory trail (last N ground coordinates)
        self.trail: List[Tuple[float, float]] = [(self.X, self.Z)]
        self.max_trail = 24

        # Lifecycle
        self.hits = 1
        self.age = 1
        self.missed_frames = 0
        self.is_coasting = False

    def predict(self, dt: float):
        """Propagate state forward using Kalman model."""
        self.kf.predict(dt)
        self.X = float(self.kf.x[0])
        self.Z = float(self.kf.x[1])
        self.vx = float(self.kf.x[2])
        self.vz = float(self.kf.x[3])

    def update(
        self,
        box_2d: List[float],
        confidence: float,
        pos_3d: Tuple[float, float],
        corners_3d: np.ndarray,
        dt: float
    ):
        """Update track with fresh sensor measurement."""
        meas_x, meas_z = pos_3d

        # 1. Kalman measurement update
        self.kf.update(meas_x, meas_z)
        raw_x = float(self.kf.x[0])
        raw_z = float(self.kf.x[1])
        self.vx = float(self.kf.x[2])
        self.vz = float(self.kf.x[3])

        # 2. 1€ Filter smoothing for visual rendering
        now = time.time()
        self.X = self.filter_x.filter(raw_x, now)
        self.Z = self.filter_z.filter(raw_z, now)

        # 3. 2D Bounding box smoothing
        self.box_2d = [
            self.box_filters[i].filter(box_2d[i], now)
            for i in range(4)
        ]
        self.confidence = 0.7 * self.confidence + 0.3 * confidence
        self.corners_3d = corners_3d

        # 4. Heading & Orientation (Yaw)
        speed_mps = math.sqrt(self.vx**2 + self.vz**2)
        self.speed_kmh = float(speed_mps * 3.6)

        if speed_mps > 1.2:
            # Moving: align with velocity vector
            target_yaw = math.atan2(self.vx, self.vz)
            self.yaw = 0.75 * self.yaw + 0.25 * target_yaw
        else:
            # Stationary (e.g. stopped at traffic light):
            # Infer orientation from 2D bounding box aspect ratio
            bw = self.box_2d[2] - self.box_2d[0]
            bh = self.box_2d[3] - self.box_2d[1]
            if bw / max(bh, 1.0) > 1.35:
                # Perpendicular crossing traffic
                target_yaw = math.pi / 2.0 if self.X > 0 else -math.pi / 2.0
            else:
                # Longitudinal traffic (aligned with road)
                target_yaw = 0.0
            self.yaw = 0.85 * self.yaw + 0.15 * target_yaw

        # 5. Collision Risk & TTC Assessment
        self.in_ego_lane = abs(self.X) <= 1.95
        if self.vz < -0.05:
            # Approaching host vehicle
            closing_speed = -self.vz
            self.ttc = float(self.Z / max(closing_speed, 0.05))
        else:
            self.ttc = None


        # Warning levels
        if self.in_ego_lane and self.ttc is not None and self.ttc < 2.5:
            self.warning_level = 'critical'
        elif self.in_ego_lane and ((self.ttc is not None and self.ttc < 4.2) or self.Z < 10.0):
            self.warning_level = 'caution'
        elif self.class_name in ('person', 'bicycle') and abs(self.X) < 2.5 and self.Z < 12.0:
            self.warning_level = 'critical' if self.Z < 7.0 else 'caution'
        else:
            self.warning_level = 'normal'

        # Trajectory trail
        self.trail.append((self.X, self.Z))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

        self.hits += 1
        self.age += 1
        self.missed_frames = 0
        self.is_coasting = False

    def predict_gap(self, dt: float):
        """Called on frame where object wasn't matched (occlusion coasting)."""
        self.predict(dt)
        self.age += 1
        self.missed_frames += 1
        self.is_coasting = True
        self.trail.append((self.X, self.Z))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)


class CarVisionTracker:
    """
    Robust Multi-Object 3D Tracker with Gated Spatial Association and Occlusion Coasting.
    """

    def __init__(self, max_missed: int = 10, metric_dist_thresh: float = 4.8):
        self.next_id = 1
        self.tracks: Dict[int, TrackedObject] = {}
        self.max_missed = max_missed
        self.dist_thresh = metric_dist_thresh

    def update(
        self,
        detections: List[Dict[str, Any]],
        dt: float = 0.033
    ) -> List[TrackedObject]:
        """
        Associates detections with existing tracks using 3D metric gating + 2D IoU.
        """
        # 1. Kalman prediction for all existing tracks
        for trk in self.tracks.values():
            trk.predict(dt)

        matched_tracks = set()
        matched_dets = set()

        track_ids = list(self.tracks.keys())

        if track_ids and detections:
            # Build cost matrix based on 3D Euclidean distance
            for d_idx, det in enumerate(detections):
                d_x, d_z = det['pos_3d']
                d_box = det['box_2d']
                d_cls = det['class_name']

                best_t_id = None
                best_cost = 999.0

                for t_id in track_ids:
                    if t_id in matched_tracks:
                        continue
                    trk = self.tracks[t_id]

                    # Class matching bonus
                    if trk.class_name != d_cls:
                        # Allow vehicle subtype cross-match (car vs truck/bus)
                        veh_classes = {'car', 'truck', 'bus'}
                        if not (trk.class_name in veh_classes and d_cls in veh_classes):
                            continue

                    # 3D Euclidean ground distance
                    dist_3d = math.hypot(trk.X - d_x, trk.Z - d_z)

                    # 2D IoU bonus
                    iou = self._calc_iou(trk.box_2d, d_box)
                    cost = dist_3d - (iou * 2.0)

                    # Spatial gating threshold
                    max_allowed_jump = self.dist_thresh + (trk.speed_kmh / 3.6) * dt * 2.5
                    if dist_3d < max_allowed_jump and cost < best_cost:
                        best_cost = cost
                        best_t_id = t_id

                if best_t_id is not None and best_cost < self.dist_thresh:
                    matched_tracks.add(best_t_id)
                    matched_dets.add(d_idx)
                    self.tracks[best_t_id].update(
                        box_2d=d_box,
                        confidence=det['confidence'],
                        pos_3d=(d_x, d_z),
                        corners_3d=det['corners_3d'],
                        dt=dt
                    )

        # 2. Update unmatched tracks with occlusion prediction
        unmatched_track_ids = [t_id for t_id in track_ids if t_id not in matched_tracks]
        for t_id in unmatched_track_ids:
            trk = self.tracks[t_id]
            trk.predict_gap(dt)

        # 3. Create new tracks for unmatched high-confidence detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets and det['confidence'] >= 0.22:
                new_trk = TrackedObject(
                    track_id=self.next_id,
                    box_2d=det['box_2d'],
                    class_name=det['class_name'],
                    confidence=det['confidence'],
                    pos_3d=det['pos_3d'],
                    corners_3d=det['corners_3d']
                )
                self.tracks[self.next_id] = new_trk
                self.next_id += 1

        # 4. Prune dead tracks
        dead_ids = [
            t_id for t_id, trk in self.tracks.items()
            if trk.missed_frames > self.max_missed or trk.Z > 140.0 or trk.Z < 0.2
        ]
        for t_id in dead_ids:
            del self.tracks[t_id]

        # 5. Return active confirmed tracks (must have >= 2 hits or close proximity)
        active = [
            trk for trk in self.tracks.values()
            if (trk.hits >= 2 or trk.Z < 15.0) and trk.missed_frames <= 3
        ]
        return active

    @staticmethod
    def _calc_iou(boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return float(iou)
