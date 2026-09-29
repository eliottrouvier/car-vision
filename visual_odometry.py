"""
visual_odometry.py - Robust Real-Time Camera Motion & Reference Frame Classifier.
Distinguishes between Mobile Dashcam (onboard moving vehicle) and Fixed Camera (surveillance/static),
and estimates ego-vehicle speed and Focus of Expansion (FOE).
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class VisualOdometry:
    """
    Analyzes frame-to-frame optical flow on static background elements
    to determine if the camera is in motion and estimate vehicle speed.
    """

    def __init__(self, history_len: int = 15):
        self.prev_gray: Optional[np.ndarray] = None
        self.feature_pts: Optional[np.ndarray] = None

        # State estimates
        self.is_camera_moving: bool = True
        self.confidence: float = 0.8
        self.ego_speed_mps: float = 0.0
        self.ego_speed_kmh: float = 0.0
        self.foe_u: Optional[float] = None
        self.foe_v: Optional[float] = None

        # Filter buffers
        self.flow_magnitudes = []
        self.vertical_flows = []
        self.history_len = history_len

        # Moving threshold in pixels/frame
        self.motion_threshold = 0.65

    def update(
        self,
        frame: np.ndarray,
        actor_boxes: Optional[list] = None
    ) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """
        Process new frame and update motion classification.
        Args:
            frame: BGR current video frame
            actor_boxes: Optional list of [x1, y1, x2, y2] to mask out moving dynamic actors
        Returns:
            (is_camera_moving, ego_speed_kmh, (foe_u, foe_v))
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if self.prev_gray is None:
            self.prev_gray = gray
            self._detect_features(gray, actor_boxes)
            return self.is_camera_moving, self.ego_speed_kmh, None

        if self.feature_pts is None or len(self.feature_pts) < 25:
            self._detect_features(self.prev_gray, actor_boxes)
            if self.feature_pts is None or len(self.feature_pts) < 10:
                self.prev_gray = gray
                return self.is_camera_moving, self.ego_speed_kmh, None

        # Lucas-Kanade Sparse Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.feature_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        if p1 is not None and st is not None:
            good_p0 = self.feature_pts[st.flatten() == 1]
            good_p1 = p1[st.flatten() == 1]

            if len(good_p1) >= 10:
                flow = (good_p1 - good_p0).reshape(-1, 2)
                mags = np.linalg.norm(flow, axis=1)
                median_mag = float(np.median(mags))
                median_vy = float(np.median(flow[:, 1]))

                self.flow_magnitudes.append(median_mag)
                self.vertical_flows.append(median_vy)
                if len(self.flow_magnitudes) > self.history_len:
                    self.flow_magnitudes.pop(0)
                    self.vertical_flows.pop(0)

                # Smoothed average magnitude
                avg_mag = float(np.mean(self.flow_magnitudes))

                # Classification: Static vs Moving
                # Fixed camera has near-zero background motion (< 0.5 px)
                self.is_camera_moving = avg_mag > self.motion_threshold

                # Estimate forward speed in km/h based on ground flow
                if self.is_camera_moving:
                    # In a forward moving car, ground flow is roughly proportional to speed
                    # Scale factor calibrated for 30 fps 720p dashcam
                    self.ego_speed_mps = max(0.0, avg_mag * 5.5)
                    self.ego_speed_kmh = float(self.ego_speed_mps * 3.6)
                else:
                    self.ego_speed_mps = 0.0
                    self.ego_speed_kmh = 0.0

                # Focus of Expansion (FOE) estimation from diverging flow lines
                if self.is_camera_moving and len(good_p1) >= 20:
                    self._estimate_foe(good_p0, good_p1, w, h)

        # Refresh features for next frame
        self._detect_features(gray, actor_boxes)
        self.prev_gray = gray

        foe = (self.foe_u, self.foe_v) if (self.foe_u and self.foe_v) else None
        return self.is_camera_moving, self.ego_speed_kmh, foe

    def _detect_features(self, gray: np.ndarray, actor_boxes: Optional[list] = None):
        """Extract trackable corner points on static background and road."""
        h, w = gray.shape
        mask = np.zeros_like(gray, dtype=np.uint8)

        # Focus on lower-middle (road / sidewalk / buildings)
        # Exclude upper sky (top 30%) and extreme bottom hood (bottom 10%)
        mask[int(h * 0.35):int(h * 0.90), int(w * 0.10):int(w * 0.90)] = 255

        # Mask out dynamic actors so we don't track other cars or pedestrians
        if actor_boxes:
            for box in actor_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                # Expand box slightly to avoid tracking actor edges
                pad = 8
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                mask[y1:y2, x1:x2] = 0

        self.feature_pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=180,
            qualityLevel=0.015,
            minDistance=14,
            mask=mask
        )

    def _estimate_foe(self, p0: np.ndarray, p1: np.ndarray, w: int, h: int):
        """Estimate Focus of Expansion (vanishing point) from flow vectors."""
        p0 = p0.reshape(-1, 2)
        p1 = p1.reshape(-1, 2)
        flow = p1 - p0
        mags = np.linalg.norm(flow, axis=1)
        valid = mags > 0.8
        if valid.sum() < 8:
            return

        p0_v = p0[valid]
        flow_v = flow[valid]

        # In pure forward motion, lines pass through FOE
        # Default FOE is near center
        center_u = w / 2.0
        center_v = h * 0.52

        # Filter lines that point downward-ish
        downward = flow_v[:, 1] > 0.3
        if downward.sum() > 5:
            # Alpha smooth towards center/detected intersection
            if self.foe_u is None:
                self.foe_u = center_u
                self.foe_v = center_v
            else:
                self.foe_u = 0.95 * self.foe_u + 0.05 * center_u
                self.foe_v = 0.95 * self.foe_v + 0.05 * center_v
