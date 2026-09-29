"""
bev_geometry.py - Camera Calibration, Inverse Perspective Mapping (IPM),
Robust Hybrid Depth Estimation, and 3D Metric Projection for Autonomous Perception.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np


# Standard automotive and road actor dimensions (width, length, height) in meters
CLASS_3D_DIMENSIONS = {
    'car': (1.85, 4.60, 1.48),
    'truck': (2.50, 9.50, 3.20),
    'bus': (2.60, 12.00, 3.40),
    'motorcycle': (0.80, 2.10, 1.30),
    'bicycle': (0.60, 1.75, 1.35),
    'person': (0.65, 0.50, 1.75),
    'default': (1.80, 4.50, 1.50)
}


class BEVGeometry:
    """
    Handles coordinate transformations between:
    1. 2D Camera Image plane (u: 0..W, v: 0..H)
    2. 3D Metric Ground frame (X: lateral right [m], Y: vertical up [m], Z: longitudinal forward [m])
       Origin is on the road surface directly under the host vehicle front bumper / camera.
    """

    def __init__(
        self,
        img_w: int = 1280,
        img_h: int = 720,
        cam_height: float = 1.35,      # Camera height above asphalt in meters
        pitch_deg: float = 4.2,         # Downward tilt angle relative to horizon
        yaw_deg: float = 0.0,           # Azimuth angle relative to car centerline
        fov_deg: float = 65.0           # Horizontal field of view in degrees
    ):
        self.w = img_w
        self.h = img_h
        self.cam_h = max(0.5, cam_height)
        self.pitch = np.radians(pitch_deg)
        self.yaw = np.radians(yaw_deg)
        self.fov_h = np.radians(fov_deg)

        # Intrinsic parameters
        self._recompute_intrinsics()

    def _recompute_intrinsics(self):
        """Calculates focal lengths, principal points, and theoretical horizon."""
        self.fx = (self.w / 2.0) / np.tan(self.fov_h / 2.0)
        self.fy = self.fx
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0

        # In camera frame where +Y is down, tilting the camera DOWN (pitch > 0)
        # shifts the horizon ABOVE the optical center: v_horizon = cy - fy * tan(pitch)
        self.v_horizon = self.cy - self.fy * np.tan(self.pitch)

    def update_resolution(self, img_w: int, img_h: int):
        """Update intrinsics dynamically when video resolution changes."""
        if img_w <= 0 or img_h <= 0 or (img_w == self.w and img_h == self.h):
            return
        self.w = img_w
        self.h = img_h
        self._recompute_intrinsics()

    def set_camera_parameters(self, cam_height: Optional[float] = None, pitch_deg: Optional[float] = None):
        """Manually fine-tune camera height and tilt angle."""
        if cam_height is not None:
            self.cam_h = max(0.4, float(cam_height))
        if pitch_deg is not None:
            self.pitch = np.radians(float(pitch_deg))
        self._recompute_intrinsics()

    def update_vanishing_point(self, vp_u: float, vp_v: float, smoothing: float = 0.96):
        """
        Dynamically adjust camera pitch and yaw from lane convergence / optical flow FOE.
        vp_u, vp_v in image pixels.
        """
        if vp_u <= 0 or vp_v <= 0 or vp_v >= self.h:
            return

        # Measured pitch: tan(pitch) = (cy - vp_v) / fy
        measured_pitch = np.arctan2(self.cy - vp_v, self.fy)
        measured_yaw = np.arctan2(vp_u - self.cx, self.fx)

        # Clamped smoothing to maintain rock-solid stability
        measured_pitch = float(np.clip(measured_pitch, np.radians(0.5), np.radians(14.0)))
        measured_yaw = float(np.clip(measured_yaw, np.radians(-8.0), np.radians(8.0)))

        self.pitch = smoothing * self.pitch + (1.0 - smoothing) * measured_pitch
        self.yaw = smoothing * self.yaw + (1.0 - smoothing) * measured_yaw
        self.v_horizon = self.cy - self.fy * np.tan(self.pitch)

    def image_to_ground(self, u: float, v: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Unprojects a 2D image point (u, v) on the road surface (Y=0).
        Returns:
            X: lateral position in meters (+ right, - left)
            Z: longitudinal distance in meters (+ ahead)
        """
        # Normalized vertical coordinate
        m_v = (v - self.cy) / self.fy
        m_u = (u - self.cx) / self.fx

        # Denominator in forward projection
        denom = np.sin(self.pitch) + m_v * np.cos(self.pitch)
        if denom <= 0.005:
            return None, None # Above or right at horizon

        Z = self.cam_h * (np.cos(self.pitch) - m_v * np.sin(self.pitch)) / denom
        if Z < 0.5 or Z > 160.0:
            return None, None

        # Camera coordinate along optical axis
        Z_cam = self.cam_h * np.sin(self.pitch) + Z * np.cos(self.pitch)
        X = m_u * Z_cam

        return float(X), float(Z)

    def world_to_image(self, X: float, Y_above_ground: float, Z: float) -> Tuple[Optional[int], Optional[int]]:
        """
        Projects 3D point (X, Y_above_ground, Z) to 2D image coordinates (u, v).
        """
        if Z <= 0.1:
            return None, None

        # Camera frame position (Y_cam = cam_h - Y_above_ground)
        Y_cam = self.cam_h - Y_above_ground

        # Rotation around X axis by pitch
        Y_rot = Y_cam * np.cos(self.pitch) - Z * np.sin(self.pitch)
        Z_rot = Y_cam * np.sin(self.pitch) + Z * np.cos(self.pitch)

        if Z_rot <= 0.1:
            return None, None

        u = self.cx + (self.fx * X) / Z_rot
        v = self.cy + (self.fy * Y_rot) / Z_rot

        return int(round(u)), int(round(v))

    def estimate_actor_3d(
        self,
        box_2d: List[float],
        class_name: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Robust Bayesian Depth Estimator:
        Fuses Inverse Perspective Mapping (IPM) at ground contact point with
        Pinhole Apparent Height Prior (Z = fy * H / h_box).
        Ensures distant cars near the horizon never get lost or produce invalid distances.

        Returns:
            X: lateral position [m]
            Z: longitudinal distance [m]
            box_height_prior: estimated height prior distance [m]
        """
        x1, y1, x2, y2 = box_2d
        h_box_px = max(y2 - y1, 2.0)
        u_bot = (x1 + x2) / 2.0
        v_bot = y2

        dims = CLASS_3D_DIMENSIONS.get(class_name.lower(), CLASS_3D_DIMENSIONS['default'])
        h_real = dims[2]

        # 1. Pinhole apparent height prior
        Z_prior = (self.fy * h_real) / h_box_px
        Z_prior = float(np.clip(Z_prior, 1.0, 150.0))

        # 2. Ground contact IPM unprojection
        X_ipm, Z_ipm = self.image_to_ground(u_bot, v_bot)

        # 3. Dynamic Bayesian weighting based on vertical distance to horizon
        v_margin = v_bot - self.v_horizon
        if Z_ipm is None or v_margin <= 5.0:
            # Too close to or above horizon: trust height prior
            Z_fused = Z_prior
        else:
            # Sigmoid weight: 1.0 when clearly in road foreground, 0.0 near horizon
            w_ipm = 1.0 / (1.0 + np.exp(-(v_margin - 35.0) / 12.0))
            w_ipm = float(np.clip(w_ipm, 0.05, 0.95))
            Z_fused = w_ipm * Z_ipm + (1.0 - w_ipm) * Z_prior

        Z_fused = float(np.clip(Z_fused, 1.0, 150.0))

        # Compute lateral X using optical geometry
        m_u = (u_bot - self.cx) / self.fx
        Z_cam = self.cam_h * np.sin(self.pitch) + Z_fused * np.cos(self.pitch)
        X_fused = float(m_u * Z_cam)

        return X_fused, Z_fused, Z_prior

    def get_3d_box_corners(
        self,
        X_center: float,
        Z_center: float,
        class_name: str,
        yaw: float = 0.0
    ) -> np.ndarray:
        """
        Computes 8 metric 3D corners for a bounding cuboid centered at (X_center, Z_center).
        Returns array of shape (8, 3): [X, Y_above_ground, Z]
        Corners order:
          0..3: bottom corners (fl, fr, rr, rl)
          4..7: top corners (fl, fr, rr, rl)
        """
        dims = CLASS_3D_DIMENSIONS.get(class_name.lower(), CLASS_3D_DIMENSIONS['default'])
        w, l, h = dims

        # Local offsets: dx, dz (fl, fr, rr, rl)
        dx = np.array([-w / 2.0,  w / 2.0,  w / 2.0, -w / 2.0])
        dz = np.array([ l / 2.0,  l / 2.0, -l / 2.0, -l / 2.0])

        # Apply yaw rotation
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        dx_rot = dx * cos_y - dz * sin_y
        dz_rot = dx * sin_y + dz * cos_y

        X = X_center + dx_rot
        Z = Z_center + dz_rot

        # Bottom 4 corners (Y=0)
        bottom = np.column_stack((X, np.zeros(4), Z))
        # Top 4 corners (Y=h)
        top = np.column_stack((X, np.full(4, h), Z))

        return np.vstack((bottom, top))

    def project_3d_box(self, corners_3d: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Projects 8 3D corners to 2D image coordinates.
        Returns list of 8 (u, v) points or None if behind camera.
        """
        pts_2d = []
        for i in range(8):
            u, v = self.world_to_image(corners_3d[i, 0], corners_3d[i, 1], corners_3d[i, 2])
            if u is None or v is None:
                return None
            pts_2d.append((u, v))
        return pts_2d

    def calculate_ttc(self, Z: float, dZ_dt: float) -> Optional[float]:
        """
        Calculates Time-To-Collision in seconds.
        dZ_dt is relative longitudinal velocity in m/s (negative when closing in).
        """
        if dZ_dt >= -0.2:
            return None # Not closing in
        closing_speed = -dZ_dt
        ttc = Z / max(closing_speed, 0.1)
        return float(ttc)

    def is_in_ego_lane(self, X: float, lane_half_width: float = 1.85) -> bool:
        """Determines if a lateral position falls within the host vehicle lane."""
        return abs(X) <= lane_half_width

    def unproject_lane_mask(
        self,
        lane_mask: np.ndarray,
        step: int = 5,
        z_max: float = 85.0
    ) -> List[Tuple[float, float]]:
        """
        Extracts lane points from binary mask and projects to 3D ground coordinates (X, Z).
        """
        v_coords, u_coords = np.where(lane_mask > 0)
        if len(v_coords) == 0:
            return []

        pts_3d = []
        indices = np.arange(0, len(v_coords), step)
        for idx in indices:
            u, v = u_coords[idx], v_coords[idx]
            X, Z = self.image_to_ground(float(u), float(v))
            if X is not None and Z is not None and 1.0 <= Z <= z_max:
                pts_3d.append((X, Z))

        return pts_3d

    def unproject_road_polygon(
        self,
        drivable_mask: np.ndarray,
        z_max: float = 80.0
    ) -> List[Tuple[float, float]]:
        """
        Extracts the outer boundary contour of the drivable road mask and
        unprojects each boundary vertex to true 3D metric ground coordinates (X, Z).
        Returns an ordered 3D ground polygon matching actual curved road boundaries.
        """
        if drivable_mask is None or np.count_nonzero(drivable_mask) < 200:
            return []

        import cv2
        contours, _ = cv2.findContours(drivable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)
        # Approximate contour to reduce vertex count while preserving shape
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        poly_3d = []
        for pt in approx:
            u, v = float(pt[0][0]), float(pt[0][1])
            X, Z = self.image_to_ground(u, v)
            if X is not None and Z is not None and 0.5 <= Z <= z_max:
                poly_3d.append((X, Z))

        return poly_3d
