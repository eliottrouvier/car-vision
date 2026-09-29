"""
bev_geometry.py - Camera Calibration, Inverse Perspective Mapping (IPM),
and 3D Metric Projection for Autonomous Vehicle Perception.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np


# Standard automotive dimensions (width, length, height) in meters
CLASS_3D_DIMENSIONS = {
    'car': (1.85, 4.60, 1.45),
    'truck': (2.50, 9.50, 3.20),
    'bus': (2.60, 12.00, 3.40),
    'motorcycle': (0.80, 2.10, 1.30),
    'bicycle': (0.60, 1.75, 1.35),
    'person': (0.65, 0.50, 1.75),
    'default': (1.80, 4.50, 1.50)
}


class BEVGeometry:
    """
    Handles coordinate frames between:
    1. 2D Camera Image plane (u: 0..W, v: 0..H)
    2. 3D Metric Ground frame (X: lateral right [m], Y: vertical up [m], Z: longitudinal forward [m])
       Origin is on the road surface directly under the host vehicle front bumper / camera.
    """

    def __init__(
        self,
        img_w: int = 1280,
        img_h: int = 720,
        cam_height: float = 1.35,      # Camera height above asphalt in meters
        pitch_deg: float = 3.5,         # Downward tilt angle
        yaw_deg: float = 0.0,           # Azimuth angle relative to car centerline
        fov_deg: float = 65.0           # Horizontal field of view in degrees
    ):
        self.w = img_w
        self.h = img_h
        self.cam_h = cam_height
        self.pitch = np.radians(pitch_deg)
        self.yaw = np.radians(yaw_deg)
        self.fov_h = np.radians(fov_deg)

        # Intrinsic parameters
        self.fx = (img_w / 2.0) / np.tan(self.fov_h / 2.0)
        self.fy = self.fx
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0

        # Estimated horizon v coordinate
        self.v_horizon = self.cy + self.fy * np.tan(self.pitch)

    def update_resolution(self, img_w: int, img_h: int):
        """Update intrinsics when resolution changes."""
        if img_w == self.w and img_h == self.h:
            return
        self.w = img_w
        self.h = img_h
        self.fx = (img_w / 2.0) / np.tan(self.fov_h / 2.0)
        self.fy = self.fx
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0
        self.v_horizon = self.cy + self.fy * np.tan(self.pitch)

    def update_vanishing_point(self, vp_u: float, vp_v: float, smoothing: float = 0.95):
        """
        Dynamically adjust camera pitch and yaw from lane vanishing point.
        vp_u, vp_v in pixels.
        """
        # Estimated pitch: tan(pitch) = (vp_v - cy) / fy
        measured_pitch = np.arctan2(vp_v - self.cy, self.fy)
        measured_yaw = np.arctan2(vp_u - self.cx, self.fx)

        # Clamped smoothing
        measured_pitch = np.clip(measured_pitch, np.radians(-10.0), np.radians(15.0))
        measured_yaw = np.clip(measured_yaw, np.radians(-10.0), np.radians(10.0))

        self.pitch = smoothing * self.pitch + (1.0 - smoothing) * measured_pitch
        self.yaw = smoothing * self.yaw + (1.0 - smoothing) * measured_yaw
        self.v_horizon = self.cy + self.fy * np.tan(self.pitch)

    def image_to_ground(self, u: float, v: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Unprojects a 2D image point (u, v) on the road surface (Y=0, at distance cam_h below camera).
        Returns:
            X: lateral position in meters (+ right, - left)
            Z: longitudinal distance in meters (+ ahead)
        """
        denom = (v - self.cy) * np.cos(self.pitch) + self.fy * np.sin(self.pitch)
        if denom <= 1e-4:
            return None, None # Above or exactly at horizon

        Z = (self.fy * self.cam_h) / denom
        if Z <= 0.5 or Z > 150.0:
            return None, None

        X = ((u - self.cx) * Z) / self.fx
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

        # Local offsets: dx, dz
        # fl, fr, rr, rl
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
        Returns list of 8 (u, v) points or None if out of bounds.
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
        dZ_dt is relative longitudinal velocity in m/s (negative if closing in).
        """
        if dZ_dt >= -0.1:
            return None # Not closing in
        closing_speed = -dZ_dt
        ttc = Z / max(closing_speed, 0.05)
        return float(ttc)

    def is_in_ego_lane(self, X: float, lane_half_width: float = 1.85) -> bool:
        """Determines if a lateral position falls within the host vehicle lane."""
        return abs(X) <= lane_half_width

    def unproject_lane_mask(
        self,
        lane_mask: np.ndarray,
        step: int = 4,
        z_max: float = 70.0
    ) -> List[Tuple[float, float]]:
        """
        Extracts lane points from binary mask and projects to 3D ground coordinates (X, Z).
        """
        # Find non-zero pixel coordinates
        v_coords, u_coords = np.where(lane_mask > 0)
        if len(v_coords) == 0:
            return []

        pts_3d = []
        # Sample points to keep computation ultra fast
        indices = np.arange(0, len(v_coords), step)
        for idx in indices:
            u, v = u_coords[idx], v_coords[idx]
            X, Z = self.image_to_ground(float(u), float(v))
            if X is not None and Z is not None and 1.0 <= Z <= z_max:
                pts_3d.append((X, Z))

        return pts_3d
