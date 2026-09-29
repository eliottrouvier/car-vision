"""
bev_widget.py - Tesla FSD / Waymo-grade 3D Bird's-Eye View (BEV) Vector Space.
Features:
- Solid Shaded 3D Automotive Models (Car, Truck, Bus, Two-Wheeler, Pedestrian)
- Luminous LED Headlights, Taillights & Ground Shadows
- Floating High-Contrast HUD Distance & Velocity Badges
- Cross-Screen Interactive Hover Synchronization
- 3 Instant Viewpoint Presets (FSD Driver, Helicopter 45°, Top-Down Zenithal)
- Dynamic Trajectory Corridor (Driving Path Ribbon) with FCW Warning States
- Faithful 3D Drivable Road Carpet & 3D Crosswalks
"""

from typing import Optional, List, Tuple, Dict
import math
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from perception_engine import PerceptionResult
from tracker import TrackedObject


class BEVWidget(QtWidgets.QWidget):
    """
    High-fidelity 3D Vector Space Bird's-Eye View displaying real-time metric perception.
    """

    actor_hovered = QtCore.Signal(object)  # Emits int track_id or None
    actor_selected = QtCore.Signal(object) # Emits int track_id or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #07090e; border-radius: 8px;")

        # Camera & View settings
        self.view_preset = "FSD" # "FSD", "HELICOPTER", "TOP_DOWN"
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start = QtCore.QPoint()
        self.is_dragging = False

        # Road animation state
        self.road_anim_offset = 0.0

        # Interactive state
        self.hovered_track_id: Optional[int] = None
        self.selected_track_id: Optional[int] = None

        # Actor screen hit-boxes for mouse hover
        self._actor_screen_positions: Dict[int, QtCore.QPointF] = {}

        # Perception state
        self.current_result: Optional[PerceptionResult] = None

        self.setMouseTracking(True)

    def set_view_preset(self, preset: str):
        """Switches camera angle between FSD Driver, Helicopter 45°, and Top-Down."""
        if preset in ("FSD", "HELICOPTER", "TOP_DOWN"):
            self.view_preset = preset
            if preset == "FSD":
                self.zoom = 1.05
            elif preset == "HELICOPTER":
                self.zoom = 0.88
            elif preset == "TOP_DOWN":
                self.zoom = 0.75
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.update()

    def set_highlighted_actor(self, track_id: Optional[int]):
        """Slot called when an actor is hovered on the left video HUD."""
        if self.hovered_track_id != track_id:
            self.hovered_track_id = track_id
            self.update()

    def update_result(self, result: PerceptionResult):
        self.current_result = result
        if result.is_camera_moving:
            speed_mps = result.ego_speed_estimate / 3.6
            self.road_anim_offset = (self.road_anim_offset + speed_mps * 0.033) % 8.0
        self.update()

    def reset_view(self):
        self.set_view_preset(self.view_preset)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 0.89
        self.zoom = float(np.clip(self.zoom * factor, 0.35, 4.0))
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self.is_dragging = True
            self.drag_start = event.pos()

            # Check if clicked on an actor
            clicked_id = self._find_actor_at(event.pos())
            if clicked_id is not None:
                self.selected_track_id = clicked_id
                self.actor_selected.emit(clicked_id)
                self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.is_dragging:
            delta = event.pos() - self.drag_start
            self.drag_start = event.pos()
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.update()
        else:
            # Hover detection
            hovered = self._find_actor_at(event.pos())
            if hovered != self.hovered_track_id:
                self.hovered_track_id = hovered
                self.actor_hovered.emit(hovered)
                self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self.is_dragging = False

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        self.reset_view()

    def _find_actor_at(self, pos: QtCore.QPoint) -> Optional[int]:
        """Finds track ID closest to mouse position."""
        closest_id = None
        min_dist = 28.0 # Pixel radius
        for t_id, pt in self._actor_screen_positions.items():
            dist = math.hypot(pt.x() - pos.x(), pt.y() - pos.y())
            if dist < min_dist:
                min_dist = dist
                closest_id = t_id
        return closest_id

    def _project(self, X: float, Z: float, w: int, h: int) -> Tuple[float, float, float]:
        """
        Projects metric ground coordinates (X [m], Z [m]) to screen pixels.
        Returns: (screen_x, screen_y, scale_factor).
        """
        cx = w / 2.0 + self.pan_x

        if self.view_preset == "FSD":
            # 3D oblique driver perspective behind ego car
            cy = h * 0.83 + self.pan_y
            depth_factor = 0.014 / max(self.zoom, 0.2)
            persp = 1.0 / (1.0 + max(Z, 0.0) * depth_factor)
            sx = cx + (X * 22.0 * self.zoom) * persp
            sy = cy - (Z * 9.8 * self.zoom) * persp
            scale = self.zoom * persp
            return sx, sy, scale

        elif self.view_preset == "HELICOPTER":
            # Elevated 45° tactical perspective
            cy = h * 0.86 + self.pan_y
            depth_factor = 0.009 / max(self.zoom, 0.2)
            persp = 1.0 / (1.0 + max(Z, 0.0) * depth_factor)
            sx = cx + (X * 18.0 * self.zoom) * persp
            sy = cy - (Z * 12.5 * self.zoom) * persp
            scale = self.zoom * persp
            return sx, sy, scale

        else:
            # TOP_DOWN: Orthographic 2D map
            cy = h * 0.88 + self.pan_y
            meter_px = 7.5 * self.zoom
            sx = cx + X * meter_px
            sy = cy - Z * meter_px
            return sx, sy, self.zoom

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)

        w = self.width()
        h = self.height()
        self._actor_screen_positions.clear()

        is_moving = self.current_result.is_camera_moving if self.current_result else True
        ego_speed = self.current_result.ego_speed_estimate if self.current_result else 0.0

        # 1. Dark titanium space gradient background
        bg_grad = QtGui.QRadialGradient(w / 2.0, h * 0.65, max(w, h))
        bg_grad.setColorAt(0.0, QtGui.QColor("#0d131c"))
        bg_grad.setColorAt(0.7, QtGui.QColor("#070a10"))
        bg_grad.setColorAt(1.0, QtGui.QColor("#040609"))
        painter.fillRect(0, 0, w, h, QtGui.QBrush(bg_grad))

        # 2. Road surface & Radar Rings
        if is_moving:
            self._draw_dynamic_asphalt_road(painter, w, h)
            self._draw_radar_rings(painter, w, h)
        else:
            self._draw_fixed_street_grid(painter, w, h)

        # 3. Faithful Drivable Road Polygon from YOLOPv2
        if self.current_result and self.current_result.bev_carpet_3d:
            self._draw_drivable_road_carpet(painter, w, h)

        # 4. Projected 3D Lanes
        if self.current_result and self.current_result.bev_lanes_3d:
            self._draw_panoptic_lanes(painter, w, h)

        # 5. Projected Crosswalks
        if self.current_result and self.current_result.crosswalks:
            self._draw_crosswalks(painter, w, h)

        # 6. Trajectory Corridor (Driving Path Ribbon)
        if self.current_result:
            self._draw_trajectory_corridor(painter, w, h)

        # 7. Ego Host Vehicle
        if is_moving:
            self._draw_ego_vehicle(painter, w, h)
        else:
            self._draw_fixed_camera_origin(painter, w, h)

        # 8. Surrounding Tracked 3D Actors (Vehicles, Pedestrians, Cyclists)
        if self.current_result is not None:
            # Sort tracks back-to-front (largest Z first) for correct depth rendering
            sorted_tracks = sorted(self.current_result.tracks, key=lambda t: t.Z, reverse=True)
            for trk in sorted_tracks:
                self._draw_3d_actor(painter, trk, w, h, is_moving)

        # 9. Reference Frame Badge & Minimalist HUD
        self._draw_hud_overlay(painter, w, h, is_moving, ego_speed)

    def _draw_dynamic_asphalt_road(self, painter: QtGui.QPainter, w: int, h: int):
        """Draws living road ribbon with moving dashed lines."""
        # Asphalt ground polygon: X from -6.5m to +6.5m, Z from 0 to 85m
        p_bl_x, p_bl_y, _ = self._project(-6.8, 0.0, w, h)
        p_br_x, p_br_y, _ = self._project(6.8, 0.0, w, h)
        p_tr_x, p_tr_y, _ = self._project(6.8, 85.0, w, h)
        p_tl_x, p_tl_y, _ = self._project(-6.8, 85.0, w, h)

        road_poly = QtGui.QPolygonF([
            QtCore.QPointF(p_bl_x, p_bl_y),
            QtCore.QPointF(p_br_x, p_br_y),
            QtCore.QPointF(p_tr_x, p_tr_y),
            QtCore.QPointF(p_tl_x, p_tl_y)
        ])

        road_grad = QtGui.QLinearGradient(p_bl_x, p_bl_y, p_tl_x, p_tl_y)
        road_grad.setColorAt(0.0, QtGui.QColor("#131a24"))
        road_grad.setColorAt(0.85, QtGui.QColor("#090d13"))
        road_grad.setColorAt(1.0, QtGui.QColor("#05080c"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(road_grad))
        painter.drawPolygon(road_poly)

        # Curb lines (X = -3.75m and +3.75m)
        curb_pen = QtGui.QPen(QtGui.QColor(0, 220, 255, 120), 2.0)
        painter.setPen(curb_pen)
        for curb_x in [-3.75, 3.75]:
            p1_x, p1_y, _ = self._project(curb_x, 0.0, w, h)
            p2_x, p2_y, _ = self._project(curb_x, 85.0, w, h)
            painter.drawLine(QtCore.QPointF(p1_x, p1_y), QtCore.QPointF(p2_x, p2_y))

        # Animated streaming dashed center line (X = 0.0)
        dash_len = 3.6
        dash_gap = 4.4
        cycle = dash_len + dash_gap
        offset = self.road_anim_offset

        dash_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 175), 2.2)
        painter.setPen(dash_pen)

        for z in np.arange(-cycle + offset, 85.0, cycle):
            z_start = max(0.5, z)
            z_end = min(85.0, z + dash_len)
            if z_end > z_start:
                s1_x, s1_y, _ = self._project(0.0, z_start, w, h)
                s2_x, s2_y, _ = self._project(0.0, z_end, w, h)
                painter.drawLine(QtCore.QPointF(s1_x, s1_y), QtCore.QPointF(s2_x, s2_y))

    def _draw_drivable_road_carpet(self, painter: QtGui.QPainter, w: int, h: int):
        """Draws faithful 3D drivable polygon extracted directly from YOLOPv2."""
        pts_3d = self.current_result.bev_carpet_3d
        if len(pts_3d) < 3:
            return

        screen_pts = []
        for X, Z in pts_3d:
            sx, sy, _ = self._project(X, Z, w, h)
            screen_pts.append(QtCore.QPointF(sx, sy))

        poly = QtGui.QPolygonF(screen_pts)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 230, 255, 80), 1.2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 180, 240, 22)))
        painter.drawPolygon(poly)

    def _draw_trajectory_corridor(self, painter: QtGui.QPainter, w: int, h: int):
        """Renders Tesla driving path ribbon ahead of the ego vehicle."""
        corridor = self.current_result.trajectory_corridor
        if len(corridor) < 2:
            return

        # Width of trajectory corridor ~ 2.2m
        hw = 1.1

        left_pts = []
        right_pts = []

        is_critical = self.current_result.fcw_alert
        for x, z in corridor:
            lx, ly, _ = self._project(x - hw, z, w, h)
            rx, ry, _ = self._project(x + hw, z, w, h)
            left_pts.append(QtCore.QPointF(lx, ly))
            right_pts.append(QtCore.QPointF(rx, ry))

        # Build closed ribbon polygon
        ribbon_poly = QtGui.QPolygonF(left_pts + list(reversed(right_pts)))

        if is_critical:
            # Pulsing crimson collision warning
            c_fill = QtGui.QColor(239, 68, 68, 85)
            c_edge = QtGui.QColor(239, 68, 68, 220)
        else:
            # Calm glowing cyan/emerald path
            c_fill = QtGui.QColor(0, 230, 255, 45)
            c_edge = QtGui.QColor(0, 230, 255, 170)

        painter.setPen(QtGui.QPen(c_edge, 1.8))
        painter.setBrush(QtGui.QBrush(c_fill))
        painter.drawPolygon(ribbon_poly)

    def _draw_fixed_street_grid(self, painter: QtGui.QPainter, w: int, h: int):
        """Draws stationary metric world grid when camera is in a fixed surveillance location."""
        p_bl_x, p_bl_y, _ = self._project(-15.0, -1.0, w, h)
        p_br_x, p_br_y, _ = self._project(15.0, -1.0, w, h)
        p_tr_x, p_tr_y, _ = self._project(15.0, 75.0, w, h)
        p_tl_x, p_tl_y, _ = self._project(-15.0, 75.0, w, h)

        ground_poly = QtGui.QPolygonF([
            QtCore.QPointF(p_bl_x, p_bl_y),
            QtCore.QPointF(p_br_x, p_br_y),
            QtCore.QPointF(p_tr_x, p_tr_y),
            QtCore.QPointF(p_tl_x, p_tl_y)
        ])
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#0e131d")))
        painter.drawPolygon(ground_poly)

        # Metric grid
        pen = QtGui.QPen(QtGui.QColor(0, 180, 240, 35), 1.0)
        painter.setPen(pen)

        for gx in np.arange(-15.0, 16.0, 5.0):
            p1_x, p1_y, _ = self._project(gx, 0.0, w, h)
            p2_x, p2_y, _ = self._project(gx, 75.0, w, h)
            painter.drawLine(QtCore.QPointF(p1_x, p1_y), QtCore.QPointF(p2_x, p2_y))

        for gz in np.arange(5.0, 76.0, 10.0):
            p1_x, p1_y, _ = self._project(-15.0, gz, w, h)
            p2_x, p2_y, _ = self._project(15.0, gz, w, h)
            painter.drawLine(QtCore.QPointF(p1_x, p1_y), QtCore.QPointF(p2_x, p2_y))

            lx, ly, _ = self._project(15.2, gz, w, h)
            painter.setFont(QtGui.QFont("SF Pro Display", 8))
            painter.setPen(QtGui.QColor(0, 180, 240, 95))
            painter.drawText(int(lx), int(ly) + 3, f"{int(gz)}m")

    def _draw_radar_rings(self, painter: QtGui.QPainter, w: int, h: int):
        """Concentric metric distance rings."""
        distances = [10.0, 25.0, 50.0, 75.0]
        for dist in distances:
            pts = []
            for i in range(37):
                angle = (i / 36.0) * math.pi
                gx = dist * math.cos(angle)
                gz = dist * math.sin(angle)
                sx, sy, _ = self._project(gx, gz, w, h)
                pts.append(QtCore.QPointF(sx, sy))

            pen = QtGui.QPen(QtGui.QColor(0, 200, 255, 35), 1.0, QtCore.Qt.DashLine)
            painter.setPen(pen)
            for k in range(len(pts) - 1):
                painter.drawLine(pts[k], pts[k + 1])

            lx, ly, _ = self._project(dist * 0.85, dist * 0.52, w, h)
            painter.setFont(QtGui.QFont("SF Pro Display", 8))
            painter.setPen(QtGui.QColor(0, 200, 255, 90))
            painter.drawText(int(lx), int(ly), f"{int(dist)}m")

    def _draw_panoptic_lanes(self, painter: QtGui.QPainter, w: int, h: int):
        """Draw lane splines from perception engine."""
        pen = QtGui.QPen(QtGui.QColor(0, 245, 255, 180), 2.2)
        painter.setPen(pen)
        for X, Z in self.current_result.bev_lanes_3d:
            sx, sy, sc = self._project(X, Z, w, h)
            pt_size = max(1.2, 2.5 * sc)
            painter.drawEllipse(QtCore.QPointF(sx, sy), pt_size, pt_size)

    def _draw_crosswalks(self, painter: QtGui.QPainter, w: int, h: int):
        """Draw 3D zebra crosswalks."""
        for cw in self.current_result.crosswalks:
            cx, cz = cw['X'], cw['Z']
            num_stripes = max(5, cw.get('stripes', 6))
            width = 4.8
            depth = 2.4
            stripe_w = width / (num_stripes * 2.0)

            for i in range(num_stripes):
                sx_offset = -width / 2.0 + i * (stripe_w * 2.0)
                c1_x, c1_y, _ = self._project(cx + sx_offset, cz - depth / 2.0, w, h)
                c2_x, c2_y, _ = self._project(cx + sx_offset + stripe_w, cz - depth / 2.0, w, h)
                c3_x, c3_y, _ = self._project(cx + sx_offset + stripe_w, cz + depth / 2.0, w, h)
                c4_x, c4_y, _ = self._project(cx + sx_offset, cz + depth / 2.0, w, h)

                poly = QtGui.QPolygonF([
                    QtCore.QPointF(c1_x, c1_y),
                    QtCore.QPointF(c2_x, c2_y),
                    QtCore.QPointF(c3_x, c3_y),
                    QtCore.QPointF(c4_x, c4_y)
                ])
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 180)))
                painter.drawPolygon(poly)

    def _draw_ego_vehicle(self, painter: QtGui.QPainter, w: int, h: int):
        """Draws the Host/Ego Vehicle in sleek metallic graphite."""
        car_w = 1.90
        car_l = 4.70
        hw = car_w / 2.0
        hl = car_l / 2.0

        # Chassis ground corners
        c1_x, c1_y, sc = self._project(-hw, hl, w, h)
        c2_x, c2_y, _  = self._project(hw, hl, w, h)
        c3_x, c3_y, _  = self._project(hw, -hl, w, h)
        c4_x, c4_y, _  = self._project(-hw, -hl, w, h)

        # Wheels
        self._draw_wheel(painter, -hw, hl * 0.65, w, h)
        self._draw_wheel(painter,  hw, hl * 0.65, w, h)
        self._draw_wheel(painter,  hw, -hl * 0.65, w, h)
        self._draw_wheel(painter, -hw, -hl * 0.65, w, h)

        # Ground contact shadow
        shadow_poly = QtGui.QPolygonF([
            QtCore.QPointF(c1_x, c1_y),
            QtCore.QPointF(c2_x, c2_y),
            QtCore.QPointF(c3_x, c3_y),
            QtCore.QPointF(c4_x, c4_y)
        ])
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 160)))
        painter.drawPolygon(shadow_poly)

        # 3D Raised Roof
        roof_shift = 16.0 * sc
        r1 = QtCore.QPointF(c1_x, c1_y - roof_shift)
        r2 = QtCore.QPointF(c2_x, c2_y - roof_shift)
        r3 = QtCore.QPointF(c3_x, c3_y - roof_shift)
        r4 = QtCore.QPointF(c4_x, c4_y - roof_shift)

        # Body chassis
        body_grad = QtGui.QLinearGradient(c4_x, c4_y, c1_x, c1_y)
        body_grad.setColorAt(0.0, QtGui.QColor("#2d3748"))
        body_grad.setColorAt(1.0, QtGui.QColor("#1a202c"))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 230, 255), 1.8))
        painter.setBrush(QtGui.QBrush(body_grad))
        painter.drawPolygon(QtGui.QPolygonF([r1, r2, r3, r4]))

        # Glass sunroof
        sunroof_pts = [
            QtCore.QPointF(r1.x() * 0.75 + r4.x() * 0.25, r1.y() * 0.75 + r4.y() * 0.25),
            QtCore.QPointF(r2.x() * 0.75 + r3.x() * 0.25, r2.y() * 0.75 + r3.y() * 0.25),
            QtCore.QPointF(r2.x() * 0.25 + r3.x() * 0.75, r2.y() * 0.25 + r3.y() * 0.75),
            QtCore.QPointF(r1.x() * 0.25 + r4.x() * 0.75, r1.y() * 0.25 + r4.y() * 0.75)
        ]
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 195, 255, 90)))
        painter.drawPolygon(QtGui.QPolygonF(sunroof_pts))

        # Headlights (luminous white/cyan)
        painter.setPen(QtGui.QPen(QtGui.QColor(240, 250, 255), 3.0))
        painter.drawPoint(r1)
        painter.drawPoint(r2)

    def _draw_wheel(self, painter: QtGui.QPainter, X: float, Z: float, w: int, h: int):
        """Draw tire rectangle on road plane."""
        p1_x, p1_y, _ = self._project(X - 0.12, Z - 0.35, w, h)
        p2_x, p2_y, _ = self._project(X + 0.12, Z - 0.35, w, h)
        p3_x, p3_y, _ = self._project(X + 0.12, Z + 0.35, w, h)
        p4_x, p4_y, _ = self._project(X - 0.12, Z + 0.35, w, h)

        poly = QtGui.QPolygonF([
            QtCore.QPointF(p1_x, p1_y),
            QtCore.QPointF(p2_x, p2_y),
            QtCore.QPointF(p3_x, p3_y),
            QtCore.QPointF(p4_x, p4_y)
        ])
        painter.setPen(QtGui.QPen(QtGui.QColor(70, 85, 105), 1.0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(16, 20, 26)))
        painter.drawPolygon(poly)

    def _draw_fixed_camera_origin(self, painter: QtGui.QPainter, w: int, h: int):
        """In fixed camera mode: render camera origin and observation frustum."""
        cam_x, cam_y, _ = self._project(0.0, 0.0, w, h)
        painter.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248), 2.0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(14, 165, 233, 90)))
        painter.drawEllipse(QtCore.QPointF(cam_x, cam_y), 6, 6)

        f_left_x, f_left_y, _ = self._project(-12.0, 50.0, w, h)
        f_right_x, f_right_y, _ = self._project(12.0, 50.0, w, h)
        frust_pen = QtGui.QPen(QtGui.QColor(56, 189, 248, 60), 1.2, QtCore.Qt.DashLine)
        painter.setPen(frust_pen)
        painter.drawLine(QtCore.QPointF(cam_x, cam_y), QtCore.QPointF(f_left_x, f_left_y))
        painter.drawLine(QtCore.QPointF(cam_x, cam_y), QtCore.QPointF(f_right_x, f_right_y))

    def _draw_3d_actor(self, painter: QtGui.QPainter, trk: TrackedObject, w: int, h: int, is_moving: bool):
        """
        Renders solid shaded 3D models with high visibility:
        - Vehicles: Shaded 3D body chassis, LED headlights, taillights, and floating HUD badge.
        - Pedestrians: Upright 3D human figure with proximity halo.
        - Two-Wheelers: Bike frame, wheels, and rider.
        """
        is_hovered = (trk.track_id == self.hovered_track_id)
        is_lead = (self.current_result and self.current_result.lead_vehicle and self.current_result.lead_vehicle.track_id == trk.track_id)

        if trk.class_name == 'person':
            self._draw_3d_pedestrian(painter, trk, w, h, is_hovered, is_lead)
        elif trk.class_name in ('bicycle', 'motorcycle'):
            self._draw_3d_twowheeler(painter, trk, w, h, is_hovered, is_lead)
        else:
            self._draw_3d_vehicle(painter, trk, w, h, is_hovered, is_lead)

    def _draw_3d_vehicle(
        self,
        painter: QtGui.QPainter,
        trk: TrackedObject,
        w: int,
        h: int,
        is_hovered: bool,
        is_lead: bool
    ):
        """Renders solid shaded 3D car/truck body with high-contrast lighting."""
        X = trk.X
        Z = trk.Z
        yaw = trk.yaw

        is_truck = trk.class_name in ('truck', 'bus')
        car_w = 2.45 if is_truck else 1.85
        car_l = 8.80 if is_truck else 4.60
        car_h = 3.00 if is_truck else 1.48

        hw = car_w / 2.0
        hl = car_l / 2.0

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # 4 Ground corners (fl, fr, rr, rl)
        corners_local = [
            (-hw,  hl), # Front-left
            ( hw,  hl), # Front-right
            ( hw, -hl), # Rear-right
            (-hw, -hl)  # Rear-left
        ]

        ground_pts = []
        for dx, dz in corners_local:
            rot_x = X + dx * cos_y + dz * sin_y
            rot_z = Z - dx * sin_y + dz * cos_y
            sx, sy, sc = self._project(rot_x, rot_z, w, h)
            ground_pts.append(QtCore.QPointF(sx, sy))

        center_sx, center_sy, sc_center = self._project(X, Z, w, h)
        self._actor_screen_positions[trk.track_id] = QtCore.QPointF(center_sx, center_sy)

        # Ground contact shadow
        shadow_poly = QtGui.QPolygonF(ground_pts)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 180)))
        painter.drawPolygon(shadow_poly)

        # Accent styling
        if trk.warning_level == 'critical':
            accent_edge = QtGui.QColor(239, 68, 68)      # Crimson Red
            body_fill = QtGui.QColor(239, 68, 68, 80)
        elif is_lead:
            accent_edge = QtGui.QColor(245, 158, 11)     # Amber Lead
            body_fill = QtGui.QColor(245, 158, 11, 70)
        elif is_hovered:
            accent_edge = QtGui.QColor(0, 245, 255)      # Electric Cyan Hover
            body_fill = QtGui.QColor(0, 230, 255, 90)
        else:
            accent_edge = QtGui.QColor(56, 189, 248)     # Tech Blue
            body_fill = QtGui.QColor(30, 41, 59, 220)

        # Raised 3D Roof Points
        roof_shift_px = car_h * 11.5 * sc_center
        roof_pts = [QtCore.QPointF(p.x(), p.y() - roof_shift_px) for p in ground_pts]

        # Draw Side Pillars & Side Panels
        pillar_pen = QtGui.QPen(accent_edge, 1.6 if (is_hovered or is_lead) else 1.2)
        painter.setPen(pillar_pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(24, 32, 45, 200)))

        # Left side panel
        left_poly = QtGui.QPolygonF([ground_pts[3], ground_pts[0], roof_pts[0], roof_pts[3]])
        painter.drawPolygon(left_poly)

        # Right side panel
        right_poly = QtGui.QPolygonF([ground_pts[1], ground_pts[2], roof_pts[2], roof_pts[1]])
        painter.drawPolygon(right_poly)

        # Front panel (facing headlights)
        front_poly = QtGui.QPolygonF([ground_pts[0], ground_pts[1], roof_pts[1], roof_pts[0]])
        painter.setBrush(QtGui.QBrush(QtGui.QColor(36, 48, 66, 230)))
        painter.drawPolygon(front_poly)

        # Rear panel (facing taillights)
        rear_poly = QtGui.QPolygonF([ground_pts[2], ground_pts[3], roof_pts[3], roof_pts[2]])
        painter.drawPolygon(rear_poly)

        # 3D Roof Top Face
        roof_poly = QtGui.QPolygonF(roof_pts)
        painter.setPen(QtGui.QPen(accent_edge, 2.0 if (is_hovered or is_lead) else 1.4))
        painter.setBrush(QtGui.QBrush(body_fill))
        painter.drawPolygon(roof_poly)

        # Luminous LED Headlights on Front Face
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 3.2))
        painter.drawPoint(roof_pts[0])
        painter.drawPoint(roof_pts[1])

        # Luminous Neon Red Taillights on Rear Face
        tail_color = QtGui.QColor(255, 30, 50) if trk.vz < -0.5 else QtGui.QColor(200, 30, 40)
        painter.setPen(QtGui.QPen(tail_color, 3.2))
        painter.drawPoint(roof_pts[2])
        painter.drawPoint(roof_pts[3])

        # Hover aura ring
        if is_hovered or is_lead:
            ring_pen = QtGui.QPen(accent_edge, 2.0, QtCore.Qt.DashLine)
            painter.setPen(ring_pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(center_sx, center_sy), 24.0 * sc_center, 14.0 * sc_center)

        # Floating HUD Badge above vehicle roof
        badge_y = min(p.y() for p in roof_pts) - 10.0
        self._draw_actor_badge(painter, center_sx, badge_y, trk, accent_edge, is_lead)

    def _draw_3d_pedestrian(
        self,
        painter: QtGui.QPainter,
        trk: TrackedObject,
        w: int,
        h: int,
        is_hovered: bool,
        is_lead: bool
    ):
        """Render upright 3D human figure with proximity ground safety halo."""
        X, Z = trk.X, trk.Z
        gx, gy, sc = self._project(X, Z, w, h)
        self._actor_screen_positions[trk.track_id] = QtCore.QPointF(gx, gy)

        # Proximity safety halo color
        if trk.warning_level == 'critical' or Z < 6.0:
            halo_color = QtGui.QColor(239, 68, 68)  # Crimson
        elif trk.warning_level == 'caution' or Z < 12.0:
            halo_color = QtGui.QColor(245, 158, 11) # Amber
        else:
            halo_color = QtGui.QColor(16, 185, 129) # Safe Green

        # Ground safety halo
        halo_rx = max(5.0, 9.0 * sc)
        halo_ry = max(3.0, 5.0 * sc)
        painter.setPen(QtGui.QPen(halo_color, 1.5))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(halo_color.red(), halo_color.green(), halo_color.blue(), 45)))
        painter.drawEllipse(QtCore.QPointF(gx, gy), halo_rx, halo_ry)

        # Upright Torso (person height ~ 1.75m)
        person_h_px = 19.0 * sc
        torso_top = gy - person_h_px * 0.75
        head_top = gy - person_h_px

        body_pen = QtGui.QPen(halo_color, max(2.5, 4.0 * sc))
        painter.setPen(body_pen)
        painter.drawLine(QtCore.QPointF(gx, gy), QtCore.QPointF(gx, torso_top))

        # Head sphere
        head_r = max(3.0, 4.0 * sc)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(halo_color))
        painter.drawEllipse(QtCore.QPointF(gx, head_top), head_r, head_r)

        # Badge
        self._draw_actor_badge(painter, gx, head_top - 8, trk, halo_color, is_lead)

    def _draw_3d_twowheeler(
        self,
        painter: QtGui.QPainter,
        trk: TrackedObject,
        w: int,
        h: int,
        is_hovered: bool,
        is_lead: bool
    ):
        """Render two-wheeler bike and rider."""
        X, Z = trk.X, trk.Z
        gx, gy, sc = self._project(X, Z, w, h)
        self._actor_screen_positions[trk.track_id] = QtCore.QPointF(gx, gy)

        color = QtGui.QColor(255, 190, 20)
        cos_y = math.cos(trk.yaw)
        sin_y = math.sin(trk.yaw)

        f_x = X + 0.9 * sin_y
        f_z = Z + 0.9 * cos_y
        r_x = X - 0.9 * sin_y
        r_z = Z - 0.9 * cos_y

        f_sx, f_sy, _ = self._project(f_x, f_z, w, h)
        r_sx, r_sy, _ = self._project(r_x, r_z, w, h)

        painter.setPen(QtGui.QPen(color, 2.4))
        painter.drawLine(QtCore.QPointF(r_sx, r_sy), QtCore.QPointF(f_sx, f_sy))

        rider_h = 16.0 * sc
        painter.drawLine(QtCore.QPointF(gx, gy), QtCore.QPointF(gx, gy - rider_h))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(color))
        painter.drawEllipse(QtCore.QPointF(gx, gy - rider_h - 2), 3, 3)

        self._draw_actor_badge(painter, gx, gy - rider_h - 10, trk, color, is_lead)

    def _draw_actor_badge(
        self,
        painter: QtGui.QPainter,
        cx: float,
        cy: float,
        trk: TrackedObject,
        accent: QtGui.QColor,
        is_lead: bool
    ):
        """Renders sleek dark glass floating HUD badge above vehicle."""
        if is_lead:
            text = f"★ LEAD #{trk.track_id} • {trk.Z:.1f}m • {trk.speed_kmh:.0f}km/h"
        else:
            text = f"#{trk.track_id} • {trk.Z:.1f}m • {trk.speed_kmh:.0f}km/h"

        font = QtGui.QFont("SF Pro Display", 9, QtGui.QFont.DemiBold)
        painter.setFont(font)
        fm = QtGui.QFontMetrics(font)
        tw = fm.horizontalAdvance(text)
        th = fm.height()

        padding_x = 6
        padding_y = 2
        bw = tw + padding_x * 2
        bh = th + padding_y * 2
        bx = cx - bw / 2.0
        by = cy - bh

        # Dark glass capsule
        painter.setPen(QtGui.QPen(accent, 1.2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(10, 15, 24, 215)))
        painter.drawRoundedRect(QtCore.QRectF(bx, by, bw, bh), 4.0, 4.0)

        # Text
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff")))
        painter.drawText(int(bx + padding_x), int(by + th - 2), text)

    def _draw_hud_overlay(self, painter: QtGui.QPainter, w: int, h: int, is_moving: bool, ego_speed: float):
        """Displays subtle bottom status pill with active camera mode and speed."""
        mode_text = f"VUE {self.view_preset}  •  {'CAMÉRA EMBARQUÉE' if is_moving else 'CAMÉRA FIXE'}  •  {ego_speed:.0f} km/h"
        font = QtGui.QFont("SF Pro Display", 9, QtGui.QFont.Bold)
        painter.setFont(font)
        fm = QtGui.QFontMetrics(font)
        tw = fm.horizontalAdvance(mode_text)

        bx = (w - tw) / 2.0 - 12
        by = h - 28
        bw = tw + 24
        bh = 20

        painter.setPen(QtGui.QPen(QtGui.QColor("#243044"), 1.0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(8, 12, 18, 200)))
        painter.drawRoundedRect(QtCore.QRectF(bx, by, bw, bh), 5.0, 5.0)

        painter.setPen(QtGui.QColor("#38bdf8"))
        painter.drawText(int(bx + 12), int(by + 14), mode_text)
