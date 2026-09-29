"""
bev_widget.py - Tesla FSD / Waymo-grade 3D Bird's-Eye View (BEV) Vector Space.
Features Dynamic Camera Motion Classification (Mobile Dashcam vs Fixed Surveillance),
Living Scrolling Asphalt Surface, Volumetric 3D Vehicle Models, Upright 3D Pedestrians,
and Metric Kinematics.
"""

from typing import Optional, List, Tuple
import math
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from perception_engine import PerceptionResult
from tracker import TrackedObject


class BEVWidget(QtWidgets.QWidget):
    """
    High-fidelity 3D Vector Space Bird's-Eye View displaying real-time metric perception,
    adapting automatically to Mobile Dashcam or Fixed Surveillance camera modes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #07090e; border-radius: 8px;")

        # Camera & View settings
        self.view_mode = "3D_FSD" # "3D_FSD" or "TOP_DOWN"
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start = QtCore.QPoint()
        self.is_dragging = False

        # Road animation state
        self.road_anim_offset = 0.0

        # State
        self.current_result: Optional[PerceptionResult] = None

        self.setMouseTracking(True)

    def set_view_mode(self, mode: str):
        if mode in ("3D_FSD", "TOP_DOWN"):
            self.view_mode = mode
            self.update()

    def update_result(self, result: PerceptionResult):
        self.current_result = result
        # Update animated road scrolling based on ego speed
        if result.is_camera_moving:
            speed_mps = result.ego_speed_estimate / 3.6
            self.road_anim_offset = (self.road_anim_offset + speed_mps * 0.033) % 8.0
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom = min(3.5, self.zoom * 1.12)
        else:
            self.zoom = max(0.4, self.zoom / 1.12)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self.is_dragging = True
            self.drag_start = event.pos()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.is_dragging:
            delta = event.pos() - self.drag_start
            self.drag_start = event.pos()
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self.is_dragging = False

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        self.reset_view()

    def _project(self, X: float, Z: float, w: int, h: int) -> Tuple[float, float, float]:
        """
        Projects metric ground coordinates (X [m], Z [m]) to screen pixels.
        Returns (screen_x, screen_y, scale_factor).
        """
        cx = w / 2.0 + self.pan_x
        if self.view_mode == "3D_FSD":
            cy = h * 0.84 + self.pan_y
            depth_factor = 0.015 / max(self.zoom, 0.2)
            persp = 1.0 / (1.0 + max(Z, 0.0) * depth_factor)
            sx = cx + (X * 24.0 * self.zoom) * persp
            sy = cy - (Z * 10.5 * self.zoom) * persp
            scale = self.zoom * persp
            return sx, sy, scale
        else:
            # Orthographic top-down map
            cy = h * 0.86 + self.pan_y
            meter_px = 8.0 * self.zoom
            sx = cx + X * meter_px
            sy = cy - Z * meter_px
            return sx, sy, self.zoom

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)

        w = self.width()
        h = self.height()

        is_moving = self.current_result.is_camera_moving if self.current_result else True
        ego_speed = self.current_result.ego_speed_estimate if self.current_result else 0.0

        # 1. Dark titanium space background
        bg_grad = QtGui.QRadialGradient(w / 2.0, h * 0.6, max(w, h))
        bg_grad.setColorAt(0.0, QtGui.QColor("#0e131b"))
        bg_grad.setColorAt(1.0, QtGui.QColor("#05070a"))
        painter.fillRect(0, 0, w, h, QtGui.QBrush(bg_grad))

        # 2. Road surface & Ground Grid
        if is_moving:
            # MODE A: Caméra Embarquée (Moving Car)
            self._draw_dynamic_asphalt_road(painter, w, h)
            self._draw_radar_rings(painter, w, h)
        else:
            # MODE B: Caméra Fixe (Surveillance / Static)
            self._draw_fixed_street_grid(painter, w, h)

        # 3. Projected 3D Lanes from Panoptic segmentation
        if self.current_result and self.current_result.bev_lanes_3d:
            self._draw_panoptic_lanes(painter, w, h)

        # 4. Projected Crosswalks
        if self.current_result and self.current_result.crosswalks:
            self._draw_crosswalks(painter, w, h)

        # 5. Ego Vehicle (Rendered ONLY in Mobile Dashcam mode)
        if is_moving:
            self._draw_ego_vehicle(painter, w, h)
        else:
            self._draw_fixed_camera_origin(painter, w, h)

        # 6. Surrounding Tracked 3D Actors (Vehicles, Pedestrians)
        if self.current_result is not None:
            for trk in self.current_result.tracks:
                self._draw_3d_actor(painter, trk, w, h, is_moving)

        # 7. Reference Frame Badge & Minimalist HUD
        self._draw_reference_frame_badge(painter, w, h, is_moving, ego_speed)

    def _draw_dynamic_asphalt_road(self, painter: QtGui.QPainter, w: int, h: int):
        """Draws living road ribbon with moving dashed lines streaming towards Ego car."""
        # Asphalt ground polygon: X from -6.0m to +6.0m, Z from 0 to 80m
        p_bl_x, p_bl_y, _ = self._project(-6.5, 0.0, w, h)
        p_br_x, p_br_y, _ = self._project(6.5, 0.0, w, h)
        p_tr_x, p_tr_y, _ = self._project(6.5, 80.0, w, h)
        p_tl_x, p_tl_y, _ = self._project(-6.5, 80.0, w, h)

        road_poly = QtGui.QPolygonF([
            QtCore.QPointF(p_bl_x, p_bl_y),
            QtCore.QPointF(p_br_x, p_br_y),
            QtCore.QPointF(p_tr_x, p_tr_y),
            QtCore.QPointF(p_tl_x, p_tl_y)
        ])

        road_grad = QtGui.QLinearGradient(p_bl_x, p_bl_y, p_tl_x, p_tl_y)
        road_grad.setColorAt(0.0, QtGui.QColor("#141a24"))
        road_grad.setColorAt(1.0, QtGui.QColor("#0a0e14"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(road_grad))
        painter.drawPolygon(road_poly)

        # Left and Right solid curb lines (X = -3.7m and +3.7m)
        curb_pen = QtGui.QPen(QtGui.QColor(0, 220, 255, 140), 2.0)
        painter.setPen(curb_pen)
        for curb_x in [-3.7, 3.7]:
            p1_x, p1_y, _ = self._project(curb_x, 0.0, w, h)
            p2_x, p2_y, _ = self._project(curb_x, 80.0, w, h)
            painter.drawLine(QtCore.QPointF(p1_x, p1_y), QtCore.QPointF(p2_x, p2_y))

        # Animated streaming dashed center line (X = 0.0)
        dash_len = 3.5
        dash_gap = 4.5
        cycle = dash_len + dash_gap
        offset = self.road_anim_offset

        dash_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 2.2)
        painter.setPen(dash_pen)

        for z in np.arange(-cycle + offset, 80.0, cycle):
            z_start = max(0.5, z)
            z_end = min(80.0, z + dash_len)
            if z_end > z_start:
                s1_x, s1_y, _ = self._project(0.0, z_start, w, h)
                s2_x, s2_y, _ = self._project(0.0, z_end, w, h)
                painter.drawLine(QtCore.QPointF(s1_x, s1_y), QtCore.QPointF(s2_x, s2_y))

    def _draw_fixed_street_grid(self, painter: QtGui.QPainter, w: int, h: int):
        """Draws stationary metric world grid when camera is in a fixed surveillance location."""
        # Street asphalt plane
        p_bl_x, p_bl_y, _ = self._project(-15.0, -1.0, w, h)
        p_br_x, p_br_y, _ = self._project(15.0, -1.0, w, h)
        p_tr_x, p_tr_y, _ = self._project(15.0, 70.0, w, h)
        p_tl_x, p_tl_y, _ = self._project(-15.0, 70.0, w, h)

        ground_poly = QtGui.QPolygonF([
            QtCore.QPointF(p_bl_x, p_bl_y),
            QtCore.QPointF(p_br_x, p_br_y),
            QtCore.QPointF(p_tr_x, p_tr_y),
            QtCore.QPointF(p_tl_x, p_tl_y)
        ])
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#10151f")))
        painter.drawPolygon(ground_poly)

        # Metric grid squares (every 5m lateral, every 10m longitudinal)
        pen = QtGui.QPen(QtGui.QColor(0, 180, 240, 35), 1.0)
        painter.setPen(pen)

        for gx in np.arange(-15.0, 16.0, 5.0):
            p1_x, p1_y, _ = self._project(gx, 0.0, w, h)
            p2_x, p2_y, _ = self._project(gx, 70.0, w, h)
            painter.drawLine(QtCore.QPointF(p1_x, p1_y), QtCore.QPointF(p2_x, p2_y))

        for gz in np.arange(5.0, 71.0, 10.0):
            p1_x, p1_y, _ = self._project(-15.0, gz, w, h)
            p2_x, p2_y, _ = self._project(15.0, gz, w, h)
            painter.drawLine(QtCore.QPointF(p1_x, p1_y), QtCore.QPointF(p2_x, p2_y))

            # Ground metric label
            lx, ly, _ = self._project(15.2, gz, w, h)
            painter.setFont(QtGui.QFont("SF Pro Display", 8))
            painter.setPen(QtGui.QColor(0, 180, 240, 90))
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

            pen = QtGui.QPen(QtGui.QColor(0, 200, 255, 30), 1.0, QtCore.Qt.DashLine)
            painter.setPen(pen)
            for k in range(len(pts) - 1):
                painter.drawLine(pts[k], pts[k + 1])

            # Label
            lx, ly, _ = self._project(dist * 0.85, dist * 0.52, w, h)
            painter.setFont(QtGui.QFont("SF Pro Display", 8))
            painter.setPen(QtGui.QColor(0, 200, 255, 90))
            painter.drawText(int(lx), int(ly), f"{int(dist)}m")

    def _draw_panoptic_lanes(self, painter: QtGui.QPainter, w: int, h: int):
        """Draw lane splines from perception engine."""
        pen = QtGui.QPen(QtGui.QColor(0, 245, 255, 170), 2.2)
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
            width = 4.5
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
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 215, 30, 190)))
                painter.drawPolygon(poly)

    def _draw_ego_vehicle(self, painter: QtGui.QPainter, w: int, h: int):
        """Detailed aerodynamic vehicle at (0, 0) with wheels and headlights."""
        # 1. Headlight beam
        p_origin_x, p_origin_y, _ = self._project(0.0, 1.2, w, h)
        p_left_x, p_left_y, _ = self._project(-4.5, 32.0, w, h)
        p_right_x, p_right_y, _ = self._project(4.5, 32.0, w, h)

        beam_poly = QtGui.QPolygonF([
            QtCore.QPointF(p_origin_x, p_origin_y),
            QtCore.QPointF(p_left_x, p_left_y),
            QtCore.QPointF(p_right_x, p_right_y)
        ])
        beam_grad = QtGui.QLinearGradient(p_origin_x, p_origin_y, p_origin_x, p_left_y)
        beam_grad.setColorAt(0.0, QtGui.QColor(0, 220, 255, 60))
        beam_grad.setColorAt(1.0, QtGui.QColor(0, 220, 255, 0))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(beam_grad))
        painter.drawPolygon(beam_poly)

        # 2. Host car body (4.6m length, 1.85m width)
        hw = 0.92
        hl = 2.30

        # Wheels (4 corners)
        self._draw_wheel(painter, -hw * 1.02, hl * 0.65, 0.0, w, h)
        self._draw_wheel(painter, hw * 1.02, hl * 0.65, 0.0, w, h)
        self._draw_wheel(painter, -hw * 1.02, -hl * 0.65, 0.0, w, h)
        self._draw_wheel(painter, hw * 1.02, -hl * 0.65, 0.0, w, h)

        # Main Chassis
        c1_x, c1_y, _ = self._project(-hw, hl, w, h)
        c2_x, c2_y, _ = self._project(hw, hl, w, h)
        c3_x, c3_y, _ = self._project(hw, -hl, w, h)
        c4_x, c4_y, _ = self._project(-hw, -hl, w, h)

        car_poly = QtGui.QPolygonF([
            QtCore.QPointF(c1_x, c1_y),
            QtCore.QPointF(c2_x, c2_y),
            QtCore.QPointF(c3_x, c3_y),
            QtCore.QPointF(c4_x, c4_y)
        ])
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 230, 255, 230), 1.8))
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#151d27")))
        painter.drawPolygon(car_poly)

        # Windshield & Roof
        w1_x, w1_y, _ = self._project(-hw * 0.72, hl * 0.35, w, h)
        w2_x, w2_y, _ = self._project(hw * 0.72, hl * 0.35, w, h)
        w3_x, w3_y, _ = self._project(hw * 0.72, -hl * 0.25, w, h)
        w4_x, w4_y, _ = self._project(-hw * 0.72, -hl * 0.25, w, h)
        ws_poly = QtGui.QPolygonF([
            QtCore.QPointF(w1_x, w1_y),
            QtCore.QPointF(w2_x, w2_y),
            QtCore.QPointF(w3_x, w3_y),
            QtCore.QPointF(w4_x, w4_y)
        ])
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 180, 255, 110)))
        painter.drawPolygon(ws_poly)

        # Headlight points
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 250, 255), 3.0))
        painter.drawPoint(QtCore.QPointF(c1_x, c1_y))
        painter.drawPoint(QtCore.QPointF(c2_x, c2_y))

    def _draw_wheel(self, painter: QtGui.QPainter, X: float, Z: float, yaw: float, w: int, h: int):
        """Draw tire rectangle on road plane."""
        # Wheel dim ~ 0.65m long, 0.22m wide
        p1_x, p1_y, sc = self._project(X - 0.11, Z - 0.32, w, h)
        p2_x, p2_y, _ = self._project(X + 0.11, Z - 0.32, w, h)
        p3_x, p3_y, _ = self._project(X + 0.11, Z + 0.32, w, h)
        p4_x, p4_y, _ = self._project(X - 0.11, Z + 0.32, w, h)

        poly = QtGui.QPolygonF([
            QtCore.QPointF(p1_x, p1_y),
            QtCore.QPointF(p2_x, p2_y),
            QtCore.QPointF(p3_x, p3_y),
            QtCore.QPointF(p4_x, p4_y)
        ])
        painter.setPen(QtGui.QPen(QtGui.QColor(80, 100, 120), 1.0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(20, 25, 32)))
        painter.drawPolygon(poly)

    def _draw_fixed_camera_origin(self, painter: QtGui.QPainter, w: int, h: int):
        """In fixed camera mode: render camera origin and observation frustum."""
        cam_x, cam_y, _ = self._project(0.0, 0.0, w, h)
        painter.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248), 2.0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(14, 165, 233, 80)))
        painter.drawEllipse(QtCore.QPointF(cam_x, cam_y), 6, 6)

        # Frustum lines
        f_left_x, f_left_y, _ = self._project(-12.0, 50.0, w, h)
        f_right_x, f_right_y, _ = self._project(12.0, 50.0, w, h)
        frust_pen = QtGui.QPen(QtGui.QColor(56, 189, 248, 60), 1.2, QtCore.Qt.DashLine)
        painter.setPen(frust_pen)
        painter.drawLine(QtCore.QPointF(cam_x, cam_y), QtCore.QPointF(f_left_x, f_left_y))
        painter.drawLine(QtCore.QPointF(cam_x, cam_y), QtCore.QPointF(f_right_x, f_right_y))

    def _draw_3d_actor(self, painter: QtGui.QPainter, trk: TrackedObject, w: int, h: int, is_moving: bool):
        """
        Renders true 3D models:
        - Vehicles: 3D body chassis with wheels, cabin, and proper heading angle.
        - Pedestrians: Upright 3D vertical standing figure with ground halo.
        - Cyclists: Upright rider on two-wheeler frame.
        """
        X = trk.X
        Z = trk.Z
        yaw = trk.yaw

        # 1. Pedestrian Model (Vertical Upright Human Figure)
        if trk.class_name == 'person':
            self._draw_3d_pedestrian(painter, trk, X, Z, w, h)
            return

        # 2. Two-Wheeler (Bicycle / Motorcycle)
        if trk.class_name in ('bicycle', 'motorcycle'):
            self._draw_3d_twowheeler(painter, trk, X, Z, yaw, w, h)
            return

        # 3. Vehicle Model (Car / Truck / Bus)
        self._draw_3d_vehicle(painter, trk, X, Z, yaw, w, h)

    def _draw_3d_pedestrian(self, painter: QtGui.QPainter, trk: TrackedObject, X: float, Z: float, w: int, h: int):
        """Render upright 3D human figure in warm coral."""
        # Ground shadow / contact halo
        gx, gy, sc = self._project(X, Z, w, h)
        halo_rx = max(4.0, 7.0 * sc)
        halo_ry = max(2.5, 4.0 * sc)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 59, 105, 120), 1.2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 59, 105, 40)))
        painter.drawEllipse(QtCore.QPointF(gx, gy), halo_rx, halo_ry)

        # Height in pixels: person height ~ 1.75m
        person_h_px = 18.0 * sc
        torso_top_y = gy - person_h_px * 0.75
        head_top_y = gy - person_h_px

        # Vertical body torso
        body_pen = QtGui.QPen(QtGui.QColor(255, 59, 105), max(2.0, 3.5 * sc))
        painter.setPen(body_pen)
        painter.drawLine(QtCore.QPointF(gx, gy), QtCore.QPointF(gx, torso_top_y))

        # Head sphere
        head_r = max(2.5, 3.5 * sc)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 80, 130)))
        painter.drawEllipse(QtCore.QPointF(gx, head_top_y), head_r, head_r)

        # Label tag
        self._draw_actor_tag(painter, gx, head_top_y - 6, f"Piéton {Z:.1f}m", QtGui.QColor(255, 59, 105))

    def _draw_3d_twowheeler(self, painter: QtGui.QPainter, trk: TrackedObject, X: float, Z: float, yaw: float, w: int, h: int):
        """Render two-wheeler bike and rider in amber gold."""
        gx, gy, sc = self._project(X, Z, w, h)

        # Bike length ~ 1.8m
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        f_x = X + 0.9 * sin_y
        f_z = Z + 0.9 * cos_y
        r_x = X - 0.9 * sin_y
        r_z = Z - 0.9 * cos_y

        f_sx, f_sy, _ = self._project(f_x, f_z, w, h)
        r_sx, r_sy, _ = self._project(r_x, r_z, w, h)

        # Bike frame
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 185, 20), 2.2))
        painter.drawLine(QtCore.QPointF(r_sx, r_sy), QtCore.QPointF(f_sx, f_sy))

        # Rider upright torso
        rider_h = 15.0 * sc
        painter.drawLine(QtCore.QPointF(gx, gy), QtCore.QPointF(gx, gy - rider_h))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 210, 40)))
        painter.drawEllipse(QtCore.QPointF(gx, gy - rider_h - 2), 3, 3)

        self._draw_actor_tag(painter, gx, gy - rider_h - 10, f"{trk.class_name.capitalize()} {Z:.1f}m", QtGui.QColor(255, 185, 20))

    def _draw_3d_vehicle(self, painter: QtGui.QPainter, trk: TrackedObject, X: float, Z: float, yaw: float, w: int, h: int):
        """Render true 3D vehicle with body chassis, wheels, windshield, and proper heading."""
        is_truck = trk.class_name in ('truck', 'bus')
        car_w = 2.4 if is_truck else 1.85
        car_l = 8.5 if is_truck else 4.6
        car_h = 2.8 if is_truck else 1.45

        hw = car_w / 2.0
        hl = car_l / 2.0

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # 4 Ground corners of the car
        # Local offsets: (dx, dz)
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

        # Ground contact shadow
        shadow_poly = QtGui.QPolygonF(ground_pts)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(10, 15, 22, 180)))
        painter.drawPolygon(shadow_poly)

        # Accent color
        if trk.warning_level == 'critical':
            accent = QtGui.QColor(255, 42, 60)
            body_fill = QtGui.QColor(255, 42, 60, 45)
        else:
            accent = QtGui.QColor(0, 230, 255)
            body_fill = QtGui.QColor(0, 230, 255, 35)

        # 3D Top roof face (shifted upwards on screen by car_h)
        _, _, sc_center = self._project(X, Z, w, h)
        roof_shift_px = car_h * 11.0 * sc_center

        roof_pts = [QtCore.QPointF(p.x(), p.y() - roof_shift_px) for p in ground_pts]

        # Draw Side pillars (connecting ground to roof)
        pillar_pen = QtGui.QPen(accent, 1.4)
        painter.setPen(pillar_pen)
        for i in range(4):
            painter.drawLine(ground_pts[i], roof_pts[i])

        # Bottom Chassis outline
        painter.setPen(QtGui.QPen(accent, 1.6))
        painter.setBrush(QtGui.QBrush(body_fill))
        painter.drawPolygon(shadow_poly)

        # Top Roof outline
        roof_poly = QtGui.QPolygonF(roof_pts)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(16, 24, 34, 210)))
        painter.drawPolygon(roof_poly)

        # Windshield on roof (front half)
        ws_pts = [
            QtCore.QPointF(roof_pts[0].x() * 0.85 + roof_pts[3].x() * 0.15, roof_pts[0].y() * 0.85 + roof_pts[3].y() * 0.15),
            QtCore.QPointF(roof_pts[1].x() * 0.85 + roof_pts[2].x() * 0.15, roof_pts[1].y() * 0.85 + roof_pts[2].y() * 0.15),
            QtCore.QPointF(roof_pts[1].x() * 0.35 + roof_pts[2].x() * 0.65, roof_pts[1].y() * 0.35 + roof_pts[2].y() * 0.65),
            QtCore.QPointF(roof_pts[0].x() * 0.35 + roof_pts[3].x() * 0.65, roof_pts[0].y() * 0.35 + roof_pts[3].y() * 0.65)
        ]
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 180, 255, 90)))
        painter.drawPolygon(QtGui.QPolygonF(ws_pts))

        # Directional Ground Arrow if moving
        if trk.speed_kmh > 4.0:
            arrow_tip_x = X + trk.vx * 1.2
            arrow_tip_z = Z + trk.vz * 1.2
            asx, asy, _ = self._project(arrow_tip_x, arrow_tip_z, w, h)
            mid_x = (ground_pts[0].x() + ground_pts[1].x()) / 2.0
            mid_y = (ground_pts[0].y() + ground_pts[1].y()) / 2.0
            arrow_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.6)
            painter.setPen(arrow_pen)
            painter.drawLine(QtCore.QPointF(mid_x, mid_y), QtCore.QPointF(asx, asy))

        # Top tag
        top_y = min(p.y() for p in roof_pts)
        lbl = f"Voiture {Z:.1f}m" if not is_truck else f"Poids lourd {Z:.1f}m"
        self._draw_actor_tag(painter, (roof_pts[0].x() + roof_pts[1].x()) / 2.0, top_y - 6, lbl, accent)

    def _draw_actor_tag(self, painter: QtGui.QPainter, x: float, y: float, text: str, color: QtGui.QColor):
        """Minimalist compact badge above actor."""
        painter.setFont(QtGui.QFont("SF Pro Display", 8, QtGui.QFont.DemiBold))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 8
        th = fm.height() + 2
        bx = int(x - tw / 2.0)
        by = int(y - th)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(10, 14, 20, 210)))
        painter.drawRoundedRect(bx, by, tw, th, 3, 3)

        painter.setPen(QtGui.QPen(color, 1.0))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(bx, by, tw, th, 3, 3)

        painter.setPen(QtGui.QColor(245, 245, 250))
        painter.drawText(QtCore.QRect(bx, by, tw, th), QtCore.Qt.AlignCenter, text)

    def _draw_reference_frame_badge(self, painter: QtGui.QPainter, w: int, h: int, is_moving: bool, speed: float):
        """Status badge indicating active spatial reference frame."""
        if is_moving:
            badge_text = f"● CAMÉRA EMBARQUÉE  •  {speed:.0f} km/h"
            badge_color = QtGui.QColor(16, 185, 129) # Emerald
        else:
            badge_text = "● CAMÉRA FIXE (SURVEILLANCE)  •  0 km/h"
            badge_color = QtGui.QColor(56, 189, 248) # Slate blue

        painter.setFont(QtGui.QFont("SF Pro Display", 8, QtGui.QFont.Bold))
        fm = painter.fontMetrics()
        bw = fm.horizontalAdvance(badge_text) + 16
        bh = fm.height() + 6

        # Top-left position
        bx = 14
        by = 14

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(10, 14, 22, 220)))
        painter.drawRoundedRect(bx, by, bw, bh, 4, 4)

        painter.setPen(QtGui.QPen(badge_color, 1.2))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(bx, by, bw, bh, 4, 4)

        painter.setPen(badge_color)
        painter.drawText(QtCore.QRect(bx, by, bw, bh), QtCore.Qt.AlignCenter, badge_text)
