"""
bev_widget.py - Tesla FSD / Waymo-style 3D Bird's-Eye View (BEV) Vector Space.
Renders Ego-vehicle, surrounding 3D traffic hulls, lane splines, crosswalk zebra stripes,
distance radar rings, and motion kinematics.
"""

from typing import Optional, List, Tuple
import math
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from perception_engine import PerceptionResult
from tracker import TrackedObject


class BEVWidget(QtWidgets.QWidget):
    """
    3D Vector Space Bird's-Eye View displaying real-time metric perception.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #080a0e; border-radius: 8px;")

        # Camera & View settings
        self.view_mode = "3D_FSD" # "3D_FSD" or "TOP_DOWN"
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start = QtCore.QPoint()
        self.is_dragging = False

        # Display toggles
        self.show_radar_rings = True
        self.show_grid = True
        self.show_trails = True
        self.show_velocity_vectors = True
        self.show_badges = True
        self.show_lanes = True
        self.show_crosswalks = True

        # State
        self.current_result: Optional[PerceptionResult] = None

        # Enable mouse tracking for interactive navigation
        self.setMouseTracking(True)

    def set_view_mode(self, mode: str):
        if mode in ("3D_FSD", "TOP_DOWN"):
            self.view_mode = mode
            self.update()

    def update_result(self, result: PerceptionResult):
        self.current_result = result
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
            cy = h * 0.82 + self.pan_y
            depth_factor = 0.016 / max(self.zoom, 0.2)
            persp = 1.0 / (1.0 + max(Z, 0.0) * depth_factor)
            sx = cx + (X * 22.0 * self.zoom) * persp
            sy = cy - (Z * 9.5 * self.zoom) * persp
            scale = self.zoom * persp
            return sx, sy, scale
        else:
            # Pure orthographic top-down
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

        # 1. Background gradient (Titanium dark space)
        bg_grad = QtGui.QRadialGradient(w / 2.0, h * 0.5, max(w, h))
        bg_grad.setColorAt(0.0, QtGui.QColor("#0d1219"))
        bg_grad.setColorAt(1.0, QtGui.QColor("#06080b"))
        painter.fillRect(0, 0, w, h, QtGui.QBrush(bg_grad))

        # 2. Metric Grid & Radar Distance Rings
        if self.show_radar_rings:
            self._draw_radar_rings(painter, w, h)

        if self.show_grid:
            self._draw_metric_grid(painter, w, h)

        # 3. Projected 3D Lanes
        if self.show_lanes and self.current_result is not None:
            self._draw_3d_lanes(painter, w, h)

        # 4. Projected 3D Crosswalks
        if self.show_crosswalks and self.current_result is not None:
            self._draw_3d_crosswalks(painter, w, h)

        # 5. Ego Vehicle & Headlight Beams
        self._draw_ego_vehicle(painter, w, h)

        # 6. Surrounding Tracked 3D Actors (Vehicles, Pedestrians)
        if self.current_result is not None:
            for trk in self.current_result.tracks:
                self._draw_tracked_actor(painter, trk, w, h)

        # 7. Tactical HUD Overlay (Mode badge, Zoom level, Cardinal axes)
        self._draw_hud_overlay(painter, w, h)

    def _draw_radar_rings(self, painter: QtGui.QPainter, w: int, h: int):
        """Draw concentric metric radar distance rings with distance callouts."""
        distances = [10.0, 25.0, 50.0, 75.0, 100.0]

        for dist in distances:
            # Sample circle points in 3D ground plane
            pts = []
            num_pts = 48
            for i in range(num_pts + 1):
                angle = (i / num_pts) * 2.0 * math.pi
                gx = dist * math.sin(angle)
                gz = dist * math.cos(angle)
                if gz >= -2.0: # Only draw in front of host vehicle
                    sx, sy, _ = self._project(gx, gz, w, h)
                    pts.append(QtCore.QPointF(sx, sy))

            if len(pts) > 1:
                pen = QtGui.QPen(QtGui.QColor(0, 200, 255, 35), 1.0, QtCore.Qt.DashLine)
                painter.setPen(pen)
                for k in range(len(pts) - 1):
                    painter.drawLine(pts[k], pts[k + 1])

                # Distance label at right flank
                lbl_x, lbl_y, _ = self._project(dist * 0.9, dist * 0.4, w, h)
                painter.setFont(QtGui.QFont("SF Pro Display", 8, QtGui.QFont.Medium))
                painter.setPen(QtGui.QColor(0, 200, 255, 110))
                painter.drawText(int(lbl_x) + 4, int(lbl_y), f"{int(dist)}m")

    def _draw_metric_grid(self, painter: QtGui.QPainter, w: int, h: int):
        """Draw subtle longitudinal and lateral road grid lines."""
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 18), 1.0)
        painter.setPen(pen)

        # Lane dividers: X = -3.7m (left lane), X = 3.7m (right lane)
        for gx in [-7.4, -3.7, 3.7, 7.4]:
            p1_x, p1_y, _ = self._project(gx, 0.0, w, h)
            p2_x, p2_y, _ = self._project(gx, 85.0, w, h)
            painter.drawLine(QtCore.QPointF(p1_x, p1_y), QtCore.QPointF(p2_x, p2_y))

        # Horizontal distance lines every 15m
        for gz in [15.0, 30.0, 45.0, 60.0, 75.0]:
            p1_x, p1_y, _ = self._project(-12.0, gz, w, h)
            p2_x, p2_y, _ = self._project(12.0, gz, w, h)
            painter.drawLine(QtCore.QPointF(p1_x, p1_y), QtCore.QPointF(p2_x, p2_y))

    def _draw_3d_lanes(self, painter: QtGui.QPainter, w: int, h: int):
        """Render lane splines in 3D ground coordinates."""
        if not self.current_result.bev_lanes_3d:
            return

        pen = QtGui.QPen(QtGui.QColor(0, 240, 255, 180), 2.2)
        painter.setPen(pen)

        # Draw dense points as glowing road ribbons
        for X, Z in self.current_result.bev_lanes_3d:
            sx, sy, sc = self._project(X, Z, w, h)
            pt_size = max(1.5, 3.0 * sc)
            painter.drawEllipse(QtCore.QPointF(sx, sy), pt_size, pt_size)

    def _draw_3d_crosswalks(self, painter: QtGui.QPainter, w: int, h: int):
        """Render metric zebra crosswalk stripes on ground plane."""
        for cw in self.current_result.crosswalks:
            cx, cz = cw['X'], cw['Z']
            num_stripes = max(5, cw.get('stripes', 6))

            # Zebra crossing width ~ 4.5m across lane, depth ~ 2.5m
            width = 4.5
            depth = 2.4
            stripe_w = width / (num_stripes * 2.0)

            for i in range(num_stripes):
                sx_offset = -width / 2.0 + i * (stripe_w * 2.0)
                # 4 ground corners of this stripe
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
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 220, 50, 180))) # Amber zebra stripe
                painter.drawPolygon(poly)

            # Badge above crosswalk
            bx, by, _ = self._project(cx, cz + depth, w, h)
            painter.setFont(QtGui.QFont("SF Pro Display", 8, QtGui.QFont.Bold))
            painter.setPen(QtGui.QColor(255, 210, 40))
            painter.drawText(int(bx) - 40, int(by) - 6, f"🚸 CROSSWALK {cz:.1f}m")

    def _draw_ego_vehicle(self, painter: QtGui.QPainter, w: int, h: int):
        """Render host aerodynamic vehicle at (0, 0) with forward illumination beam."""
        # 1. Forward headlight illumination cone
        p_origin_x, p_origin_y, _ = self._project(0.0, 1.0, w, h)
        p_left_x, p_left_y, _ = self._project(-5.0, 35.0, w, h)
        p_right_x, p_right_y, _ = self._project(5.0, 35.0, w, h)

        beam_poly = QtGui.QPolygonF([
            QtCore.QPointF(p_origin_x, p_origin_y),
            QtCore.QPointF(p_left_x, p_left_y),
            QtCore.QPointF(p_right_x, p_right_y)
        ])

        beam_grad = QtGui.QLinearGradient(p_origin_x, p_origin_y, p_origin_x, p_left_y)
        beam_grad.setColorAt(0.0, QtGui.QColor(0, 220, 255, 75))
        beam_grad.setColorAt(1.0, QtGui.QColor(0, 220, 255, 0))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(beam_grad))
        painter.drawPolygon(beam_poly)

        # 2. Host Vehicle Body (Length 4.7m, Width 1.9m)
        hw = 0.95
        hl = 2.35
        # 4 corners of car body in ground plane
        c_fl_x, c_fl_y, sc = self._project(-hw, hl, w, h)
        c_fr_x, c_fr_y, _ = self._project(hw, hl, w, h)
        c_rr_x, c_rr_y, _ = self._project(hw, -hl, w, h)
        c_rl_x, c_rl_y, _ = self._project(-hw, -hl, w, h)

        car_poly = QtGui.QPolygonF([
            QtCore.QPointF(c_fl_x, c_fl_y),
            QtCore.QPointF(c_fr_x, c_fr_y),
            QtCore.QPointF(c_rr_x, c_rr_y),
            QtCore.QPointF(c_rl_x, c_rl_y)
        ])

        # Solid sleek hull
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#1b2430")))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 230, 255, 230), 2.0))
        painter.drawPolygon(car_poly)

        # Windshield
        w_fl_x, w_fl_y, _ = self._project(-hw * 0.75, hl * 0.4, w, h)
        w_fr_x, w_fr_y, _ = self._project(hw * 0.75, hl * 0.4, w, h)
        w_rr_x, w_rr_y, _ = self._project(hw * 0.75, -hl * 0.2, w, h)
        w_rl_x, w_rl_y, _ = self._project(-hw * 0.75, -hl * 0.2, w, h)
        ws_poly = QtGui.QPolygonF([
            QtCore.QPointF(w_fl_x, w_fl_y),
            QtCore.QPointF(w_fr_x, w_fr_y),
            QtCore.QPointF(w_rr_x, w_rr_y),
            QtCore.QPointF(w_rl_x, w_rl_y)
        ])
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 180, 255, 120)))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawPolygon(ws_poly)

        # Headlight accents (bright cyan)
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 250, 255), 3.0))
        painter.drawPoint(QtCore.QPointF(c_fl_x, c_fl_y))
        painter.drawPoint(QtCore.QPointF(c_fr_x, c_fr_y))

        # Taillight accents (ruby red)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 30, 30), 3.0))
        painter.drawPoint(QtCore.QPointF(c_rl_x, c_rl_y))
        painter.drawPoint(QtCore.QPointF(c_rr_x, c_rr_y))

        # EGO label
        painter.setFont(QtGui.QFont("SF Pro Display", 8, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor(0, 240, 255))
        painter.drawText(int(c_fl_x + c_fr_x) // 2 - 12, int(c_rr_y) + 16, "EGO")

    def _draw_tracked_actor(self, painter: QtGui.QPainter, trk: TrackedObject, w: int, h: int):
        """Render oriented 3D cuboid, motion trail, velocity vector, and metric badge."""
        # 1. Motion Trail
        if self.show_trails and len(trk.trail) > 1:
            pen = QtGui.QPen(QtGui.QColor(0, 200, 255, 50), 1.5, QtCore.Qt.DotLine)
            painter.setPen(pen)
            trail_pts = []
            for tx, tz in trk.trail:
                sx, sy, _ = self._project(tx, tz, w, h)
                trail_pts.append(QtCore.QPointF(sx, sy))
            for k in range(len(trail_pts) - 1):
                painter.drawLine(trail_pts[k], trail_pts[k + 1])

        # 2. Color assignment
        if trk.class_name in ('person', 'bicycle'):
            hull_color = QtGui.QColor(255, 60, 180)
            fill_color = QtGui.QColor(255, 60, 180, 70)
        elif trk.warning_level == 'critical':
            hull_color = QtGui.QColor(255, 40, 40)
            fill_color = QtGui.QColor(255, 40, 40, 90)
        elif trk.warning_level == 'warning':
            hull_color = QtGui.QColor(255, 170, 0)
            fill_color = QtGui.QColor(255, 170, 0, 70)
        else:
            hull_color = QtGui.QColor(0, 230, 255)
            fill_color = QtGui.QColor(0, 230, 255, 50)

        # 3. 3D Cuboid in BEV
        # Corners order: 0..3 bottom (fl, fr, rr, rl), 4..7 top (fl, fr, rr, rl)
        corners = trk.corners_3d
        screen_pts = []
        for i in range(8):
            # In BEV, Y is vertical above ground
            # Top-down perspective projects both bottom and top
            cx, cy, sc = self._project(corners[i, 0], corners[i, 2], w, h)
            # In 3D FSD view, shift top face upward on screen based on height Y
            if self.view_mode == "3D_FSD" and i >= 4:
                height_px = corners[i, 1] * 14.0 * sc
                cy -= height_px
            screen_pts.append(QtCore.QPointF(cx, cy))

        # Draw bottom face
        pen = QtGui.QPen(hull_color, 1.8)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(fill_color))

        # Side pillars
        for i in range(4):
            painter.drawLine(screen_pts[i], screen_pts[i + 4])

        # Top face
        top_poly = QtGui.QPolygonF([screen_pts[4], screen_pts[5], screen_pts[6], screen_pts[7]])
        painter.drawPolygon(top_poly)

        # 4. Velocity Vector Arrow
        if self.show_velocity_vectors and (abs(trk.vx) > 0.3 or abs(trk.vz) > 0.3):
            tip_x = trk.X + trk.vx * 1.0 # 1-second lookahead
            tip_z = trk.Z + trk.vz * 1.0
            p1_x, p1_y, _ = self._project(trk.X, trk.Z, w, h)
            p2_x, p2_y, _ = self._project(tip_x, tip_z, w, h)
            arrow_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.8)
            painter.setPen(arrow_pen)
            painter.drawLine(QtCore.QPointF(p1_x, p1_y), QtCore.QPointF(p2_x, p2_y))

        # 5. Metric Callout Badge
        if self.show_badges:
            badge_x = int(screen_pts[4].x() + screen_pts[5].x()) // 2
            badge_y = int(min(screen_pts[4].y(), screen_pts[5].y())) - 10

            lbl_text = f"#{trk.track_id} {trk.class_name.upper()} • {trk.Z:.1f}m"
            if trk.speed_kmh > 3.0:
                lbl_text += f" • {trk.speed_kmh:.0f}km/h"
            if trk.ttc is not None:
                lbl_text += f" • TTC {trk.ttc:.1f}s"

            painter.setFont(QtGui.QFont("SF Pro Display", 8, QtGui.QFont.Bold))
            fm = painter.fontMetrics()
            bw = fm.horizontalAdvance(lbl_text) + 12
            bh = fm.height() + 4
            bx = badge_x - bw // 2
            by = badge_y - bh

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(10, 15, 20, 220)))
            painter.drawRoundedRect(bx, by, bw, bh, 3, 3)

            painter.setPen(QtGui.QPen(hull_color, 1.0))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRoundedRect(bx, by, bw, bh, 3, 3)

            painter.setPen(QtGui.QColor(245, 245, 250))
            painter.drawText(QtCore.QRect(bx, by, bw, bh), QtCore.Qt.AlignCenter, lbl_text)

    def _draw_hud_overlay(self, painter: QtGui.QPainter, w: int, h: int):
        """Tactical HUD status, mode indicator, and coordinate scales."""
        # Top-left Mode Indicator
        painter.setFont(QtGui.QFont("SF Pro Display", 8, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor(0, 220, 255, 170))
        mode_str = "3D FSD VECTOR SPACE" if self.view_mode == "3D_FSD" else "TOP-DOWN BEV RADAR"
        painter.drawText(16, 24, f"TESLA {mode_str} • ZOOM {self.zoom:.1f}X")

        # Bottom-right Legend
        painter.setFont(QtGui.QFont("SF Pro Display", 8))
        painter.setPen(QtGui.QColor(140, 160, 180))
        painter.drawText(w - 180, h - 16, "DRAG: PAN • SCROLL: ZOOM")
