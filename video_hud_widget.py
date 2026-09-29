"""
video_hud_widget.py - Real Dashcam Video View with Minimalist Augmented Reality (AR) HUD.
Renders clean tactical corner brackets, 3D wireframe boxes, drivable road carpet,
pedestrian/vehicle distance badges, and interactive hover synchronization.
"""

from typing import Optional, List, Tuple, Dict
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from perception_engine import PerceptionResult
from bev_geometry import BEVGeometry


FRENCH_CLASSES = {
    'car': 'Voiture',
    'person': 'Piéton',
    'bicycle': 'Vélo',
    'motorcycle': 'Moto',
    'bus': 'Bus',
    'truck': 'Camion',
    'default': 'Cible'
}


class VideoHUDWidget(QtWidgets.QWidget):
    """
    High-performance QWidget displaying camera video stream with a sleek,
    minimalist HUD overlay (Tesla / Mobileye inspired).
    """

    actor_hovered = QtCore.Signal(object)  # Emits int track_id or None
    actor_selected = QtCore.Signal(object) # Emits int track_id or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #07090d; border-radius: 8px;")

        # Options
        self.show_drivable_carpet = True
        self.show_lane_lines = True
        self.show_boxes = True
        self.box_mode = "2D_BRACKETS" # "2D_BRACKETS" or "3D_CUBOID"
        self.show_crosswalks = True
        self.show_horizon = False

        # State
        self.current_frame: Optional[np.ndarray] = None
        self.current_result: Optional[PerceptionResult] = None
        self.geom: Optional[BEVGeometry] = None

        # Interactive state
        self.hovered_track_id: Optional[int] = None
        self.selected_track_id: Optional[int] = None

        # Box screen mappings: track_id -> [sx1, sy1, sx2, sy2]
        self._screen_boxes: Dict[int, List[float]] = {}

        self.setMouseTracking(True)

    def update_frame(self, frame: np.ndarray, result: PerceptionResult, geom: BEVGeometry):
        self.current_frame = frame
        self.current_result = result
        self.geom = geom
        self.update()

    def set_box_mode(self, mode: str):
        if mode in ("2D_BRACKETS", "3D_CUBOID", "OFF"):
            self.box_mode = mode
            self.show_boxes = (mode != "OFF")
            self.update()

    def set_highlighted_actor(self, track_id: Optional[int]):
        """Highlights box when hovered on the 3D BEV screen."""
        if self.hovered_track_id != track_id:
            self.hovered_track_id = track_id
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pos = event.pos()
        hovered = self._find_actor_at(pos.x(), pos.y())
        if hovered != self.hovered_track_id:
            self.hovered_track_id = hovered
            self.actor_hovered.emit(hovered)
            self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            pos = event.pos()
            clicked = self._find_actor_at(pos.x(), pos.y())
            if clicked is not None:
                self.selected_track_id = clicked
                self.actor_selected.emit(clicked)
                self.update()

    def _find_actor_at(self, px: float, py: float) -> Optional[int]:
        """Finds track ID whose 2D screen box contains (px, py)."""
        for t_id, box in self._screen_boxes.items():
            if box[0] <= px <= box[2] and box[1] <= py <= box[3]:
                return t_id
        return None

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        w_widget = self.width()
        h_widget = self.height()
        self._screen_boxes.clear()

        if self.current_frame is None:
            painter.fillRect(0, 0, w_widget, h_widget, QtGui.QColor("#080b0f"))
            painter.setPen(QtGui.QColor("#64748b"))
            painter.setFont(QtGui.QFont("SF Pro Display", 13, QtGui.QFont.Medium))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "AUCUN SIGNAL VIDÉO")
            return

        frame = self.current_frame
        fh, fw = frame.shape[:2]

        scale = min(w_widget / fw, h_widget / fh)
        dw = int(fw * scale)
        dh = int(fh * scale)
        dx = (w_widget - dw) // 2
        dy = (h_widget - dh) // 2

        display_img = frame.copy()

        # 1. Drivable Road Carpet Overlay
        if self.show_drivable_carpet and self.current_result is not None:
            da_mask = self.current_result.drivable_mask
            if da_mask is not None and da_mask.max() > 0:
                overlay = display_img.copy()
                overlay[da_mask > 0] = [170, 225, 50] # Glowing cyan-green
                cv2.addWeighted(overlay, 0.22, display_img, 0.78, 0, display_img)

        # 2. Lane Lines Overlay
        if self.show_lane_lines and self.current_result is not None:
            lane_mask = self.current_result.lane_mask
            if lane_mask is not None and lane_mask.max() > 0:
                display_img[lane_mask > 0] = [255, 230, 0] # Electric yellow/cyan

        # Convert to QImage and draw
        display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(
            display_rgb.data,
            fw,
            fh,
            fw * 3,
            QtGui.QImage.Format_RGB888
        )
        target_rect = QtCore.QRect(dx, dy, dw, dh)
        painter.drawImage(target_rect, qimg)

        # Coordinate transform from frame pixels to widget coordinates:
        def to_widget(x: float, y: float) -> Tuple[float, float]:
            return dx + x * scale, dy + y * scale

        # 3. Horizon line (if enabled)
        if self.show_horizon and self.current_result is not None:
            _, hy = to_widget(0, self.current_result.horizon_y)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 230, 255, 90), 1.0, QtCore.Qt.DashLine))
            painter.drawLine(QtCore.QPointF(dx, hy), QtCore.QPointF(dx + dw, hy))

        # 4. Trajectory Corridor Road Projection
        if self.current_result and self.geom:
            self._draw_road_trajectory_corridor(painter, to_widget)

        # 5. Actors Bounding Boxes & Distance Badges
        if self.show_boxes and self.current_result is not None:
            for trk in self.current_result.tracks:
                self._draw_actor_hud(painter, trk, to_widget)

        # 6. Forward Collision Warning Flash Banner
        if self.current_result and self.current_result.fcw_alert:
            self._draw_fcw_alert(painter, dx, dy, dw, dh)

    def _draw_road_trajectory_corridor(self, painter: QtGui.QPainter, to_widget):
        """Draws projected trajectory corridor onto the 2D video road."""
        corridor = self.current_result.trajectory_corridor
        if len(corridor) < 2 or self.geom is None:
            return

        hw = 1.05
        left_screen = []
        right_screen = []

        for x, z in corridor:
            u_l, v_l = self.geom.world_to_image(x - hw, 0.0, z)
            u_r, v_r = self.geom.world_to_image(x + hw, 0.0, z)
            if u_l is not None and v_l is not None:
                sx, sy = to_widget(u_l, v_l)
                left_screen.append(QtCore.QPointF(sx, sy))
            if u_r is not None and v_r is not None:
                sx, sy = to_widget(u_r, v_r)
                right_screen.append(QtCore.QPointF(sx, sy))

        if len(left_screen) >= 2 and len(right_screen) >= 2:
            poly = QtGui.QPolygonF(left_screen + list(reversed(right_screen)))
            is_fcw = self.current_result.fcw_alert
            fill = QtGui.QColor(239, 68, 68, 50) if is_fcw else QtGui.QColor(0, 230, 255, 30)
            edge = QtGui.QColor(239, 68, 68, 190) if is_fcw else QtGui.QColor(0, 230, 255, 120)
            painter.setPen(QtGui.QPen(edge, 1.5))
            painter.setBrush(QtGui.QBrush(fill))
            painter.drawPolygon(poly)

    def _draw_actor_hud(self, painter: QtGui.QPainter, trk, to_widget):
        """Draws tactical corner brackets and dark glass info badge for an actor."""
        x1, y1, x2, y2 = trk.box_2d
        sx1, sy1 = to_widget(x1, y1)
        sx2, sy2 = to_widget(x2, y2)
        bw = sx2 - sx1
        bh = sy2 - sy1

        # Store screen hit-box
        self._screen_boxes[trk.track_id] = [sx1, sy1, sx2, sy2]

        is_hovered = (trk.track_id == self.hovered_track_id)
        is_lead = (self.current_result and self.current_result.lead_vehicle and self.current_result.lead_vehicle.track_id == trk.track_id)

        # Color palette
        if trk.warning_level == 'critical':
            color = QtGui.QColor(239, 68, 68)  # Crimson
        elif is_lead:
            color = QtGui.QColor(245, 158, 11) # Amber Gold
        elif is_hovered:
            color = QtGui.QColor(0, 245, 255)  # Electric Cyan
        elif trk.class_name == 'person':
            color = QtGui.QColor(255, 90, 130) # Coral
        else:
            color = QtGui.QColor(56, 189, 248)  # Sky Blue

        line_w = 2.4 if (is_hovered or is_lead) else 1.8
        pen = QtGui.QPen(color, line_w)
        painter.setPen(pen)

        if self.box_mode == "2D_BRACKETS":
            # Tactical viseur corner brackets
            k = min(12.0, bw * 0.28, bh * 0.28)
            # Top-left
            painter.drawLine(QtCore.QPointF(sx1, sy1 + k), QtCore.QPointF(sx1, sy1))
            painter.drawLine(QtCore.QPointF(sx1, sy1), QtCore.QPointF(sx1 + k, sy1))
            # Top-right
            painter.drawLine(QtCore.QPointF(sx2 - k, sy1), QtCore.QPointF(sx2, sy1))
            painter.drawLine(QtCore.QPointF(sx2, sy1), QtCore.QPointF(sx2, sy1 + k))
            # Bottom-left
            painter.drawLine(QtCore.QPointF(sx1, sy2 - k), QtCore.QPointF(sx1, sy2))
            painter.drawLine(QtCore.QPointF(sx1, sy2), QtCore.QPointF(sx1 + k, sy2))
            # Bottom-right
            painter.drawLine(QtCore.QPointF(sx2 - k, sy2), QtCore.QPointF(sx2, sy2))
            painter.drawLine(QtCore.QPointF(sx2, sy2), QtCore.QPointF(sx2, sy2 - k))

            # Subtle glow if hovered or lead
            if is_hovered or is_lead:
                glow_pen = QtGui.QPen(QtGui.QColor(color.red(), color.green(), color.blue(), 50), 4.0)
                painter.setPen(glow_pen)
                painter.drawRect(QtCore.QRectF(sx1, sy1, bw, bh))
                painter.setPen(pen)

        elif self.box_mode == "3D_CUBOID" and self.geom:
            # 3D projected cuboid
            corners_3d = self.geom.get_3d_box_corners(trk.X, trk.Z, trk.class_name, yaw=trk.yaw)
            pts_2d = self.geom.project_3d_box(corners_3d)
            if pts_2d:
                screen_pts = [to_widget(u, v) for u, v in pts_2d]
                self._draw_wireframe_cuboid(painter, screen_pts, color)
            else:
                painter.drawRect(QtCore.QRectF(sx1, sy1, bw, bh))

        # Bottom distance tag capsule
        fr_name = FRENCH_CLASSES.get(trk.class_name, trk.class_name.capitalize())
        if is_lead:
            tag_text = f"★ LEAD #{trk.track_id} • {trk.Z:.1f}m"
        else:
            tag_text = f"#{trk.track_id} {fr_name} • {trk.Z:.1f}m"

        font = QtGui.QFont("SF Pro Display", 9, QtGui.QFont.Bold)
        painter.setFont(font)
        fm = QtGui.QFontMetrics(font)
        tw = fm.horizontalAdvance(tag_text)
        th = fm.height()

        tag_x = sx1 + (bw - tw) / 2.0 - 5
        tag_y = sy2 + 3
        tag_w = tw + 10
        tag_h = th + 2

        painter.setPen(QtGui.QPen(color, 1.0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(10, 15, 24, 220)))
        painter.drawRoundedRect(QtCore.QRectF(tag_x, tag_y, tag_w, tag_h), 3.0, 3.0)

        painter.setPen(QtGui.QColor("#ffffff"))
        painter.drawText(int(tag_x + 5), int(tag_y + th - 2), tag_text)

    def _draw_wireframe_cuboid(self, painter: QtGui.QPainter, pts: List[Tuple[float, float]], color: QtGui.QColor):
        """Draws 12 edges of a 3D bounding box."""
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4), # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pillars
        ]
        for i1, i2 in edges:
            p1 = QtCore.QPointF(pts[i1][0], pts[i1][1])
            p2 = QtCore.QPointF(pts[i2][0], pts[i2][1])
            painter.drawLine(p1, p2)

    def _draw_fcw_alert(self, painter: QtGui.QPainter, dx: int, dy: int, dw: int, dh: int):
        """Draws a pulsing collision warning banner."""
        banner_h = 32
        rect = QtCore.QRect(dx + 20, dy + 15, dw - 40, banner_h)
        painter.setPen(QtGui.QPen(QtGui.QColor("#ff2a3c"), 1.5))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(239, 68, 68, 200)))
        painter.drawRoundedRect(rect, 6, 6)

        painter.setPen(QtGui.QColor("#ffffff"))
        painter.setFont(QtGui.QFont("SF Pro Display", 11, QtGui.QFont.Bold))
        painter.drawText(rect, QtCore.Qt.AlignCenter, "⚠️ ALERTE RISQUE DE COLLISION IMMINENTE")
