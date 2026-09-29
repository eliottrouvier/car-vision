"""
video_hud_widget.py - Real Dashcam Video View with Augmented Reality (AR) HUD Overlay.
Renders drivable area carpet, lane lines, 3D wireframe bounding boxes, crosswalk badges,
and safety alerts.
"""

from typing import Optional, List, Tuple
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from perception_engine import PerceptionResult
from bev_geometry import BEVGeometry


class VideoHUDWidget(QtWidgets.QWidget):
    """
    High-performance QWidget displaying camera video stream with HUD overlays.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #0b0d10; border-radius: 8px;")

        # Options
        self.show_drivable_carpet = True
        self.show_lane_lines = True
        self.show_3d_boxes = True
        self.show_crosswalks = True
        self.show_telemetry = True
        self.show_horizon = True

        # State
        self.current_frame: Optional[np.ndarray] = None
        self.current_result: Optional[PerceptionResult] = None
        self.geom: Optional[BEVGeometry] = None

    def update_frame(self, frame: np.ndarray, result: PerceptionResult, geom: BEVGeometry):
        """Update displayed data and trigger repaint."""
        self.current_frame = frame
        self.current_result = result
        self.geom = geom
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        w_widget = self.width()
        h_widget = self.height()

        if self.current_frame is None:
            # Draw placeholder
            painter.fillRect(0, 0, w_widget, h_widget, QtGui.QColor("#0d1117"))
            painter.setPen(QtGui.QColor("#8b949e"))
            painter.setFont(QtGui.QFont("SF Pro Display", 14, QtGui.QFont.DemiBold))
            painter.drawText(
                self.rect(),
                QtCore.Qt.AlignCenter,
                "WAITING FOR DASHCAM VIDEO FEED..."
            )
            return

        frame = self.current_frame
        fh, fw = frame.shape[:2]

        # Calculate aspect ratio scaling
        scale = min(w_widget / fw, h_widget / fh)
        dw = int(fw * scale)
        dh = int(fh * scale)
        dx = (w_widget - dw) // 2
        dy = (h_widget - dh) // 2

        # 1. Prepare base RGB image with segmentation overlays
        display_img = frame.copy()

        # Drivable road carpet overlay (Emerald green tint)
        if self.show_drivable_carpet and self.current_result is not None:
            da_mask = self.current_result.drivable_mask
            if da_mask is not None and da_mask.max() > 0:
                overlay = display_img.copy()
                # Vibrant turquoise / emerald carpet
                overlay[da_mask > 0] = [180, 230, 40]
                cv2.addWeighted(overlay, 0.30, display_img, 0.70, 0, display_img)

        # Lane lines overlay (Cyan / Neon Blue glow)
        if self.show_lane_lines and self.current_result is not None:
            lane_mask = self.current_result.lane_mask
            if lane_mask is not None and lane_mask.max() > 0:
                display_img[lane_mask > 0] = [255, 235, 0] # Neon cyan in BGR

        # Crosswalk highlight
        if self.show_crosswalks and self.current_result is not None:
            for cw in self.current_result.crosswalks:
                x1, y1, x2, y2 = cw['box']
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 215, 255), 2)

        # Convert to QImage and draw
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * fw
        qimg = QtGui.QImage(rgb_img.data, fw, fh, bytes_per_line, QtGui.QImage.Format_RGB888)
        dest_rect = QtCore.QRect(dx, dy, dw, dh)
        painter.drawImage(dest_rect, qimg)

        # Helper to convert frame coords (u, v) to widget screen coords
        def to_screen(u, v):
            return dx + int(u * scale), dy + int(v * scale)

        # 2. Draw 3D Wireframe Bounding Boxes & Badges
        if self.show_3d_boxes and self.current_result is not None and self.geom is not None:
            for trk in self.current_result.tracks:
                # Color code
                if trk.class_name in ('person', 'bicycle'):
                    edge_color = QtGui.QColor(255, 60, 180, 230) # Neon magenta
                    fill_color = QtGui.QColor(255, 60, 180, 40)
                elif trk.warning_level == 'critical':
                    edge_color = QtGui.QColor(255, 45, 45, 250) # Bright red
                    fill_color = QtGui.QColor(255, 45, 45, 60)
                elif trk.warning_level == 'warning':
                    edge_color = QtGui.QColor(255, 170, 0, 230) # Amber
                    fill_color = QtGui.QColor(255, 170, 0, 40)
                else:
                    edge_color = QtGui.QColor(0, 230, 255, 220) # Cyan
                    fill_color = QtGui.QColor(0, 230, 255, 30)

                # Project 3D cuboid corners to 2D
                pts_2d = self.geom.project_3d_box(trk.corners_3d)

                if pts_2d is not None and len(pts_2d) == 8:
                    screen_pts = [to_screen(u, v) for u, v in pts_2d]

                    # Edges: bottom (0-1-2-3), top (4-5-6-7), pillars (0-4, 1-5, 2-6, 3-7)
                    pen = QtGui.QPen(edge_color, 2.0, QtCore.Qt.SolidLine)
                    painter.setPen(pen)

                    # Top and bottom faces
                    for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0),
                                  (4, 5), (5, 6), (6, 7), (7, 4),
                                  (0, 4), (1, 5), (2, 6), (3, 7)]:
                        p1 = QtCore.QPoint(*screen_pts[i])
                        p2 = QtCore.QPoint(*screen_pts[j])
                        painter.drawLine(p1, p2)

                    # Shaded front face (0-1-5-4)
                    front_poly = QtGui.QPolygon([
                        QtCore.QPoint(*screen_pts[0]),
                        QtCore.QPoint(*screen_pts[1]),
                        QtCore.QPoint(*screen_pts[5]),
                        QtCore.QPoint(*screen_pts[4])
                    ])
                    painter.setBrush(QtGui.QBrush(fill_color))
                    painter.drawPolygon(front_poly)

                    # Floating HUD Badge above top center
                    top_u = (screen_pts[4][0] + screen_pts[5][0]) / 2.0
                    top_v = min(screen_pts[4][1], screen_pts[5][1]) - 12
                else:
                    # Fallback to 2D box if 3D corners projection out of view
                    bx1, by1, bx2, by2 = trk.box_2d
                    sx1, sy1 = to_screen(bx1, by1)
                    sx2, sy2 = to_screen(bx2, by2)
                    pen = QtGui.QPen(edge_color, 2.0)
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawRect(QtCore.QRect(sx1, sy1, sx2 - sx1, sy2 - sy1))
                    top_u = (sx1 + sx2) / 2.0
                    top_v = sy1 - 12

                # Draw Badge text
                badge_text = f"#{trk.track_id} {trk.class_name.upper()} • {trk.Z:.1f}m"
                if trk.ttc is not None:
                    badge_text += f" • TTC {trk.ttc:.1f}s"

                painter.setFont(QtGui.QFont("SF Pro Display", 9, QtGui.QFont.Bold))
                fm = painter.fontMetrics()
                txt_w = fm.horizontalAdvance(badge_text) + 12
                txt_h = fm.height() + 4

                badge_x = int(top_u - txt_w / 2.0)
                badge_y = int(top_v - txt_h)

                # Badge background pill
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(15, 20, 25, 210)))
                painter.drawRoundedRect(badge_x, badge_y, txt_w, txt_h, 4, 4)

                # Accent border on pill
                painter.setPen(QtGui.QPen(edge_color, 1.2))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRoundedRect(badge_x, badge_y, txt_w, txt_h, 4, 4)

                # Badge text
                painter.setPen(QtGui.QColor(245, 245, 250))
                painter.drawText(
                    QtCore.QRect(badge_x, badge_y, txt_w, txt_h),
                    QtCore.Qt.AlignCenter,
                    badge_text
                )

        # 3. Crosswalk Badges
        if self.show_crosswalks and self.current_result is not None:
            for cw in self.current_result.crosswalks:
                x1, y1, x2, y2 = cw['box']
                sx1, sy1 = to_screen(x1, y1)
                sx2, sy2 = to_screen(x2, y2)
                badge_text = f"🚸 CROSSWALK • {cw['Z']:.1f}m"
                painter.setFont(QtGui.QFont("SF Pro Display", 10, QtGui.QFont.Bold))
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(badge_text) + 16
                th = fm.height() + 6
                bx = int((sx1 + sx2) / 2.0 - tw / 2.0)
                by = int(sy1 - th - 5)

                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(230, 160, 0, 220)))
                painter.drawRoundedRect(bx, by, tw, th, 4, 4)
                painter.setPen(QtGui.QColor(10, 10, 15))
                painter.drawText(QtCore.QRect(bx, by, tw, th), QtCore.Qt.AlignCenter, badge_text)

        # 4. Horizon line
        if self.show_horizon and self.current_result is not None:
            hy = self.current_result.horizon_y
            _, shy = to_screen(0, hy)
            if dy <= shy <= dy + dh:
                pen = QtGui.QPen(QtGui.QColor(0, 200, 255, 60), 1.0, QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(dx, shy, dx + dw, shy)

        # 5. Top Alert HUD Banner
        if self.current_result is not None:
            if self.current_result.fcw_alert:
                self._draw_alert_banner(painter, dx, dy, dw, "⚠️ FORWARD COLLISION WARNING - BRAKE!", QtGui.QColor(220, 20, 60))
            elif self.current_result.pedestrian_alert:
                self._draw_alert_banner(painter, dx, dy, dw, "⚠️ PEDESTRIAN HAZARD DETECTED", QtGui.QColor(240, 120, 0))

        # 6. Windshield Glass Tint & HUD Corner Brackets
        self._draw_hud_decorations(painter, dx, dy, dw, dh)

    def _draw_alert_banner(self, painter: QtGui.QPainter, x: int, y: int, w: int, text: str, bg_color: QtGui.QColor):
        banner_h = 36
        banner_w = min(460, w - 40)
        bx = x + (w - banner_w) // 2
        by = y + 16

        painter.setPen(QtCore.Qt.NoPen)
        bg = QtGui.QColor(bg_color)
        bg.setAlpha(220)
        painter.setBrush(QtGui.QBrush(bg))
        painter.drawRoundedRect(bx, by, banner_w, banner_h, 6, 6)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.5))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(bx, by, banner_w, banner_h, 6, 6)

        painter.setFont(QtGui.QFont("SF Pro Display", 11, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(QtCore.QRect(bx, by, banner_w, banner_h), QtCore.Qt.AlignCenter, text)

    def _draw_hud_decorations(self, painter: QtGui.QPainter, x: int, y: int, w: int, h: int):
        """Draw sci-fi / Tesla HUD tactical corner brackets."""
        c_len = 18
        pen = QtGui.QPen(QtGui.QColor(0, 220, 255, 120), 2.0)
        painter.setPen(pen)

        # Top-left
        painter.drawLine(x + 8, y + 8, x + 8 + c_len, y + 8)
        painter.drawLine(x + 8, y + 8, x + 8, y + 8 + c_len)

        # Top-right
        painter.drawLine(x + w - 8, y + 8, x + w - 8 - c_len, y + 8)
        painter.drawLine(x + w - 8, y + 8, x + w - 8, y + 8 + c_len)

        # Bottom-left
        painter.drawLine(x + 8, y + h - 8, x + 8 + c_len, y + h - 8)
        painter.drawLine(x + 8, y + h - 8, x + 8, y + h - 8 - c_len)

        # Bottom-right
        painter.drawLine(x + w - 8, y + h - 8, x + w - 8 - c_len, y + h - 8)
        painter.drawLine(x + w - 8, y + h - 8, x + w - 8, y + h - 8 - c_len)

        # Top-left watermark
        painter.setFont(QtGui.QFont("SF Pro Display", 8, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor(0, 220, 255, 150))
        painter.drawText(x + 16, y + 24, "AR WINDSHIELD HUD • CAM-FRONT")
