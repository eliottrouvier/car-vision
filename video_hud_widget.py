"""
video_hud_widget.py - Real Dashcam Video View with Minimalist Augmented Reality (AR) HUD.
Renders clean tactical corner brackets, 3D wireframe boxes, drivable road carpet,
and pedestrian/vehicle distance badges.
"""

from typing import Optional, List, Tuple
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
    'truck': 'Camion'
}


class VideoHUDWidget(QtWidgets.QWidget):
    """
    High-performance QWidget displaying camera video stream with a sleek,
    minimalist HUD overlay (Tesla / Mobileye inspired).
    """

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

    def update_frame(self, frame: np.ndarray, result: PerceptionResult, geom: BEVGeometry):
        self.current_frame = frame
        self.current_result = result
        self.geom = geom
        self.update()

    def set_box_mode(self, mode: str):
        if mode in ("2D_BRACKETS", "3D_CUBOID"):
            self.box_mode = mode
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        w_widget = self.width()
        h_widget = self.height()

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

        # 1. Tapis de route (Emerald tint subtil et transparent)
        if self.show_drivable_carpet and self.current_result is not None:
            da_mask = self.current_result.drivable_mask
            if da_mask is not None and da_mask.max() > 0:
                overlay = display_img.copy()
                overlay[da_mask > 0] = [170, 225, 50]
                cv2.addWeighted(overlay, 0.22, display_img, 0.78, 0, display_img)

        # 2. Lignes de voies (Bleu cyan subtil)
        if self.show_lane_lines and self.current_result is not None:
            lane_mask = self.current_result.lane_mask
            if lane_mask is not None and lane_mask.max() > 0:
                display_img[lane_mask > 0] = [255, 230, 0]

        # 3. Passages piétons
        if self.show_crosswalks and self.current_result is not None:
            for cw in self.current_result.crosswalks:
                x1, y1, x2, y2 = cw['box']
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # Dessin de l'image
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * fw
        qimg = QtGui.QImage(rgb_img.data, fw, fh, bytes_per_line, QtGui.QImage.Format_RGB888)
        dest_rect = QtCore.QRect(dx, dy, dw, dh)
        painter.drawImage(dest_rect, qimg)

        def to_screen(u, v):
            return dx + int(u * scale), dy + int(v * scale)

        # 4. Boîtes de détection épurées (2D Brackets ou 3D)
        if self.show_boxes and self.current_result is not None:
            for trk in self.current_result.tracks:
                class_label = FRENCH_CLASSES.get(trk.class_name, trk.class_name.capitalize())

                # Palette de couleurs minimaliste
                if trk.warning_level == 'critical':
                    accent_color = QtGui.QColor(255, 42, 60) # Rouge alerte
                    fill_color = QtGui.QColor(255, 42, 60, 40)
                elif trk.class_name == 'person':
                    accent_color = QtGui.QColor(255, 65, 120) # Rose corail
                    fill_color = QtGui.QColor(255, 65, 120, 30)
                elif trk.class_name in ('bicycle', 'motorcycle'):
                    accent_color = QtGui.QColor(255, 185, 20) # Ambre
                    fill_color = QtGui.QColor(255, 185, 20, 30)
                else:
                    accent_color = QtGui.QColor(0, 230, 255) # Cyan électrique
                    fill_color = QtGui.QColor(0, 230, 255, 25)

                if self.box_mode == "3D_CUBOID" and self.geom is not None:
                    # Rendu 3D Cuboïde filaire épuré
                    pts_2d = self.geom.project_3d_box(trk.corners_3d)
                    if pts_2d is not None and len(pts_2d) == 8:
                        screen_pts = [to_screen(u, v) for u, v in pts_2d]
                        pen = QtGui.QPen(accent_color, 1.6)
                        painter.setPen(pen)

                        for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0),
                                      (4, 5), (5, 6), (6, 7), (7, 4),
                                      (0, 4), (1, 5), (2, 6), (3, 7)]:
                            painter.drawLine(QtCore.QPoint(*screen_pts[i]), QtCore.QPoint(*screen_pts[j]))

                        top_u = (screen_pts[4][0] + screen_pts[5][0]) / 2.0
                        top_v = min(screen_pts[4][1], screen_pts[5][1])
                        self._draw_label_pill(painter, top_u, top_v - 8, class_label, trk.Z, trk.ttc, accent_color)
                        continue

                # Rendu 2D Viseur / Tactical Brackets (Par défaut, ultra-épuré)
                bx1, by1, bx2, by2 = trk.box_2d
                sx1, sy1 = to_screen(bx1, by1)
                sx2, sy2 = to_screen(bx2, by2)
                bw = sx2 - sx1
                bh = sy2 - sy1

                if bw > 8 and bh > 8:
                    # Remplissage très subtil
                    painter.fillRect(QtCore.QRect(sx1, sy1, bw, bh), QtGui.QBrush(fill_color))

                    # 4 coins en L fins
                    c_len = min(14, max(5, bw // 4), max(5, bh // 4))
                    pen = QtGui.QPen(accent_color, 2.0, QtCore.Qt.SolidLine, QtCore.Qt.SquareCap)
                    painter.setPen(pen)

                    # Haut-gauche
                    painter.drawLine(sx1, sy1, sx1 + c_len, sy1)
                    painter.drawLine(sx1, sy1, sx1, sy1 + c_len)

                    # Haut-droite
                    painter.drawLine(sx2, sy1, sx2 - c_len, sy1)
                    painter.drawLine(sx2, sy1, sx2, sy1 + c_len)

                    # Bas-gauche
                    painter.drawLine(sx1, sy2, sx1 + c_len, sy2)
                    painter.drawLine(sx1, sy2, sx1, sy2 - c_len)

                    # Bas-droite
                    painter.drawLine(sx2, sy2, sx2 - c_len, sy2)
                    painter.drawLine(sx2, sy2, sx2, sy2 - c_len)

                    # Petite étiquette discrète au-dessus
                    mid_u = (sx1 + sx2) / 2.0
                    self._draw_label_pill(painter, mid_u, sy1 - 6, class_label, trk.Z, trk.ttc, accent_color)

        # 5. Horizon subtil si activé
        if self.show_horizon and self.current_result is not None:
            hy = self.current_result.horizon_y
            _, shy = to_screen(0, hy)
            if dy <= shy <= dy + dh:
                pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 40), 1.0, QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(dx, shy, dx + dw, shy)

        # 6. Alerte anticollision si active
        if self.current_result is not None and self.current_result.fcw_alert:
            self._draw_alert_banner(painter, dx, dy, dw, "⚠️ RISQUE DE COLLISION IMMINENT", QtGui.QColor(230, 30, 50))
        elif self.current_result is not None and self.current_result.pedestrian_alert:
            self._draw_alert_banner(painter, dx, dy, dw, "⚠️ ATTENTION PIÉTON SUR LA VOIE", QtGui.QColor(255, 120, 20))

    def _draw_label_pill(
        self,
        painter: QtGui.QPainter,
        center_x: float,
        top_y: float,
        label: str,
        dist_z: float,
        ttc: Optional[float],
        accent_color: QtGui.QColor
    ):
        """Dessine une étiquette discrète et élégante (ex: 'Piéton 4.2m')."""
        text = f"{label} {dist_z:.1f}m"
        if ttc is not None and ttc < 3.0:
            text += f" • {ttc:.1f}s"

        painter.setFont(QtGui.QFont("SF Pro Display", 8, QtGui.QFont.DemiBold))
        fm = painter.fontMetrics()
        pw = fm.horizontalAdvance(text) + 10
        ph = fm.height() + 2

        px = int(center_x - pw / 2.0)
        py = int(top_y - ph)

        # Fond sombre discret
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(10, 14, 20, 210)))
        painter.drawRoundedRect(px, py, pw, ph, 3, 3)

        # Bordure fine
        painter.setPen(QtGui.QPen(accent_color, 1.0))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(px, py, pw, ph, 3, 3)

        # Texte
        painter.setPen(QtGui.QColor(240, 245, 250))
        painter.drawText(QtCore.QRect(px, py, pw, ph), QtCore.Qt.AlignCenter, text)

    def _draw_alert_banner(self, painter: QtGui.QPainter, x: int, y: int, w: int, text: str, color: QtGui.QColor):
        bh = 32
        bw = min(360, w - 40)
        bx = x + (w - bw) // 2
        by = y + 14

        painter.setPen(QtCore.Qt.NoPen)
        bg = QtGui.QColor(color)
        bg.setAlpha(200)
        painter.setBrush(QtGui.QBrush(bg))
        painter.drawRoundedRect(bx, by, bw, bh, 5, 5)

        painter.setFont(QtGui.QFont("SF Pro Display", 10, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(QtCore.QRect(bx, by, bw, bh), QtCore.Qt.AlignCenter, text)
