"""
main_window.py - Minimalist Cockpit Dashboard for Car-Vision.
Features Apple/Tesla-inspired Dark Glass UI, Dual-Screen View (Windshield HUD + BEV),
2D Tactical Brackets / 3D Cuboids, and Native Apple Silicon M4 MPS Acceleration.
"""

from typing import Optional, List, Dict, Any
import os
import time
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from perception_engine import PanopticPerceptionEngine, PerceptionResult
from video_hud_widget import VideoHUDWidget
from bev_widget import BEVWidget


class VideoProcessingThread(QtCore.QThread):
    """Background worker thread processing frames asynchronously."""
    frame_processed = QtCore.Signal(np.ndarray, object)
    playback_state_changed = QtCore.Signal(bool)
    position_changed = QtCore.Signal(int, int)

    def __init__(self, engine: PanopticPerceptionEngine):
        super().__init__()
        self.engine = engine
        self.video_source: Optional[str] = None
        self.is_running = True
        self.is_paused = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.target_fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.loop_video = True
        self.seek_requested = -1
        self.step_requested = 0

    def open_source(self, source_path_or_idx):
        self.is_paused = True
        if self.cap is not None:
            self.cap.release()

        self.video_source = source_path_or_idx
        self.cap = cv2.VideoCapture(source_path_or_idx)
        if self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.target_fps = fps if (fps and 5.0 <= fps <= 60.0) else 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_idx = 0
            self.is_paused = False
            self.playback_state_changed.emit(True)
        else:
            print(f"[Worker] Could not open video source: {source_path_or_idx}")

    def toggle_play_pause(self):
        self.is_paused = not self.is_paused
        self.playback_state_changed.emit(not self.is_paused)

    def seek_frame(self, frame_idx: int):
        self.seek_requested = frame_idx

    def stop(self):
        self.is_running = False
        self.wait(2000)
        if self.cap is not None:
            self.cap.release()

    def run(self):
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                self.msleep(30)
                continue

            if self.seek_requested >= 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_requested)
                self.current_frame_idx = self.seek_requested
                self.seek_requested = -1

            if self.is_paused:
                self.msleep(30)
                continue

            t_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                if self.loop_video and self.total_frames > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0
                    continue
                else:
                    self.is_paused = True
                    self.playback_state_changed.emit(False)
                    self.msleep(30)
                    continue

            self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Run perception pipeline
            result = self.engine.process_frame(frame, enable_panoptic=True, enable_detection=True)

            self.frame_processed.emit(frame, result)
            self.position_changed.emit(self.current_frame_idx, self.total_frames)

            # Match native video frame rate
            proc_time = time.time() - t_start
            desired_frame_time = 1.0 / self.target_fps
            sleep_time = max(0.001, desired_frame_time - proc_time)
            self.msleep(int(sleep_time * 1000))


class MainWindow(QtWidgets.QMainWindow):
    """
    Sleek Minimalist Autonomous Driving Dashboard.
    """

    def __init__(self, sample_dir: str = "samples"):
        super().__init__()
        self.setWindowTitle("Car-Vision")
        self.resize(1380, 820)
        self.sample_dir = sample_dir

        self.engine = PanopticPerceptionEngine()

        self.worker = VideoProcessingThread(self.engine)
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.playback_state_changed.connect(self._on_playback_state_changed)
        self.worker.position_changed.connect(self._on_position_changed)

        self._build_ui()
        self._apply_minimal_theme()

        self.worker.start()

        # Load default city driving video
        default_video = os.path.join(self.sample_dir, "city_paris.mp4")
        if not os.path.exists(default_video):
            default_video = os.path.join(self.sample_dir, "highway.mp4")
        if os.path.exists(default_video):
            self.worker.open_source(default_video)

    def _build_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(8)

        # 1. Barre supérieure ultra-fine
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)

        # 2. Séparateur double écran épuré
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(4)

        # Écran Gauche (Pare-brise HUD)
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.video_hud = VideoHUDWidget()
        left_header = self._create_left_header()
        left_layout.addWidget(left_header)
        left_layout.addWidget(self.video_hud, stretch=1)

        # Écran Droit (Tesla BEV)
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.bev_view = BEVWidget()
        right_header = self._create_right_header()
        right_layout.addWidget(right_header)
        right_layout.addWidget(self.bev_view, stretch=1)

        self.splitter.addWidget(left_container)
        self.splitter.addWidget(right_container)
        self.splitter.setSizes([690, 690])
        main_layout.addWidget(self.splitter, stretch=1)

        # 3. Barre inférieure minimaliste
        bottom_bar = self._create_bottom_bar()
        main_layout.addWidget(bottom_bar)

    def _create_top_bar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("background-color: #0c0f16; border-radius: 6px; padding: 2px;")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(10, 4, 10, 4)

        # Titre minimaliste & statut M4
        title = QtWidgets.QLabel("CAR-VISION")
        title.setStyleSheet("font-family: 'SF Pro Display'; font-size: 13px; font-weight: 800; color: #ffffff; letter-spacing: 1.5px;")
        layout.addWidget(title)

        status_dot = QtWidgets.QLabel("● Apple M4")
        status_dot.setStyleSheet("font-size: 10px; font-weight: 600; color: #10b981; margin-left: 6px;")
        layout.addWidget(status_dot)

        layout.addSpacing(16)

        # Télémétrie discrète en texte brut
        self.lbl_telemetry = QtWidgets.QLabel("0 FPS  •  0 ms  •  0 cibles")
        self.lbl_telemetry.setStyleSheet("font-size: 11px; font-weight: 500; color: #94a3b8; font-family: -apple-system, sans-serif;")
        layout.addWidget(self.lbl_telemetry)

        layout.addStretch()

        # Sélecteur de source compact
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.setStyleSheet("""
            QComboBox {
                background-color: #161c28;
                color: #e2e8f0;
                border: 1px solid #283346;
                border-radius: 5px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #161c28;
                selection-background-color: #0284c7;
                color: #e2e8f0;
            }
        """)
        self._populate_sample_videos()
        self.source_combo.currentIndexChanged.connect(self._on_sample_changed)
        layout.addWidget(self.source_combo)

        btn_open = QtWidgets.QPushButton("Ouvrir...")
        btn_open.setStyleSheet(self._btn_style())
        btn_open.clicked.connect(self._on_open_file)
        layout.addWidget(btn_open)

        btn_webcam = QtWidgets.QPushButton("Webcam")
        btn_webcam.setStyleSheet(self._btn_style())
        btn_webcam.clicked.connect(lambda: self.worker.open_source(0))
        layout.addWidget(btn_webcam)

        return frame

    def _create_left_header(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(4, 2, 4, 2)

        lbl = QtWidgets.QLabel("PARE-BRISE HUD")
        lbl.setStyleSheet("font-size: 10px; font-weight: 700; color: #64748b; letter-spacing: 1px;")
        layout.addWidget(lbl)
        layout.addStretch()

        # Bascule 2D / 3D
        self.btn_box_toggle = QtWidgets.QPushButton("Viseur 2D")
        self.btn_box_toggle.setStyleSheet(self._btn_style(active=True))
        self.btn_box_toggle.clicked.connect(self._toggle_box_mode)
        layout.addWidget(self.btn_box_toggle)

        chk_carpet = QtWidgets.QCheckBox("Tapis")
        chk_carpet.setChecked(True)
        chk_carpet.setStyleSheet("color: #94a3b8; font-size: 10px;")
        chk_carpet.toggled.connect(lambda c: setattr(self.video_hud, 'show_drivable_carpet', c))
        layout.addWidget(chk_carpet)

        chk_lanes = QtWidgets.QCheckBox("Lignes")
        chk_lanes.setChecked(True)
        chk_lanes.setStyleSheet("color: #94a3b8; font-size: 10px;")
        chk_lanes.toggled.connect(lambda c: setattr(self.video_hud, 'show_lane_lines', c))
        layout.addWidget(chk_lanes)

        return w

    def _toggle_box_mode(self):
        if self.video_hud.box_mode == "2D_BRACKETS":
            self.video_hud.set_box_mode("3D_CUBOID")
            self.btn_box_toggle.setText("Cuboïdes 3D")
        else:
            self.video_hud.set_box_mode("2D_BRACKETS")
            self.btn_box_toggle.setText("Viseur 2D")

    def _create_right_header(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(4, 2, 4, 2)

        lbl = QtWidgets.QLabel("RADAR 3D BEV")
        lbl.setStyleSheet("font-size: 10px; font-weight: 700; color: #64748b; letter-spacing: 1px;")
        layout.addWidget(lbl)
        layout.addStretch()

        self.btn_bev_mode = QtWidgets.QPushButton("3D FSD")
        self.btn_bev_mode.setStyleSheet(self._btn_style(active=True))
        self.btn_bev_mode.clicked.connect(self._toggle_bev_mode)
        layout.addWidget(self.btn_bev_mode)

        btn_reset = QtWidgets.QPushButton("Recentrer")
        btn_reset.setStyleSheet(self._btn_style())
        btn_reset.clicked.connect(self.bev_view.reset_view)
        layout.addWidget(btn_reset)

        return w

    def _toggle_bev_mode(self):
        if self.bev_view.view_mode == "3D_FSD":
            self.bev_view.set_view_mode("TOP_DOWN")
            self.btn_bev_mode.setText("Top-Down")
        else:
            self.bev_view.set_view_mode("3D_FSD")
            self.btn_bev_mode.setText("3D FSD")

    def _create_bottom_bar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("background-color: #0c0f16; border-radius: 6px; padding: 2px;")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(12)

        self.btn_play = QtWidgets.QPushButton("⏸")
        self.btn_play.setFixedWidth(32)
        self.btn_play.setStyleSheet(self._btn_style(active=True))
        self.btn_play.clicked.connect(self.worker.toggle_play_pause)
        layout.addWidget(self.btn_play)

        # Timeline fine
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #1e2638;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #00e6ff;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 10px;
                margin-top: -3px;
                margin-bottom: -3px;
                border-radius: 5px;
            }
        """)
        self.slider.sliderMoved.connect(self.worker.seek_frame)
        layout.addWidget(self.slider)

        self.lbl_time = QtWidgets.QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-size: 10px; font-weight: 500; color: #64748b; font-family: monospace;")
        layout.addWidget(self.lbl_time)

        # Statut compact du véhicule suivi
        self.lbl_status = QtWidgets.QLabel("Voie dégagée")
        self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 600; color: #10b981;")
        layout.addWidget(self.lbl_status)

        return frame

    def _btn_style(self, active: bool = False) -> str:
        bg = "#1e293b" if not active else "#0284c7"
        fg = "#cbd5e1" if not active else "#ffffff"
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #334155;
            }}
        """

    def _populate_sample_videos(self):
        self.source_combo.clear()
        if os.path.exists(self.sample_dir):
            files = [f for f in os.listdir(self.sample_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
            # Prioritize city_paris.mp4 as first item
            ordered = []
            if "city_paris.mp4" in files:
                ordered.append("city_paris.mp4")
            for f in sorted(files):
                if f not in ordered:
                    ordered.append(f)

            for f in ordered:
                label = f
                if f == "city_paris.mp4":
                    label = "Ville (Paris • Voitures & Piétons)"
                elif f == "highway.mp4":
                    label = "Autoroute (Trafic rapide)"
                elif f == "person-bicycle-car-detection.mp4":
                    label = "Piétons & Cyclistes"
                elif f == "car-detection.mp4":
                    label = "Circulation urbaine"
                self.source_combo.addItem(label, os.path.join(self.sample_dir, f))

    def _on_sample_changed(self, idx: int):
        path = self.source_combo.itemData(idx)
        if path and os.path.exists(path):
            self.worker.open_source(path)

    def _on_open_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choisir une vidéo",
            "",
            "Vidéos (*.mp4 *.mov *.avi *.mkv)"
        )
        if file_path:
            self.worker.open_source(file_path)

    def _on_frame_processed(self, frame: np.ndarray, result: PerceptionResult):
        self.video_hud.update_frame(frame, result, self.engine.geom)
        self.bev_view.update_result(result)

        fps = 1000.0 / max(result.inference_time_ms, 1.0)
        self.lbl_telemetry.setText(f"{fps:.0f} FPS  •  {result.inference_time_ms:.0f} ms  •  {len(result.tracks)} cibles")

        # Statut épuré
        if result.fcw_alert:
            self.lbl_status.setText("⚠️ RISQUE DE COLLISION")
            self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 700; color: #ef4444;")
        elif result.pedestrian_alert:
            self.lbl_status.setText("⚠️ PIÉTON PROCHE")
            self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 700; color: #f59e0b;")
        elif result.lead_vehicle:
            lv = result.lead_vehicle
            self.lbl_status.setText(f"Véhicule devant : {lv.Z:.1f}m")
            self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 500; color: #38bdf8;")
        else:
            self.lbl_status.setText("Voie libre")
            self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 500; color: #10b981;")

    def _on_playback_state_changed(self, is_playing: bool):
        self.btn_play.setText("⏸" if is_playing else "▶")

    def _on_position_changed(self, current: int, total: int):
        if total > 0:
            self.slider.setMaximum(total)
            if not self.slider.isSliderDown():
                self.slider.setValue(current)

            cur_sec = int(current / max(self.worker.target_fps, 1.0))
            tot_sec = int(total / max(self.worker.target_fps, 1.0))
            self.lbl_time.setText(f"{cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}")

    def _apply_minimal_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #06080c;
            }
            QSplitter::handle {
                background-color: #121824;
            }
            QSplitter::handle:hover {
                background-color: #00e6ff;
            }
        """)

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.worker.stop()
        event.accept()
