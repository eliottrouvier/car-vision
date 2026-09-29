"""
main_window.py - Main Cockpit Dashboard for Car-Vision Autonomous Perception.
Features Dual-Screen View (Windshield AR HUD + Tesla FSD 3D BEV), Telemetry Gauges,
Video Transport Controls, and Apple Silicon M4 MPS Acceleration.
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
    """
    Background worker thread decoding video frames and executing the
    unified panoptic perception pipeline asynchronously.
    """
    frame_processed = QtCore.Signal(np.ndarray, object)
    playback_state_changed = QtCore.Signal(bool)
    position_changed = QtCore.Signal(int, int) # current_frame, total_frames

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

        # Feature toggles
        self.enable_panoptic = True
        self.enable_detection = True

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

    def step_frame(self, delta: int):
        self.step_requested = delta

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

            # Handle seek
            if self.seek_requested >= 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_requested)
                self.current_frame_idx = self.seek_requested
                self.seek_requested = -1

            # Handle step
            if self.is_paused and self.step_requested == 0:
                self.msleep(30)
                continue

            if self.step_requested != 0:
                target = max(0, self.current_frame_idx + self.step_requested)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                self.current_frame_idx = target
                self.step_requested = 0

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
            result = self.engine.process_frame(
                frame,
                enable_panoptic=self.enable_panoptic,
                enable_detection=self.enable_detection
            )

            # Emit results
            self.frame_processed.emit(frame, result)
            self.position_changed.emit(self.current_frame_idx, self.total_frames)

            # Frame rate throttling
            proc_time = time.time() - t_start
            desired_frame_time = 1.0 / self.target_fps
            sleep_time = max(0.001, desired_frame_time - proc_time)
            self.msleep(int(sleep_time * 1000))


class MainWindow(QtWidgets.QMainWindow):
    """
    Main Cockpit Window integrating Windshield AR HUD, Tesla BEV, and Telemetry.
    """

    def __init__(self, sample_dir: str = "samples"):
        super().__init__()
        self.setWindowTitle("Car-Vision • Autonomous Panoptic Perception Cockpit")
        self.resize(1440, 880)
        self.sample_dir = sample_dir

        # Initialize perception engine
        self.engine = PanopticPerceptionEngine()

        # Initialize processing thread
        self.worker = VideoProcessingThread(self.engine)
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.playback_state_changed.connect(self._on_playback_state_changed)
        self.worker.position_changed.connect(self._on_position_changed)

        # Build UI
        self._build_ui()
        self._apply_dark_theme()

        # Start thread
        self.worker.start()

        # Auto-load initial highway sample if present
        default_video = os.path.join(self.sample_dir, "highway.mp4")
        if os.path.exists(default_video):
            self.worker.open_source(default_video)

    def _build_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # 1. Top Header Bar
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)

        # 2. Central Dual-View Splitter
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(8)

        # Left Panel (AR Dashcam HUD)
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.video_hud = VideoHUDWidget()
        left_header = self._create_left_header()
        left_layout.addWidget(left_header)
        left_layout.addWidget(self.video_hud)

        # Right Panel (Tesla 3D BEV)
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.bev_view = BEVWidget()
        right_header = self._create_right_header()
        right_layout.addWidget(right_header)
        right_layout.addWidget(self.bev_view)

        self.splitter.addWidget(left_container)
        self.splitter.addWidget(right_container)
        self.splitter.setSizes([720, 720])
        main_layout.addWidget(self.splitter, stretch=1)

        # 3. Bottom Telemetry & Transport Controls
        bottom_bar = self._create_bottom_bar()
        main_layout.addWidget(bottom_bar)

    def _create_top_bar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("background-color: #12161f; border-radius: 8px; padding: 6px;")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 4, 12, 4)

        # Logo & Subtitle
        logo_layout = QtWidgets.QVBoxLayout()
        logo_layout.setSpacing(1)
        title = QtWidgets.QLabel("CAR-VISION")
        title.setStyleSheet("font-family: 'SF Pro Display'; font-size: 16px; font-weight: 800; color: #00e6ff; letter-spacing: 1px;")
        sub = QtWidgets.QLabel("AUTONOMOUS PANOPTIC PERCEPTION • M4 MPS")
        sub.setStyleSheet("font-family: 'SF Pro Display'; font-size: 9px; font-weight: 600; color: #8b9bb4;")
        logo_layout.addWidget(title)
        logo_layout.addWidget(sub)
        layout.addLayout(logo_layout)

        layout.addSpacing(20)

        # Telemetry Badges
        self.badge_fps = self._create_badge("FPS", "0.0")
        self.badge_latency = self._create_badge("LATENCY", "0 ms")
        self.badge_targets = self._create_badge("TARGETS", "0")
        self.badge_fcw = self._create_badge("COLLISION RISK", "NORMAL", color="#00e676")

        layout.addWidget(self.badge_fps)
        layout.addWidget(self.badge_latency)
        layout.addWidget(self.badge_targets)
        layout.addWidget(self.badge_fcw)

        layout.addStretch()

        # Video Source Controls
        src_label = QtWidgets.QLabel("SOURCE:")
        src_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #8b9bb4;")
        layout.addWidget(src_label)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a2230;
                color: #e6edf3;
                border: 1px solid #2d3748;
                border-radius: 6px;
                padding: 5px 12px;
                font-size: 11px;
                font-weight: 600;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1a2230;
                selection-background-color: #007799;
                color: #e6edf3;
            }
        """)
        self._populate_sample_videos()
        self.source_combo.currentIndexChanged.connect(self._on_sample_changed)
        layout.addWidget(self.source_combo)

        btn_open = QtWidgets.QPushButton("📁 Open File...")
        btn_open.setStyleSheet(self._btn_style("#1e293b", "#00e6ff"))
        btn_open.clicked.connect(self._on_open_file)
        layout.addWidget(btn_open)

        btn_webcam = QtWidgets.QPushButton("📹 Live Cam")
        btn_webcam.setStyleSheet(self._btn_style("#1e293b", "#00e676"))
        btn_webcam.clicked.connect(self._on_open_webcam)
        layout.addWidget(btn_webcam)

        return frame

    def _create_badge(self, label: str, value: str, color: str = "#00e6ff") -> QtWidgets.QFrame:
        f = QtWidgets.QFrame()
        f.setStyleSheet("background-color: #161c27; border: 1px solid #252e3d; border-radius: 6px; padding: 3px 10px;")
        l = QtWidgets.QVBoxLayout(f)
        l.setContentsMargins(4, 2, 4, 2)
        l.setSpacing(0)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("font-size: 8px; font-weight: bold; color: #73849c;")
        val = QtWidgets.QLabel(value)
        val.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {color}; font-family: monospace;")
        val.setObjectName("badge_val")
        l.addWidget(lbl)
        l.addWidget(val)
        return f

    def _update_badge(self, badge_frame: QtWidgets.QFrame, text: str, color: Optional[str] = None):
        val_widget = badge_frame.findChild(QtWidgets.QLabel, "badge_val")
        if val_widget:
            val_widget.setText(text)
            if color:
                val_widget.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {color}; font-family: monospace;")

    def _create_left_header(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(4, 0, 4, 0)

        lbl = QtWidgets.QLabel("WINDSHIELD AR HUD (PANOPTIC VIEW)")
        lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #00e6ff; letter-spacing: 0.5px;")
        layout.addWidget(lbl)
        layout.addStretch()

        # Toggles
        chk_carpet = QtWidgets.QCheckBox("Road Carpet")
        chk_carpet.setChecked(True)
        chk_carpet.setStyleSheet("color: #cbd5e1; font-size: 10px;")
        chk_carpet.toggled.connect(lambda c: setattr(self.video_hud, 'show_drivable_carpet', c))
        layout.addWidget(chk_carpet)

        chk_lanes = QtWidgets.QCheckBox("Lanes")
        chk_lanes.setChecked(True)
        chk_lanes.setStyleSheet("color: #cbd5e1; font-size: 10px;")
        chk_lanes.toggled.connect(lambda c: setattr(self.video_hud, 'show_lane_lines', c))
        layout.addWidget(chk_lanes)

        chk_boxes = QtWidgets.QCheckBox("3D Cuboids")
        chk_boxes.setChecked(True)
        chk_boxes.setStyleSheet("color: #cbd5e1; font-size: 10px;")
        chk_boxes.toggled.connect(lambda c: setattr(self.video_hud, 'show_3d_boxes', c))
        layout.addWidget(chk_boxes)

        chk_cross = QtWidgets.QCheckBox("Crosswalks")
        chk_cross.setChecked(True)
        chk_cross.setStyleSheet("color: #cbd5e1; font-size: 10px;")
        chk_cross.toggled.connect(lambda c: setattr(self.video_hud, 'show_crosswalks', c))
        layout.addWidget(chk_cross)

        return w

    def _create_right_header(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(4, 0, 4, 0)

        lbl = QtWidgets.QLabel("TESLA FSD 3D BEV (VECTOR SPACE)")
        lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #00e6ff; letter-spacing: 0.5px;")
        layout.addWidget(lbl)
        layout.addStretch()

        # Mode Buttons
        self.btn_fsd_mode = QtWidgets.QPushButton("3D FSD View")
        self.btn_fsd_mode.setStyleSheet(self._btn_style("#005577", "#ffffff"))
        self.btn_fsd_mode.clicked.connect(lambda: self._set_bev_mode("3D_FSD"))
        layout.addWidget(self.btn_fsd_mode)

        self.btn_top_mode = QtWidgets.QPushButton("Top-Down")
        self.btn_top_mode.setStyleSheet(self._btn_style("#1e293b", "#8b9bb4"))
        self.btn_top_mode.clicked.connect(lambda: self._set_bev_mode("TOP_DOWN"))
        layout.addWidget(self.btn_top_mode)

        btn_reset = QtWidgets.QPushButton("↺ Reset")
        btn_reset.setStyleSheet(self._btn_style("#1e293b", "#cbd5e1"))
        btn_reset.clicked.connect(self.bev_view.reset_view)
        layout.addWidget(btn_reset)

        return w

    def _set_bev_mode(self, mode: str):
        self.bev_view.set_view_mode(mode)
        if mode == "3D_FSD":
            self.btn_fsd_mode.setStyleSheet(self._btn_style("#005577", "#ffffff"))
            self.btn_top_mode.setStyleSheet(self._btn_style("#1e293b", "#8b9bb4"))
        else:
            self.btn_fsd_mode.setStyleSheet(self._btn_style("#1e293b", "#8b9bb4"))
            self.btn_top_mode.setStyleSheet(self._btn_style("#005577", "#ffffff"))

    def _create_bottom_bar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("background-color: #12161f; border-radius: 8px; padding: 6px;")
        v_layout = QtWidgets.QVBoxLayout(frame)
        v_layout.setContentsMargins(12, 6, 12, 6)
        v_layout.setSpacing(6)

        # 1. Timeline & Transport Bar
        transport_layout = QtWidgets.QHBoxLayout()

        self.btn_play = QtWidgets.QPushButton("⏸ Pause")
        self.btn_play.setStyleSheet(self._btn_style("#005577", "#ffffff", bold=True))
        self.btn_play.clicked.connect(self.worker.toggle_play_pause)
        transport_layout.addWidget(self.btn_play)

        btn_prev = QtWidgets.QPushButton("⏮")
        btn_prev.setFixedWidth(36)
        btn_prev.setStyleSheet(self._btn_style("#1e293b", "#cbd5e1"))
        btn_prev.clicked.connect(lambda: self.worker.step_frame(-15))
        transport_layout.addWidget(btn_prev)

        btn_next = QtWidgets.QPushButton("⏭")
        btn_next.setFixedWidth(36)
        btn_next.setStyleSheet(self._btn_style("#1e293b", "#cbd5e1"))
        btn_next.clicked.connect(lambda: self.worker.step_frame(15))
        transport_layout.addWidget(btn_next)

        # Timeline Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00e6ff;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        self.slider.sliderMoved.connect(self.worker.seek_frame)
        transport_layout.addWidget(self.slider)

        self.lbl_time = QtWidgets.QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-size: 11px; font-weight: bold; color: #8b9bb4; font-family: monospace;")
        transport_layout.addWidget(self.lbl_time)

        v_layout.addLayout(transport_layout)

        # 2. Telemetry Gauges Ribbon
        telemetry_layout = QtWidgets.QHBoxLayout()
        telemetry_layout.setSpacing(16)

        self.lbl_lead = QtWidgets.QLabel("LEAD VEHICLE: NONE")
        self.lbl_lead.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8;")
        telemetry_layout.addWidget(self.lbl_lead)

        self.lbl_ttc = QtWidgets.QLabel("TTC: --")
        self.lbl_ttc.setStyleSheet("font-size: 11px; font-weight: bold; color: #00e676;")
        telemetry_layout.addWidget(self.lbl_ttc)

        self.lbl_crosswalk_stat = QtWidgets.QLabel("CROSSWALK: NONE")
        self.lbl_crosswalk_stat.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8;")
        telemetry_layout.addWidget(self.lbl_crosswalk_stat)

        telemetry_layout.addStretch()

        # Engine mode status
        eng_label = QtWidgets.QLabel("ENGINE: YOLOPv2 + YOLO11 (MPS ACCELERATED)")
        eng_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #38bdf8;")
        telemetry_layout.addWidget(eng_label)

        v_layout.addLayout(telemetry_layout)

        return frame

    def _btn_style(self, bg: str, fg: str, bold: bool = False) -> str:
        fw = "bold" if bold else "600"
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: {fw};
            }}
            QPushButton:hover {{
                background-color: #2e3d55;
                border-color: #00e6ff;
            }}
            QPushButton:pressed {{
                background-color: #007799;
            }}
        """

    def _populate_sample_videos(self):
        self.source_combo.clear()
        if os.path.exists(self.sample_dir):
            files = [f for f in os.listdir(self.sample_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
            files.sort()
            for f in files:
                self.source_combo.addItem(f, os.path.join(self.sample_dir, f))

    def _on_sample_changed(self, idx: int):
        path = self.source_combo.itemData(idx)
        if path and os.path.exists(path):
            self.worker.open_source(path)

    def _on_open_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Video Clip",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv)"
        )
        if file_path:
            self.worker.open_source(file_path)

    def _on_open_webcam(self):
        self.worker.open_source(0)

    def _on_frame_processed(self, frame: np.ndarray, result: PerceptionResult):
        # Update video HUD
        self.video_hud.update_frame(frame, result, self.engine.geom)

        # Update BEV
        self.bev_view.update_result(result)

        # Update Top Telemetry Badges
        fps = 1000.0 / max(result.inference_time_ms, 1.0)
        self._update_badge(self.badge_fps, f"{fps:.1f}")
        self._update_badge(self.badge_latency, f"{result.inference_time_ms:.0f} ms")
        self._update_badge(self.badge_targets, str(len(result.tracks)))

        # FCW Badge
        if result.fcw_alert:
            self._update_badge(self.badge_fcw, "COLLISION ALERT!", color="#ff2d2d")
        elif result.pedestrian_alert:
            self._update_badge(self.badge_fcw, "PEDESTRIAN HAZARD", color="#ff9800")
        else:
            self._update_badge(self.badge_fcw, "CLEAR PATH", color="#00e676")

        # Bottom Gauges
        if result.lead_vehicle:
            lv = result.lead_vehicle
            self.lbl_lead.setText(f"LEAD VEHICLE: {lv.class_name.upper()} @ {lv.Z:.1f}m ({lv.speed_kmh:.0f} km/h)")
            if lv.ttc is not None:
                color = "#ff2d2d" if lv.ttc < 2.5 else "#ffb703"
                self.lbl_ttc.setText(f"TTC: {lv.ttc:.1f}s")
                self.lbl_ttc.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {color};")
            else:
                self.lbl_ttc.setText("TTC: SAFE")
                self.lbl_ttc.setStyleSheet("font-size: 11px; font-weight: bold; color: #00e676;")
        else:
            self.lbl_lead.setText("LEAD VEHICLE: NONE")
            self.lbl_ttc.setText("TTC: --")
            self.lbl_ttc.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8;")

        # Crosswalk status
        if result.crosswalks:
            cw = result.crosswalks[0]
            self.lbl_crosswalk_stat.setText(f"CROSSWALK: {cw['Z']:.1f}m ({cw['stripes']} STRIPES)")
            self.lbl_crosswalk_stat.setStyleSheet("font-size: 11px; font-weight: bold; color: #ffb703;")
        else:
            self.lbl_crosswalk_stat.setText("CROSSWALK: NONE")
            self.lbl_crosswalk_stat.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8;")

    def _on_playback_state_changed(self, is_playing: bool):
        if is_playing:
            self.btn_play.setText("⏸ Pause")
            self.btn_play.setStyleSheet(self._btn_style("#005577", "#ffffff", bold=True))
        else:
            self.btn_play.setText("▶ Play")
            self.btn_play.setStyleSheet(self._btn_style("#008855", "#ffffff", bold=True))

    def _on_position_changed(self, current: int, total: int):
        if total > 0:
            self.slider.setMaximum(total)
            if not self.slider.isSliderDown():
                self.slider.setValue(current)

            cur_sec = int(current / max(self.worker.target_fps, 1.0))
            tot_sec = int(total / max(self.worker.target_fps, 1.0))
            self.lbl_time.setText(f"{cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}")

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #090c10;
            }
            QSplitter::handle {
                background-color: #1e2638;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background-color: #00e6ff;
            }
            QToolTip {
                background-color: #161c27;
                color: #f0f6fc;
                border: 1px solid #30363d;
                padding: 4px;
            }
        """)

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.worker.stop()
        event.accept()
