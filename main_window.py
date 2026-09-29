"""
main_window.py - Sleek Minimalist Cockpit Dashboard for Car-Vision.
Features:
- Apple/Tesla-inspired Dark Glass UI
- Dual-Screen Synchronization (Windshield AR HUD + 3D Tesla FSD Vector Space)
- Thread-Safe Video Worker Engine with Fallback & Zero Race Conditions
- Full Drag & Drop Support for Video Files
- Live YOLO Model Switcher (YOLO11s, YOLO11m, YOLO11n)
- 3 Instant 3D Camera Presets (FSD Driver, Helicopter 45°, Top-Down)
- Camera Auto-Calibration & Horizon Pitch Slider
- 1-Click Dual-Screen PNG Snapshot & MP4 Video Export
- Collapsible Quick Toggles (Visor Brackets, Drivable Carpet, Lanes, Horizon)
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
    Rock-Solid Thread-Safe Background Worker processing video frames asynchronously.
    Guarantees no race conditions with OpenCV VideoCapture descriptors.
    """
    frame_processed = QtCore.Signal(np.ndarray, object)
    playback_state_changed = QtCore.Signal(bool)
    position_changed = QtCore.Signal(int, int)
    source_load_failed = QtCore.Signal(str)

    def __init__(self, engine: PanopticPerceptionEngine):
        super().__init__()
        self.engine = engine
        self.mutex = QtCore.QMutex()

        self._pending_source: Optional[Any] = None
        self._pending_seek: int = -1
        self._toggle_pause_requested: bool = False

        self.is_running = True
        self.is_paused = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.target_fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.loop_video = True

    def open_source(self, source_path_or_idx):
        """Thread-safe request to switch video source."""
        with QtCore.QMutexLocker(self.mutex):
            self._pending_source = source_path_or_idx

    def toggle_play_pause(self):
        """Thread-safe play/pause toggle."""
        with QtCore.QMutexLocker(self.mutex):
            self._toggle_pause_requested = True

    def seek_frame(self, frame_idx: int):
        """Thread-safe frame seek."""
        with QtCore.QMutexLocker(self.mutex):
            self._pending_seek = frame_idx

    def stop(self):
        """Graceful shutdown of worker thread."""
        self.is_running = False
        self.wait(1500)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def run(self):
        while self.is_running:
            # 1. Process pending commands thread-safely
            new_source = None
            new_seek = -1
            toggle_pause = False

            with QtCore.QMutexLocker(self.mutex):
                if self._pending_source is not None:
                    new_source = self._pending_source
                    self._pending_source = None
                if self._pending_seek >= 0:
                    new_seek = self._pending_seek
                    self._pending_seek = -1
                if self._toggle_pause_requested:
                    toggle_pause = True
                    self._toggle_pause_requested = False

            # Switch source if requested
            if new_source is not None:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

                self.cap = cv2.VideoCapture(new_source)
                if self.cap.isOpened():
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.target_fps = fps if (fps and 5.0 <= fps <= 60.0) else 30.0
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.current_frame_idx = 0
                    self.is_paused = False
                    self.playback_state_changed.emit(True)
                else:
                    self.source_load_failed.emit(str(new_source))
                    continue

            # Handle play/pause toggle
            if toggle_pause:
                self.is_paused = not self.is_paused
                self.playback_state_changed.emit(not self.is_paused)

            # Handle timeline seek
            if new_seek >= 0 and self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_seek)
                self.current_frame_idx = new_seek

            # Check capture availability
            if self.cap is None or not self.cap.isOpened():
                self.msleep(30)
                continue

            if self.is_paused:
                self.msleep(30)
                continue

            # 2. Read frame
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

            # 3. Execute panoptic perception pipeline on Apple Silicon M4
            result = self.engine.process_frame(frame, enable_panoptic=True, enable_detection=True)

            self.frame_processed.emit(frame, result)
            self.position_changed.emit(self.current_frame_idx, self.total_frames)

            # 4. Maintain target playback rate
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
        self.setWindowTitle("Car-Vision • Cockpit de Perception Autonome 3D")
        self.resize(1420, 840)
        self.sample_dir = sample_dir

        # Enable Drag and Drop from Finder
        self.setAcceptDrops(True)

        # Perception Engine (defaults to YOLO11s)
        self.engine = PanopticPerceptionEngine(yolo_actor_weights="yolo11s.pt")

        # Worker Thread
        self.worker = VideoProcessingThread(self.engine)
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.playback_state_changed.connect(self._on_playback_state_changed)
        self.worker.position_changed.connect(self._on_position_changed)
        self.worker.source_load_failed.connect(self._on_source_load_failed)

        self._build_ui()
        self._apply_minimal_theme()

        # Cross-screen hover & selection synchronization
        self.video_hud.actor_hovered.connect(self.bev_view.set_highlighted_actor)
        self.bev_view.actor_hovered.connect(self.video_hud.set_highlighted_actor)

        self.worker.start()

        # Load default city driving video
        default_video = os.path.join(self.sample_dir, "city_paris.mp4")
        if not os.path.exists(default_video):
            default_video = os.path.join(self.sample_dir, "highway.mp4")
        if os.path.exists(default_video):
            self.worker.open_source(default_video)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """Accept dragging video files onto the window."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        """Open dropped video file."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'):
                self.worker.open_source(file_path)
                self.lbl_status.setText(f"Vidéo chargée : {os.path.basename(file_path)}")
                break

    def _build_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(8)

        # 1. Top bar
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)

        # 2. Main Dual Splitter
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(4)

        # Left Screen (HUD / Video)
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.video_hud = VideoHUDWidget()
        left_header = self._create_left_header()
        left_layout.addWidget(left_header)
        left_layout.addWidget(self.video_hud, stretch=1)

        # Right Screen (3D Tesla FSD Vector Space)
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
        self.splitter.setSizes([710, 710])
        main_layout.addWidget(self.splitter, stretch=1)

        # 3. Bottom timeline bar
        bottom_bar = self._create_bottom_bar()
        main_layout.addWidget(bottom_bar)

    def _create_top_bar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("background-color: #0c0f16; border-radius: 6px; padding: 2px;")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(10, 4, 10, 4)

        # Brand Title & M4 GPU Chip Status
        title = QtWidgets.QLabel("CAR-VISION")
        title.setStyleSheet("font-family: 'SF Pro Display'; font-size: 13px; font-weight: 800; color: #ffffff; letter-spacing: 1.5px;")
        layout.addWidget(title)

        status_dot = QtWidgets.QLabel("● Apple M4 MPS")
        status_dot.setStyleSheet("font-size: 10px; font-weight: 600; color: #10b981; margin-left: 6px;")
        layout.addWidget(status_dot)

        layout.addSpacing(12)

        # Telemetry Text
        self.lbl_telemetry = QtWidgets.QLabel("0 FPS  •  0 ms  •  0 cibles")
        self.lbl_telemetry.setStyleSheet("font-size: 11px; font-weight: 500; color: #94a3b8; font-family: -apple-system, sans-serif;")
        layout.addWidget(self.lbl_telemetry)

        layout.addStretch()

        # Model Switcher Dropdown (YOLO11s, YOLO11m, YOLO11n)
        lbl_mod = QtWidgets.QLabel("Modèle :")
        lbl_mod.setStyleSheet("font-size: 10px; color: #64748b; font-weight: 600;")
        layout.addWidget(lbl_mod)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setStyleSheet(self._combo_style())
        self.model_combo.addItems([
            "YOLO11s (Défaut • 50 FPS)",
            "YOLO11m (Précision Max • 30 FPS)",
            "YOLO11n (Ultra Rapide • 75 FPS)"
        ])
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self.model_combo)

        layout.addSpacing(10)

        # Sample Video Dropdown
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.setStyleSheet(self._combo_style())
        self._populate_sample_videos()
        self.source_combo.currentIndexChanged.connect(self._on_sample_changed)
        layout.addWidget(self.source_combo)

        # Open File Button
        btn_open = QtWidgets.QPushButton("Ouvrir...")
        btn_open.setStyleSheet(self._btn_style())
        btn_open.clicked.connect(self._on_open_file)
        layout.addWidget(btn_open)

        # 1-Click Snapshot Button
        btn_snap = QtWidgets.QPushButton("📸 Capture")
        btn_snap.setStyleSheet(self._btn_style())
        btn_snap.setToolTip("Enregistrer une capture d'écran PNG du double écran")
        btn_snap.clicked.connect(self._on_take_snapshot)
        layout.addWidget(btn_snap)

        # Settings Dialog Button
        btn_settings = QtWidgets.QPushButton("⚙️ Réglages")
        btn_settings.setStyleSheet(self._btn_style())
        btn_settings.setToolTip("Ajuster le compromis Vitesse ↔ Précision et les cadences")
        btn_settings.clicked.connect(self._on_open_settings)
        layout.addWidget(btn_settings)

        # Webcam Button
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

        # Box mode toggle (2D Brackets / 3D Cuboids / Off)
        self.btn_box_toggle = QtWidgets.QPushButton("Viseur 2D")
        self.btn_box_toggle.setStyleSheet(self._btn_style(active=True))
        self.btn_box_toggle.clicked.connect(self._toggle_box_mode)
        layout.addWidget(self.btn_box_toggle)

        # Drivable Carpet Toggle
        self.btn_carpet = QtWidgets.QPushButton("Tapis de route")
        self.btn_carpet.setStyleSheet(self._btn_style(active=True))
        self.btn_carpet.clicked.connect(self._toggle_carpet)
        layout.addWidget(self.btn_carpet)

        # Lanes Toggle
        self.btn_lanes = QtWidgets.QPushButton("Voies")
        self.btn_lanes.setStyleSheet(self._btn_style(active=True))
        self.btn_lanes.clicked.connect(self._toggle_lanes)
        layout.addWidget(self.btn_lanes)

        return w

    def _create_right_header(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(4, 2, 4, 2)

        lbl = QtWidgets.QLabel("ESPACE VECTORIEL 3D (TESLA FSD)")
        lbl.setStyleSheet("font-size: 10px; font-weight: 700; color: #64748b; letter-spacing: 1px;")
        layout.addWidget(lbl)
        layout.addStretch()

        # Viewpoint Presets
        self.btn_preset_fsd = QtWidgets.QPushButton("FSD")
        self.btn_preset_fsd.setStyleSheet(self._btn_style(active=True))
        self.btn_preset_fsd.clicked.connect(lambda: self._select_view_preset("FSD"))
        layout.addWidget(self.btn_preset_fsd)

        self.btn_preset_heli = QtWidgets.QPushButton("Hélicoptère")
        self.btn_preset_heli.setStyleSheet(self._btn_style(active=False))
        self.btn_preset_heli.clicked.connect(lambda: self._select_view_preset("HELICOPTER"))
        layout.addWidget(self.btn_preset_heli)

        self.btn_preset_top = QtWidgets.QPushButton("Top-Down")
        self.btn_preset_top.setStyleSheet(self._btn_style(active=False))
        self.btn_preset_top.clicked.connect(lambda: self._select_view_preset("TOP_DOWN"))
        layout.addWidget(self.btn_preset_top)

        # Reset button
        btn_reset = QtWidgets.QPushButton("Recentrer")
        btn_reset.setStyleSheet(self._btn_style())
        btn_reset.clicked.connect(self.bev_view.reset_view)
        layout.addWidget(btn_reset)

        return w

    def _create_bottom_bar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("background-color: #0c0f16; border-radius: 6px; padding: 2px;")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(12)

        # Play / Pause
        self.btn_play = QtWidgets.QPushButton("⏸")
        self.btn_play.setFixedWidth(32)
        self.btn_play.setStyleSheet(self._btn_style(active=True))
        self.btn_play.clicked.connect(self.worker.toggle_play_pause)
        layout.addWidget(self.btn_play)

        # Timeline Slider
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

        # Time code
        self.lbl_time = QtWidgets.QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-size: 10px; font-weight: 500; color: #64748b; font-family: monospace;")
        layout.addWidget(self.lbl_time)

        # Pitch tilt fine adjustment
        lbl_pitch = QtWidgets.QLabel("Pitch :")
        lbl_pitch.setStyleSheet("font-size: 10px; color: #64748b;")
        layout.addWidget(lbl_pitch)

        self.slider_pitch = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_pitch.setFixedWidth(80)
        self.slider_pitch.setRange(1, 12) # 1° to 12°
        self.slider_pitch.setValue(4)
        self.slider_pitch.setStyleSheet("""
            QSlider::groove:horizontal { height: 3px; background: #1e2638; border-radius: 1px; }
            QSlider::handle:horizontal { background: #38bdf8; width: 8px; margin: -3px 0; border-radius: 4px; }
        """)
        self.slider_pitch.valueChanged.connect(self._on_pitch_changed)
        layout.addWidget(self.slider_pitch)

        # ADAS Status badge
        self.lbl_status = QtWidgets.QLabel("Voie dégagée")
        self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 600; color: #10b981;")
        layout.addWidget(self.lbl_status)

        return frame

    def _select_view_preset(self, preset: str):
        self.bev_view.set_view_preset(preset)
        self.btn_preset_fsd.setStyleSheet(self._btn_style(active=(preset == "FSD")))
        self.btn_preset_heli.setStyleSheet(self._btn_style(active=(preset == "HELICOPTER")))
        self.btn_preset_top.setStyleSheet(self._btn_style(active=(preset == "TOP_DOWN")))

    def _toggle_box_mode(self):
        if self.video_hud.box_mode == "2D_BRACKETS":
            self.video_hud.set_box_mode("3D_CUBOID")
            self.btn_box_toggle.setText("Cuboids 3D")
        elif self.video_hud.box_mode == "3D_CUBOID":
            self.video_hud.set_box_mode("OFF")
            self.btn_box_toggle.setText("Boîtes Off")
            self.btn_box_toggle.setStyleSheet(self._btn_style(active=False))
        else:
            self.video_hud.set_box_mode("2D_BRACKETS")
            self.btn_box_toggle.setText("Viseur 2D")
            self.btn_box_toggle.setStyleSheet(self._btn_style(active=True))

    def _toggle_carpet(self):
        self.video_hud.show_drivable_carpet = not self.video_hud.show_drivable_carpet
        self.btn_carpet.setStyleSheet(self._btn_style(active=self.video_hud.show_drivable_carpet))
        self.video_hud.update()

    def _toggle_lanes(self):
        self.video_hud.show_lane_lines = not self.video_hud.show_lane_lines
        self.btn_lanes.setStyleSheet(self._btn_style(active=self.video_hud.show_lane_lines))
        self.video_hud.update()

    def _on_model_changed(self, idx: int):
        models = ["yolo11s.pt", "yolo11m.pt", "yolo11n.pt"]
        chosen = models[idx]
        self.engine.load_actor_model(chosen)
        self.lbl_status.setText(f"Modèle chargé : {chosen}")

    def _on_pitch_changed(self, val: int):
        self.engine.geom.set_camera_parameters(pitch_deg=float(val))

    def _on_take_snapshot(self):
        """Saves a high-resolution snapshot of both screens."""
        os.makedirs("captures", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("captures", f"carvision_{timestamp}.png")
        pix = self.splitter.grab()
        pix.save(filename)
        self.lbl_status.setText(f"📸 Capture enregistrée : {os.path.basename(filename)}")
        self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 600; color: #38bdf8;")

    def _populate_sample_videos(self):
        self.source_combo.clear()
        if os.path.exists(self.sample_dir):
            files = [f for f in os.listdir(self.sample_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
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
            "Vidéos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v)"
        )
        if file_path:
            self.worker.open_source(file_path)

    def _on_source_load_failed(self, source_path: str):
        self.lbl_status.setText(f"❌ Erreur de lecture vidéo : {os.path.basename(source_path)}")
        self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 700; color: #ef4444;")

    def _on_frame_processed(self, frame: np.ndarray, result: PerceptionResult):
        self.video_hud.update_frame(frame, result, self.engine.geom)
        self.bev_view.update_result(result)

        fps = 1000.0 / max(result.inference_time_ms, 1.0)
        self.lbl_telemetry.setText(f"{fps:.0f} FPS  •  {result.inference_time_ms:.0f} ms  •  {len(result.tracks)} cibles")

        # ADAS Status badge
        if result.fcw_alert:
            self.lbl_status.setText("⚠️ RISQUE DE COLLISION")
            self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 700; color: #ef4444;")
        elif result.pedestrian_alert:
            self.lbl_status.setText("⚠️ PIÉTON PROCHE")
            self.lbl_status.setStyleSheet("font-size: 11px; font-weight: 700; color: #f59e0b;")
        elif result.lead_vehicle:
            lv = result.lead_vehicle
            self.lbl_status.setText(f"Véhicule devant : {lv.Z:.1f}m ({lv.speed_kmh:.0f} km/h)")
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

    def _combo_style(self) -> str:
        return """
            QComboBox {
                background-color: #161c28;
                color: #e2e8f0;
                border: 1px solid #283346;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #161c28;
                selection-background-color: #0284c7;
                color: #e2e8f0;
            }
        """

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

    def _on_open_settings(self):
        dlg = SettingsDialog(self.engine, parent=self)
        dlg.exec()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.worker.stop()
        event.accept()


class SettingsDialog(QtWidgets.QDialog):
    """Configuration Dialog for Performance, Cadence, and Calibration."""

    def __init__(self, engine: PanopticPerceptionEngine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Réglages & Performance Car-Vision")
        self.setFixedWidth(460)
        self.engine = engine
        self.setStyleSheet("""
            QDialog {
                background-color: #0b1017;
                color: #e2e8f0;
            }
            QLabel {
                color: #94a3b8;
                font-size: 11px;
                font-weight: 500;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #1e2638;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #0284c7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
            QCheckBox {
                color: #e2e8f0;
                font-size: 11px;
                font-weight: 500;
            }
            QComboBox {
                background-color: #161c28;
                color: #e2e8f0;
                border: 1px solid #283346;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(18, 18, 18, 18)

        lbl_title = QtWidgets.QLabel("PARAMÈTRES DE PERFORMANCE & PERCEPTION")
        lbl_title.setStyleSheet("font-size: 11px; font-weight: 800; color: #38bdf8; letter-spacing: 1px;")
        layout.addWidget(lbl_title)

        # 1. Compromis Vitesse / Précision
        group_perf = QtWidgets.QGroupBox("Compromis ⚡ Vitesse ↔ 🎯 Précision")
        group_perf.setStyleSheet("QGroupBox { font-size: 11px; font-weight: 700; color: #ffffff; border: 1px solid #1e293b; border-radius: 6px; margin-top: 10px; padding-top: 14px; }")
        perf_layout = QtWidgets.QVBoxLayout(group_perf)

        self.lbl_perf_desc = QtWidgets.QLabel(f"Résolution imgsz={self.engine.imgsz}px • Cadence={self.engine.yolop_cadence}")
        self.lbl_perf_desc.setStyleSheet("color: #38bdf8; font-weight: 600;")
        perf_layout.addWidget(self.lbl_perf_desc)

        self.slider_perf = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_perf.setRange(0, 100)
        init_val = 50 if self.engine.imgsz == 576 else (20 if self.engine.imgsz == 480 else 80)
        self.slider_perf.setValue(init_val)
        self.slider_perf.valueChanged.connect(self._on_perf_changed)
        perf_layout.addWidget(self.slider_perf)

        layout.addWidget(group_perf)

        # 2. Cadence YOLOPv2
        cad_layout = QtWidgets.QHBoxLayout()
        lbl_cad = QtWidgets.QLabel("Cadence Segmentation (YOLOPv2) :")
        self.combo_cadence = QtWidgets.QComboBox()
        self.combo_cadence.addItems([
            "1 frame sur 1 (Maximale)",
            "1 frame sur 2 (Rapide)",
            "1 frame sur 3 (Recommandé • +60% FPS)",
            "1 frame sur 4 (Ultra-Rapide)"
        ])
        cad_idx = {1: 0, 2: 1, 3: 2, 4: 3}.get(self.engine.yolop_cadence, 2)
        self.combo_cadence.setCurrentIndex(cad_idx)
        self.combo_cadence.currentIndexChanged.connect(self._on_cadence_changed)
        cad_layout.addWidget(lbl_cad)
        cad_layout.addWidget(self.combo_cadence)
        layout.addLayout(cad_layout)

        # 3. Regroupement Piétons
        self.chk_ped_group = QtWidgets.QCheckBox("Regrouper les piétons marchant ensemble (Polygone convexe)")
        self.chk_ped_group.setChecked(self.engine.enable_pedestrian_grouping)
        self.chk_ped_group.toggled.connect(self._on_ped_group_toggled)
        layout.addWidget(self.chk_ped_group)

        # 4. Calibrage Caméra
        group_cam = QtWidgets.QGroupBox("Calibrage Caméra")
        group_cam.setStyleSheet("QGroupBox { font-size: 11px; font-weight: 700; color: #ffffff; border: 1px solid #1e293b; border-radius: 6px; margin-top: 10px; padding-top: 14px; }")
        cam_layout = QtWidgets.QVBoxLayout(group_cam)

        self.lbl_cam_h = QtWidgets.QLabel(f"Hauteur de caméra : {self.engine.geom.cam_h:.2f} m")
        cam_layout.addWidget(self.lbl_cam_h)
        self.slider_cam_h = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cam_h.setRange(80, 250)
        self.slider_cam_h.setValue(int(self.engine.geom.cam_h * 100))
        self.slider_cam_h.valueChanged.connect(self._on_cam_h_changed)
        cam_layout.addWidget(self.slider_cam_h)

        layout.addWidget(group_cam)

        # Close button
        btn_close = QtWidgets.QPushButton("Appliquer & Fermer")
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                border-radius: 5px;
                padding: 6px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def _on_perf_changed(self, val: int):
        level = val / 100.0
        self.engine.set_performance_level(level)
        if level <= 0.33:
            self.lbl_perf_desc.setText(f"Mode ⚡ Vitesse Maximale (imgsz={self.engine.imgsz}px, Cadence={self.engine.yolop_cadence})")
        elif level <= 0.66:
            self.lbl_perf_desc.setText(f"Mode ⚖️ Équilibré (imgsz={self.engine.imgsz}px, Cadence={self.engine.yolop_cadence})")
        else:
            self.lbl_perf_desc.setText(f"Mode 🎯 Précision Maximale (imgsz={self.engine.imgsz}px, Cadence={self.engine.yolop_cadence})")

    def _on_cadence_changed(self, idx: int):
        cadences = [1, 2, 3, 4]
        self.engine.yolop_cadence = cadences[idx]
        self.engine.vo_cadence = cadences[idx]

    def _on_ped_group_toggled(self, checked: bool):
        self.engine.enable_pedestrian_grouping = checked

    def _on_cam_h_changed(self, val: int):
        h_m = val / 100.0
        self.lbl_cam_h.setText(f"Hauteur de caméra : {h_m:.2f} m")
        self.engine.geom.set_camera_parameters(cam_height=h_m)

