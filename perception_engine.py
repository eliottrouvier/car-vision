"""
perception_engine.py - Unified Panoptic Perception Engine for Autonomous Driving.
Integrates YOLOPv2 (Drivable Road Carpet & Lane Line Segmentation) with
YOLO11 Object Detection, Crosswalk Detection, and 3D Metric Projection.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import time
import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from bev_geometry import BEVGeometry
from tracker import CarVisionTracker, TrackedObject
from visual_odometry import VisualOdometry


class PerceptionResult:
    """Encapsulates the complete perception output for one video frame."""
    def __init__(
        self,
        frame_idx: int,
        timestamp: float,
        inference_time_ms: float,
        drivable_mask: np.ndarray,
        lane_mask: np.ndarray,
        tracks: List[TrackedObject],
        crosswalks: List[Dict[str, Any]],
        bev_lanes_3d: List[Tuple[float, float]],
        bev_carpet_3d: List[Tuple[float, float]],
        lead_vehicle: Optional[TrackedObject],
        fcw_alert: bool,
        pedestrian_alert: bool,
        ego_speed_estimate: float,
        horizon_y: int,
        is_camera_moving: bool = True,
        foe: Optional[Tuple[float, float]] = None
    ):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.inference_time_ms = inference_time_ms
        self.drivable_mask = drivable_mask
        self.lane_mask = lane_mask
        self.tracks = tracks
        self.crosswalks = crosswalks
        self.bev_lanes_3d = bev_lanes_3d
        self.bev_carpet_3d = bev_carpet_3d
        self.lead_vehicle = lead_vehicle
        self.fcw_alert = fcw_alert
        self.pedestrian_alert = pedestrian_alert
        self.ego_speed_estimate = ego_speed_estimate
        self.horizon_y = horizon_y
        self.is_camera_moving = is_camera_moving
        self.foe = foe


class PanopticPerceptionEngine:
    """
    Unified high-performance panoptic engine targeting Apple Silicon M4 GPU (MPS).
    """

    def __init__(
        self,
        yolop_weights: str = "weights/yolopv2.pt",
        yolo_actor_weights: str = "yolo11n.pt",
        cam_height: float = 1.35,
        pitch_deg: float = 3.5,
        fov_deg: float = 65.0
    ):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[PerceptionEngine] Initializing on device: {self.device}")

        # Load YOLOPv2 for Panoptic Road Segmentation
        self.yolop_model = None
        if not os.path.exists(yolop_weights):
            print(f"[PerceptionEngine] {yolop_weights} not found. Auto-downloading...")
            os.makedirs(os.path.dirname(yolop_weights) or '.', exist_ok=True)
            try:
                import requests
                url = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
                r = requests.get(url, stream=True, timeout=60)
                with open(yolop_weights, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                print("[PerceptionEngine] Download complete.")
            except Exception as e:
                print(f"[PerceptionEngine] Auto-download failed: {e}")

        if os.path.exists(yolop_weights):
            print(f"[PerceptionEngine] Loading YOLOPv2 from {yolop_weights}...")
            try:
                self.yolop_model = torch.jit.load(yolop_weights, map_location=self.device)
                self.yolop_model.eval()
                print("[PerceptionEngine] YOLOPv2 loaded successfully.")
            except Exception as e:
                print(f"[PerceptionEngine] Warning: Could not load YOLOPv2: {e}")

        # Load YOLO11 for Actor Detection (Cars, Trucks, Buses, Pedestrians, Cyclists)
        self.actor_model = None
        if os.path.exists(yolo_actor_weights):
            print(f"[PerceptionEngine] Loading YOLO11 from {yolo_actor_weights}...")
            try:
                self.actor_model = YOLO(yolo_actor_weights)
                print("[PerceptionEngine] YOLO11 loaded successfully.")
            except Exception as e:
                print(f"[PerceptionEngine] Warning: Could not load YOLO11: {e}")

        # BEV Geometry & Tracker
        self.geom = BEVGeometry(
            img_w=1280,
            img_h=720,
            cam_height=cam_height,
            pitch_deg=pitch_deg,
            fov_deg=fov_deg
        )
        self.tracker = CarVisionTracker(max_missed=6, metric_dist_thresh=4.5)
        self.vo = VisualOdometry()

        self.last_timestamp = time.time()
        self.frame_idx = 0
        self.ego_speed_estimate = 0.0

    def _preprocess_yolop(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Resize & pad frame to YOLOPv2 expected input (384, 640).
        Valid content is placed in lines 12:372 (height 360).
        """
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (640, 360))
        padded = np.zeros((384, 640, 3), dtype=np.uint8)
        padded[12:372, :] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(self.device).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor, (orig_w, orig_h)

    def _postprocess_yolop_masks(
        self,
        seg_out: torch.Tensor,
        ll_out: torch.Tensor,
        orig_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode drivable area mask and lane line mask scaled back to original resolution.
        """
        orig_w, orig_h = orig_shape

        # Drivable area: channel 1 is road
        da_crop = seg_out[0, 1, 12:372, :].cpu().numpy()
        da_binary = (da_crop > 0.5).astype(np.uint8) * 255
        da_full = cv2.resize(da_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Lane lines: channel 0
        ll_crop = ll_out[0, 0, 12:372, :].cpu().numpy()
        ll_binary = (ll_crop > 0.45).astype(np.uint8) * 255
        ll_full = cv2.resize(ll_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return da_full, ll_full

    def _detect_actors(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.28
    ) -> List[Dict[str, Any]]:
        """
        Detects road actors (vehicles, pedestrians, cyclists) and generates 3D positions.
        """
        detections = []
        if self.actor_model is None:
            return detections

        # Target road classes in COCO:
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        target_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        results = self.actor_model(frame, device='mps', verbose=False, conf=conf_thresh)
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                if cls_id not in target_classes:
                    continue

                class_name = target_classes[cls_id]
                conf = float(b.conf.item())
                xyxy = [float(x) for x in b.xyxy[0].tolist()]

                # Bottom contact point of vehicle / person
                u_bot = (xyxy[0] + xyxy[2]) / 2.0
                v_bot = xyxy[3]

                # Project to ground plane
                X, Z = self.geom.image_to_ground(u_bot, v_bot)
                if X is None or Z is None or Z < 0.5 or Z > 120.0:
                    continue

                # Compute 3D bounding cuboid
                corners_3d = self.geom.get_3d_box_corners(X, Z, class_name)

                detections.append({
                    'box_2d': xyxy,
                    'class_name': class_name,
                    'confidence': conf,
                    'pos_3d': (X, Z),
                    'corners_3d': corners_3d
                })

        return detections

    def _detect_crosswalks(
        self,
        frame: np.ndarray,
        drivable_mask: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detects zebra crossings on the drivable road surface.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_top = int(h * 0.45)

        if drivable_mask is not None:
            road_gray = cv2.bitwise_and(gray, gray, mask=drivable_mask)
        else:
            road_gray = gray

        road_pixels = road_gray[roi_top:, :]
        valid_pixels = road_pixels[road_pixels > 25]
        if len(valid_pixels) < 100:
            return []

        thresh_val = np.percentile(valid_pixels, 88)
        thresh_val = max(160, min(thresh_val, 245))

        _, white_mask = cv2.threshold(road_gray, thresh_val, 255, cv2.THRESH_BINARY)
        white_mask[:roi_top, :] = 0

        # Morphological filter for zebra stripes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        stripes = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(stripes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stripe_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 60:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bh / max(bw, 1)
            if 0.8 <= aspect <= 8.0 and bh >= 18:
                stripe_boxes.append((x, y, bw, bh))

        crosswalks = []
        if len(stripe_boxes) >= 3:
            stripe_boxes.sort(key=lambda b: b[0])
            groups = []
            cur_group = [stripe_boxes[0]]
            for i in range(1, len(stripe_boxes)):
                prev = cur_group[-1]
                curr = stripe_boxes[i]
                dx = curr[0] - (prev[0] + prev[2])
                y_diff = abs(curr[1] - prev[1])
                if 5 <= dx <= 120 and y_diff <= 40:
                    cur_group.append(curr)
                else:
                    if len(cur_group) >= 3:
                        groups.append(cur_group)
                    cur_group = [curr]
            if len(cur_group) >= 3:
                groups.append(cur_group)

            for grp in groups:
                min_x = min(b[0] for b in grp)
                max_x = max(b[0] + b[2] for b in grp)
                min_y = min(b[1] for b in grp)
                max_y = max(b[1] + b[3] for b in grp)
                u_mid = (min_x + max_x) / 2.0
                v_bot = float(max_y)
                X, Z = self.geom.image_to_ground(u_mid, v_bot)
                if Z is not None and 2.0 <= Z <= 75.0:
                    crosswalks.append({
                        'box': [min_x, min_y, max_x, max_y],
                        'X': X,
                        'Z': Z,
                        'stripes': len(grp)
                    })

        return crosswalks

    def process_frame(
        self,
        frame: np.ndarray,
        enable_panoptic: bool = True,
        enable_detection: bool = True
    ) -> PerceptionResult:
        """
        Runs complete perception pipeline on a single frame.
        """
        t0 = time.time()
        self.frame_idx += 1
        h, w = frame.shape[:2]
        self.geom.update_resolution(w, h)

        dt = max(0.01, min(t0 - self.last_timestamp, 0.2))
        self.last_timestamp = t0

        drivable_mask = np.zeros((h, w), dtype=np.uint8)
        lane_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Panoptic Segmentation (YOLOPv2)
        if enable_panoptic and self.yolop_model is not None:
            tensor, orig_shape = self._preprocess_yolop(frame)
            with torch.no_grad():
                out = self.yolop_model(tensor)
                # out[0]: detection, out[1]: drivable area, out[2]: lane lines
                seg_out = out[1]
                ll_out = out[2]
                drivable_mask, lane_mask = self._postprocess_yolop_masks(seg_out, ll_out, orig_shape)

        # 2. Object Detection & 3D Projection
        detections = []
        if enable_detection:
            detections = self._detect_actors(frame)

        # 3. Multi-Object Tracking
        tracks = self.tracker.update(detections, dt=dt)

        # 4. Crosswalk Detection
        crosswalks = self._detect_crosswalks(frame, drivable_mask if enable_panoptic else None)

        # 5. Project 3D Lanes & Drivable Boundaries to BEV
        bev_lanes_3d = self.geom.unproject_lane_mask(lane_mask, step=6, z_max=80.0)

        # Extract drivable road boundary points for BEV carpet
        bev_carpet_3d = []
        if enable_panoptic:
            # Sample edge points of drivable mask
            contours, _ = cv2.findContours(drivable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                sampled_pts = largest_c[::10] # Subsample
                for pt in sampled_pts:
                    u, v = pt[0]
                    X, Z = self.geom.image_to_ground(float(u), float(v))
                    if X is not None and Z is not None and 1.0 <= Z <= 75.0:
                        bev_carpet_3d.append((X, Z))

        # 6. Safety Flags
        lead_vehicle = None
        fcw_alert = False
        pedestrian_alert = False

        min_lead_z = 999.0
        for trk in tracks:
            if trk.warning_level == 'critical':
                fcw_alert = True
            if trk.class_name in ('person', 'bicycle') and trk.warning_level in ('critical', 'caution'):
                pedestrian_alert = True

            if trk.in_ego_lane and trk.Z > 1.0 and trk.Z < min_lead_z:
                min_lead_z = trk.Z
                lead_vehicle = trk

        # 6. Optical Flow Camera Motion & Visual Odometry
        actor_boxes = [trk.box_2d for trk in tracks]
        is_camera_moving, ego_speed, foe = self.vo.update(frame, actor_boxes)
        if is_camera_moving and foe is not None:
            self.geom.update_vanishing_point(foe[0], foe[1])
        self.ego_speed_estimate = ego_speed

        # Horizon y coordinate
        horizon_y = int(round(self.geom.v_horizon))

        inference_time_ms = (time.time() - t0) * 1000.0

        return PerceptionResult(
            frame_idx=self.frame_idx,
            timestamp=t0,
            inference_time_ms=inference_time_ms,
            drivable_mask=drivable_mask,
            lane_mask=lane_mask,
            tracks=tracks,
            crosswalks=crosswalks,
            bev_lanes_3d=bev_lanes_3d,
            bev_carpet_3d=bev_carpet_3d,
            lead_vehicle=lead_vehicle,
            fcw_alert=fcw_alert,
            pedestrian_alert=pedestrian_alert,
            ego_speed_estimate=self.ego_speed_estimate,
            horizon_y=horizon_y,
            is_camera_moving=is_camera_moving,
            foe=foe
        )
