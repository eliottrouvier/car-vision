"""
perception_engine.py - Unified Panoptic Perception Engine for Autonomous Driving.
Integrates YOLOPv2 Panoptic Road Segmentation with YOLO11 High-Accuracy Actor Detection,
Pedestrian Crowd Clustering, Adaptive Cadence Scheduling, and 3D Metric Projection.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import time
import math
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
        foe: Optional[Tuple[float, float]] = None,
        trajectory_corridor: Optional[List[Tuple[float, float]]] = None,
        pedestrian_groups: Optional[List[Dict[str, Any]]] = None
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
        self.trajectory_corridor = trajectory_corridor or []
        self.pedestrian_groups = pedestrian_groups or []


class PanopticPerceptionEngine:
    """
    Unified high-performance panoptic engine targeting Apple Silicon M4 GPU (MPS).
    Includes adaptive cadence scheduling and pedestrian crowd clustering.
    """

    def __init__(
        self,
        yolop_weights: str = "weights/yolopv2.pt",
        yolo_actor_weights: str = "yolo11s.pt",
        cam_height: float = 1.35,
        pitch_deg: float = -4.5,
        fov_deg: float = 65.0
    ):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[PerceptionEngine] Initializing on device: {self.device}")

        # 1. Load YOLOPv2 for Panoptic Road Segmentation
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
                print("[PerceptionEngine] YOLOPv2 download complete.")
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

        # 2. Load YOLO11 for Actor Detection (Default: YOLO11s)
        self.actor_model = None
        self.actor_model_name = yolo_actor_weights
        self.load_actor_model(yolo_actor_weights)

        # 3. BEV Geometry, Tracker & Visual Odometry
        self.geom = BEVGeometry(
            img_w=1280,
            img_h=720,
            cam_height=cam_height,
            pitch_deg=pitch_deg,
            fov_deg=fov_deg
        )
        self.tracker = CarVisionTracker(max_missed=5, metric_dist_thresh=4.5)
        self.vo = VisualOdometry()

        # Performance tuning & adaptive cadence
        self.yolop_cadence = 3             # Run YOLOPv2 every N frames
        self.vo_cadence = 3                # Run Optical Flow every N frames
        self.imgsz = 576                   # YOLO inference size (fast on MPS)
        self.actor_conf = 0.35             # Robust confidence threshold (filters false alarms)
        self.enable_pedestrian_grouping = True

        # Cached panoptic representations for intermediate frames
        self._cached_drivable_mask: Optional[np.ndarray] = None
        self._cached_lane_mask: Optional[np.ndarray] = None
        self._cached_bev_lanes_3d: List[Tuple[float, float]] = []
        self._cached_bev_carpet_3d: List[Tuple[float, float]] = []
        self._cached_crosswalks: List[Dict[str, Any]] = []
        self._cached_is_moving = True
        self._cached_ego_speed = 0.0
        self._cached_foe: Optional[Tuple[float, float]] = None

        self.last_timestamp = time.time()
        self.frame_idx = 0
        self.ego_speed_estimate = 0.0

    def load_actor_model(self, model_name: str):
        """Loads or switches YOLO model on Apple Silicon MPS."""
        print(f"[PerceptionEngine] Loading actor model: {model_name}...")
        try:
            self.actor_model = YOLO(model_name)
            self.actor_model_name = model_name
            print(f"[PerceptionEngine] Actor model {model_name} loaded successfully.")
        except Exception as e:
            print(f"[PerceptionEngine] Failed to load {model_name}: {e}")

    def set_performance_level(self, level: float):
        """
        Adjusts performance trade-off smoothly between 0.0 (Max Speed) and 1.0 (Max Precision).
        """
        level = float(np.clip(level, 0.0, 1.0))
        if level <= 0.33:
            # High-FPS Mode
            self.imgsz = 512
            self.actor_conf = 0.38
            self.yolop_cadence = 4
            self.vo_cadence = 4
        elif level <= 0.66:
            # Balanced Mode
            self.imgsz = 576
            self.actor_conf = 0.35
            self.yolop_cadence = 3
            self.vo_cadence = 3
        else:
            # High-Precision Mode
            self.imgsz = 640
            self.actor_conf = 0.32
            self.yolop_cadence = 2
            self.vo_cadence = 2

    def _preprocess_yolop(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Resize & pad frame to YOLOPv2 expected input (384, 640).
        Content placed in rows 12:372 (height 360).
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
        Extracts drivable road surface mask and lane lines mask.
        """
        orig_w, orig_h = orig_shape

        # 1. Drivable area
        seg_pred = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        seg_cropped = seg_pred[12:372, :]
        drivable_mask = cv2.resize(seg_cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 2. Lane lines
        ll_pred = (torch.sigmoid(ll_out).squeeze() > 0.45).cpu().numpy().astype(np.uint8)
        ll_cropped = ll_pred[12:372, :]
        lane_mask = cv2.resize(ll_cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return drivable_mask, lane_mask

    def _detect_actors(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        High-Recall Road Actor Detection with Hybrid 3D Metric Coordinates.
        """
        detections = []
        if self.actor_model is None:
            return detections

        target_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        fh, fw = frame.shape[:2]
        hood_cutoff = fh * 0.91

        results = self.actor_model(
            frame,
            device='mps',
            verbose=False,
            conf=self.actor_conf,
            imgsz=self.imgsz,
            iou=0.45
        )
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                if cls_id not in target_classes:
                    continue

                class_name = target_classes[cls_id]
                conf = float(b.conf.item())
                xyxy = [float(x) for x in b.xyxy[0].tolist()]

                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                if bw < 14 or bh < 14:
                    continue

                # Anti-hallucination rule 1: Exclude vehicle hood and wipers reflection
                if xyxy[3] > hood_cutoff:
                    continue

                # Anti-hallucination rule 2: Physical aspect ratio plausibility
                aspect = bw / max(bh, 1.0)
                if class_name in ('car', 'truck', 'bus') and (aspect < 0.50 or aspect > 3.2):
                    continue
                if class_name == 'person' and (aspect < 0.15 or aspect > 1.25):
                    continue

                # Anti-hallucination rule 3: Minimum box size in lower screen region
                # A vehicle cannot physically be small if touching the foreground asphalt
                if class_name in ('car', 'truck', 'bus') and (bw < 30 and bh < 30) and xyxy[3] > fh * 0.70:
                    continue

                # Hybrid 3D Depth Estimation
                X, Z, _ = self.geom.estimate_actor_3d(xyxy, class_name)
                if X is None or Z is None or Z < 0.5 or Z > 140.0:
                    continue

                # Anti-hallucination rule 4: Enforce optical-physical consistency
                # A car closer than 12m MUST have a prominent pixel height (>= 40px)
                if class_name in ('car', 'truck', 'bus') and Z < 12.0 and bh < 40:
                    continue

                corners_3d = self.geom.get_3d_box_corners(X, Z, class_name, yaw=0.0)

                detections.append({
                    'box_2d': xyxy,
                    'class_name': class_name,
                    'confidence': conf,
                    'pos_3d': (X, Z),
                    'corners_3d': corners_3d
                })

        return detections

    def _cluster_pedestrians(
        self,
        detections: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Crowd Clustering: Merges pedestrians walking together into unified
        'person_group' entities with 2D & 3D convex hull boundaries.
        Significantly reduces tracking overhead and cleans up visual clutter.
        """
        if not self.enable_pedestrian_grouping:
            return detections, []

        persons = [d for d in detections if d['class_name'] == 'person']
        others = [d for d in detections if d['class_name'] != 'person']

        if len(persons) <= 1:
            return detections, []

        n = len(persons)
        adj = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                xi, zi = persons[i]['pos_3d']
                xj, zj = persons[j]['pos_3d']
                dist_3d = math.hypot(xi - xj, zi - zj)

                bi = persons[i]['box_2d']
                bj = persons[j]['box_2d']
                dx = max(0.0, min(bi[2], bj[2]) - max(bi[0], bj[0]))

                # Cluster if within 1.6m on ground or overlapping horizontally
                if dist_3d < 1.6 or (dx > 0 and dist_3d < 2.5):
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()
        clusters = []
        for i in range(n):
            if i not in visited:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append(comp)

        merged = list(others)
        groups = []

        for comp in clusters:
            if len(comp) == 1:
                merged.append(persons[comp[0]])
            else:
                members = [persons[k] for k in comp]
                min_x1 = min(m['box_2d'][0] for m in members)
                min_y1 = min(m['box_2d'][1] for m in members)
                max_x2 = max(m['box_2d'][2] for m in members)
                max_y2 = max(m['box_2d'][3] for m in members)

                # 2D Convex Hull
                corners_2d = []
                for m in members:
                    b = m['box_2d']
                    corners_2d.extend([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
                hull_2d_raw = cv2.convexHull(np.array(corners_2d, dtype=np.float32))
                hull_2d = [tuple(pt[0]) for pt in hull_2d_raw]

                # 3D Centroid and Convex Hull
                x_mean = float(np.mean([m['pos_3d'][0] for m in members]))
                z_mean = float(np.mean([m['pos_3d'][1] for m in members]))

                pts_3d = []
                for m in members:
                    mx, mz = m['pos_3d']
                    r = 0.5
                    pts_3d.extend([[mx - r, mz], [mx + r, mz], [mx, mz - r], [mx, mz + r]])
                hull_3d_raw = cv2.convexHull(np.array(pts_3d, dtype=np.float32))
                hull_3d = [tuple(pt[0]) for pt in hull_3d_raw]

                corners_3d = self.geom.get_3d_box_corners(x_mean, z_mean, 'person_group', yaw=0.0)

                grp_det = {
                    'box_2d': [min_x1, min_y1, max_x2, max_y2],
                    'class_name': 'person_group',
                    'confidence': float(np.mean([m['confidence'] for m in members])),
                    'pos_3d': (x_mean, z_mean),
                    'corners_3d': corners_3d,
                    'group_size': len(members),
                    'hull_2d': hull_2d,
                    'hull_3d': hull_3d
                }
                merged.append(grp_det)
                groups.append(grp_det)

        return merged, groups

    def _estimate_lane_vanishing_point(self, lane_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Calculates vanishing point where left and right lane boundaries converge.
        """
        h, w = lane_mask.shape
        roi = lane_mask[int(h * 0.40):, :]
        v_offset = int(h * 0.40)

        mid_x = w // 2
        left_pts = np.argwhere(roi[:, :mid_x] > 0)
        right_pts = np.argwhere(roi[:, mid_x:] > 0)

        if len(left_pts) < 25 or len(right_pts) < 25:
            return None

        # Sample points to speed up fitLine
        sub_l = left_pts[::8]
        sub_r = right_pts[::8]

        left_line = cv2.fitLine(np.column_stack((sub_l[:, 1], sub_l[:, 0] + v_offset)), cv2.DIST_L2, 0, 0.01, 0.01)
        right_line = cv2.fitLine(np.column_stack((sub_r[:, 1] + mid_x, sub_r[:, 0] + v_offset)), cv2.DIST_L2, 0, 0.01, 0.01)

        vx1, vy1, x1, y1 = [float(val[0]) for val in left_line]
        vx2, vy2, x2, y2 = [float(val[0]) for val in right_line]

        denom = vx1 * vy2 - vy1 * vx2
        if abs(denom) < 1e-4:
            return None

        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
        vp_x = x1 + t * vx1
        vp_y = y1 + t * vy1

        if 0 < vp_x < w and 0.25 * h < vp_y < 0.72 * h:
            return float(vp_x), float(vp_y)

        return None

    def _detect_crosswalks(
        self,
        frame: np.ndarray,
        drivable_mask: Optional[np.ndarray]
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

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        stripes = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(stripes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stripe_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 60:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / float(bh) if bh > 0 else 0
            if 0.15 <= aspect <= 1.2:
                stripe_boxes.append((x, y, bw, bh))

        if len(stripe_boxes) < 3:
            return []

        stripe_boxes.sort(key=lambda b: b[0])
        groups = []
        cur_group = [stripe_boxes[0]]
        for i in range(1, len(stripe_boxes)):
            prev = cur_group[-1]
            curr = stripe_boxes[i]
            dx = curr[0] - (prev[0] + prev[2])
            dy = abs(curr[1] - prev[1])
            if dx < 45 and dy < 25:
                cur_group.append(curr)
            else:
                if len(cur_group) >= 3:
                    groups.append(cur_group)
                cur_group = [curr]
        if len(cur_group) >= 3:
            groups.append(cur_group)

        crosswalks = []
        for grp in groups:
            min_x = min(b[0] for b in grp)
            max_x = max(b[0] + b[2] for b in grp)
            min_y = min(b[1] for b in grp)
            max_y = max(b[1] + b[3] for b in grp)
            X, Z, _ = self.geom.estimate_actor_3d([min_x, min_y, max_x, max_y], "default")
            if Z is not None and 2.0 <= Z <= 75.0:
                crosswalks.append({
                    'box': [min_x, min_y, max_x, max_y],
                    'X': X,
                    'Z': Z,
                    'stripes': len(grp)
                })

        return crosswalks

    def compute_trajectory_corridor(
        self,
        ego_speed: float,
        is_moving: bool,
        length_m: float = 38.0
    ) -> List[Tuple[float, float]]:
        """
        Computes the projected path ribbon ahead of the ego vehicle.
        """
        pts = []
        num_steps = 18
        curve_factor = math.tan(self.geom.yaw) * 0.6 if is_moving else 0.0

        for i in range(num_steps + 1):
            s = (i / float(num_steps))
            z = 0.5 + s * length_m
            x = curve_factor * (z**1.35) * 0.04
            pts.append((x, z))
        return pts

    def process_frame(
        self,
        frame: np.ndarray,
        enable_panoptic: bool = True,
        enable_detection: bool = True
    ) -> PerceptionResult:
        """
        Runs complete perception pipeline with adaptive cadence scheduling.
        """
        t0 = time.time()
        self.frame_idx += 1
        h, w = frame.shape[:2]
        self.geom.update_resolution(w, h)

        dt = max(0.01, min(t0 - self.last_timestamp, 0.2))
        self.last_timestamp = t0

        # Determine whether to run YOLOPv2 on this frame
        run_panoptic = enable_panoptic and (
            (self.frame_idx % self.yolop_cadence == 0) or (self._cached_drivable_mask is None)
        )

        if run_panoptic and self.yolop_model is not None:
            tensor, orig_shape = self._preprocess_yolop(frame)
            with torch.no_grad():
                out = self.yolop_model(tensor)
                seg_out = out[1]
                ll_out = out[2]
                drivable_mask, lane_mask = self._postprocess_yolop_masks(seg_out, ll_out, orig_shape)

            self._cached_drivable_mask = drivable_mask
            self._cached_lane_mask = lane_mask

            # Project 3D Lanes (Vectorized with NumPy)
            self._cached_bev_lanes_3d = self.geom.unproject_lane_mask(lane_mask, step=12, z_max=85.0)

            # Project 3D Road Polygon
            if np.count_nonzero(drivable_mask) > 100:
                self._cached_bev_carpet_3d = self.geom.unproject_road_polygon(drivable_mask, z_max=80.0)

            # Crosswalks
            self._cached_crosswalks = self._detect_crosswalks(frame, drivable_mask)

            # Lane Vanishing Point
            lane_vp = self._estimate_lane_vanishing_point(lane_mask)
            if lane_vp is not None:
                self.geom.update_vanishing_point(lane_vp[0], lane_vp[1], smoothing=0.97)

        drivable_mask = self._cached_drivable_mask if self._cached_drivable_mask is not None else np.zeros((h, w), dtype=np.uint8)
        lane_mask = self._cached_lane_mask if self._cached_lane_mask is not None else np.zeros((h, w), dtype=np.uint8)
        bev_lanes_3d = self._cached_bev_lanes_3d
        bev_carpet_3d = self._cached_bev_carpet_3d
        crosswalks = self._cached_crosswalks

        # 2. Object Detection (Cars, Trucks, Pedestrians, Cyclists)
        detections = []
        if enable_detection:
            detections = self._detect_actors(frame)

        # 3. Crowd Clustering (Merge pedestrians walking together)
        clustered_detections, pedestrian_groups = self._cluster_pedestrians(detections)

        # 4. Multi-Object Kinematic 3D Tracking
        tracks = self.tracker.update(clustered_detections, dt=dt)

        # 5. Visual Odometry & Motion Cadence
        run_vo = (self.frame_idx % self.vo_cadence == 0)
        if run_vo:
            actor_boxes = [trk.box_2d for trk in tracks]
            is_moving, ego_speed, foe = self.vo.update(frame, actor_boxes)
            self._cached_is_moving = is_moving
            self._cached_ego_speed = ego_speed
            self._cached_foe = foe
            if is_moving and foe is not None:
                self.geom.update_vanishing_point(foe[0], foe[1], smoothing=0.98)

        is_camera_moving = self._cached_is_moving
        ego_speed = self._cached_ego_speed
        foe = self._cached_foe
        self.ego_speed_estimate = ego_speed

        # 6. Safety Flags & Lead Vehicle Assessment
        lead_vehicle = None
        fcw_alert = False
        pedestrian_alert = False

        min_lead_z = 999.0
        for trk in tracks:
            if trk.warning_level == 'critical':
                fcw_alert = True
            if trk.class_name in ('person', 'person_group', 'bicycle') and trk.warning_level in ('critical', 'caution'):
                pedestrian_alert = True

            if trk.in_ego_lane and 1.0 < trk.Z < min_lead_z:
                min_lead_z = trk.Z
                lead_vehicle = trk

        # 7. Driving Trajectory Corridor
        trajectory_corridor = self.compute_trajectory_corridor(
            ego_speed=ego_speed,
            is_moving=is_camera_moving,
            length_m=35.0
        )

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
            ego_speed_estimate=ego_speed,
            horizon_y=horizon_y,
            is_camera_moving=is_camera_moving,
            foe=foe,
            trajectory_corridor=trajectory_corridor,
            pedestrian_groups=pedestrian_groups
        )
