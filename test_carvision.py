"""
test_carvision.py - Comprehensive Test Suite for Car-Vision.
Verifies MPS acceleration, hybrid depth estimation, 3D Kalman tracking,
drivable road polygon unprojection, and sample video playback.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

from bev_geometry import BEVGeometry
from tracker import CarVisionTracker, TrackedObject, KalmanFilter3D, OneEuroFilter
from perception_engine import PanopticPerceptionEngine


def test_bev_geometry_hybrid_depth():
    print("[TEST 1/7] Testing BEV Geometry & Robust Hybrid Depth Estimator...")
    geom = BEVGeometry(1280, 720, cam_height=1.35, pitch_deg=4.2, fov_deg=65.0)

    # 1. Close vehicle contact point
    X, Z = geom.image_to_ground(640, 520)
    assert X is not None and Z is not None, "Failed to unproject valid image point"
    assert 2.0 <= Z <= 15.0, f"Unexpected distance for foreground car: {Z:.2f}m"

    # 2. Hybrid depth estimation on distant car near horizon
    box_distant = [600, 310, 680, 335] # box height = 25px
    X_d, Z_d, Z_prior = geom.estimate_actor_3d(box_distant, "car")
    assert Z_d is not None and 10.0 <= Z_d <= 80.0, f"Hybrid depth estimator failed for distant car: Z={Z_d}"

    # 3. 3D cuboid corners and 2D projection
    corners = geom.get_3d_box_corners(X_d, Z_d, 'car', yaw=0.1)
    assert corners.shape == (8, 3), f"Unexpected corners shape: {corners.shape}"

    pts_2d = geom.project_3d_box(corners)
    assert pts_2d is not None and len(pts_2d) == 8, "Failed to project 3D box to 2D image"
    print(f"  -> BEV Geometry PASSED. Distant car: X={X_d:.2f}m, Z={Z_d:.2f}m, prior={Z_prior:.2f}m")


def test_tracker_kalman_and_one_euro():
    print("[TEST 2/7] Testing 3D Kalman Filter, 1€ Filter & Multi-Object Tracking...")
    geom = BEVGeometry(1280, 720)
    tracker = CarVisionTracker(max_missed=6, metric_dist_thresh=5.0)

    # Frame 1: Car detected at Z=10m (hits=1, Z < 15m so active immediately)
    corners1 = geom.get_3d_box_corners(0.0, 10.0, 'car')
    det1 = [{
        'box_2d': [580, 420, 700, 520],
        'class_name': 'car',
        'confidence': 0.88,
        'pos_3d': (0.0, 10.0),
        'corners_3d': corners1
    }]
    tracks1 = tracker.update(det1, dt=0.033)
    assert len(tracks1) == 1, "Expected 1 active tracked object"
    assert tracks1[0].track_id == 1

    # Frame 2: Same car moved to Z=9.2m (closing in at ~24 m/s relative)
    corners2 = geom.get_3d_box_corners(0.0, 9.2, 'car')
    det2 = [{
        'box_2d': [575, 430, 705, 535],
        'class_name': 'car',
        'confidence': 0.92,
        'pos_3d': (0.0, 9.2),
        'corners_3d': corners2
    }]
    tracks2 = tracker.update(det2, dt=0.033)
    assert len(tracks2) == 1, "Expected 1 active tracked object"
    assert tracks2[0].track_id == 1, "Track ID must remain persistent"
    assert tracks2[0].vz < 0, f"Vehicle must have closing velocity, got vz={tracks2[0].vz}"
    assert tracks2[0].ttc is not None and tracks2[0].ttc > 0, "TTC must be computed"

    # Frame 3: Temporary occlusion (no detection) -> track must coast via Kalman prediction
    tracks3 = tracker.update([], dt=0.033)
    assert len(tracks3) == 1, "Track must coast through temporary occlusion"
    assert tracks3[0].is_coasting, "Track should be marked as coasting"
    print(f"  -> Tracker & Kalman PASSED. Track #1: vz={tracks2[0].vz:.1f} m/s, TTC={tracks2[0].ttc:.2f}s, coasting={tracks3[0].is_coasting}")


def test_samples_all():
    print("[TEST 3/7] Testing All Sample Driving Videos Opening & Decoding...")
    sample_files = ["city_paris.mp4", "highway.mp4", "car-detection.mp4", "person-bicycle-car-detection.mp4"]
    for s in sample_files:
        path = os.path.join("samples", s)
        assert os.path.exists(path), f"Missing sample video: {path}"
        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), f"Cannot open sample: {path}"
        ret, frame = cap.read()
        assert ret and frame is not None, f"Cannot read frame from: {path}"
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"  -> Verified {s} ({frame.shape[1]}x{frame.shape[0]} @ {fps:.1f} FPS, {frames_count} frames)")
    print("  -> Sample videos test PASSED.")


def test_perception_engine_paris():
    print("[TEST 4/7] Testing Panoptic Perception Engine on Paris City Traffic (MPS GPU)...")
    engine = PanopticPerceptionEngine(yolo_actor_weights="yolo11s.pt")
    assert engine.device.type == 'mps', f"Expected Apple Silicon MPS device, got {engine.device.type}"

    cap = cv2.VideoCapture("samples/city_paris.mp4")
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read city_paris.mp4 frame"

    res = engine.process_frame(frame)
    assert res.drivable_mask is not None, "Drivable mask missing"
    assert res.lane_mask is not None, "Lane mask missing"
    assert len(res.tracks) >= 3, f"Expected multiple urban actors detected, got {len(res.tracks)}"
    print(f"  -> Paris City Perception PASSED. Inference: {res.inference_time_ms:.1f} ms, Detected {len(res.tracks)} actors.")


def test_drivable_road_carpet_unprojection():
    print("[TEST 5/7] Testing Drivable Road Surface 3D Polygon Extraction...")
    engine = PanopticPerceptionEngine()

    cap = cv2.VideoCapture("samples/highway.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read highway frame"

    res = engine.process_frame(frame)
    assert len(res.bev_carpet_3d) >= 3, f"Expected 3D road carpet polygon points, got {len(res.bev_carpet_3d)}"
    assert len(res.trajectory_corridor) >= 10, f"Expected trajectory corridor points, got {len(res.trajectory_corridor)}"
    print(f"  -> Road Carpet PASSED. Carpet points: {len(res.bev_carpet_3d)}, Corridor points: {len(res.trajectory_corridor)}")


def test_model_switching():
    print("[TEST 6/7] Testing Dynamic YOLO Model Switching (s -> m -> n)...")
    engine = PanopticPerceptionEngine(yolo_actor_weights="yolo11s.pt")
    assert engine.actor_model_name == "yolo11s.pt"

    engine.load_actor_model("yolo11n.pt")
    assert engine.actor_model_name == "yolo11n.pt"

    engine.load_actor_model("yolo11s.pt")
    assert engine.actor_model_name == "yolo11s.pt"
    print("  -> Model Switching PASSED.")


def test_pedestrian_clustering_and_cadence():
    print("[TEST 7/7] Testing Pedestrian Crowd Clustering & Cadence Caching...")
    engine = PanopticPerceptionEngine()

    # 1. Test crowd clustering logic
    mock_detections = [
        {
            'box_2d': [500, 350, 530, 430],
            'class_name': 'person',
            'confidence': 0.85,
            'pos_3d': (1.0, 12.0),
            'corners_3d': np.zeros((8, 3))
        },
        {
            'box_2d': [525, 352, 555, 432],
            'class_name': 'person',
            'confidence': 0.82,
            'pos_3d': (1.4, 12.2),
            'corners_3d': np.zeros((8, 3))
        },
        {
            'box_2d': [300, 320, 420, 410],
            'class_name': 'car',
            'confidence': 0.91,
            'pos_3d': (-3.0, 18.0),
            'corners_3d': np.zeros((8, 3))
        }
    ]

    merged, groups = engine._cluster_pedestrians(mock_detections)
    assert len(groups) == 1, f"Expected 1 pedestrian group, got {len(groups)}"
    assert groups[0]['group_size'] == 2, f"Expected group size 2, got {groups[0]['group_size']}"
    assert groups[0]['class_name'] == 'person_group'
    assert len(merged) == 2, f"Expected 2 merged detections (1 car + 1 person_group), got {len(merged)}"

    # 2. Test cadence caching
    cap = cv2.VideoCapture("samples/city_paris.mp4")
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read test frame"

    # Frame 1: Full inference
    res1 = engine.process_frame(frame)
    # Frame 2 & 3: Cadence cached frames
    res2 = engine.process_frame(frame)
    res3 = engine.process_frame(frame)

    assert res2.drivable_mask is not None and res2.lane_mask is not None
    assert res2.bev_carpet_3d is not None
    assert res2.inference_time_ms < res1.inference_time_ms * 0.8 or res2.inference_time_ms < 50.0
    print(f"  -> Pedestrian Clustering & Cadence PASSED. Frame 1: {res1.inference_time_ms:.1f}ms, Frame 2 (cached): {res2.inference_time_ms:.1f}ms")


if __name__ == "__main__":
    print("=== Starting Car-Vision Comprehensive Test Suite ===")
    t_start = time.time()
    test_bev_geometry_hybrid_depth()
    test_tracker_kalman_and_one_euro()
    test_samples_all()
    test_perception_engine_paris()
    test_drivable_road_carpet_unprojection()
    test_model_switching()
    test_pedestrian_clustering_and_cadence()
    print(f"=== ALL 7 TESTS PASSED in {time.time() - t_start:.2f}s ===")
