"""
test_carvision.py - Automated Smoke Test Suite for Car-Vision.
Verifies MPS acceleration, panoptic inference, BEV projection, and tracker.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

from bev_geometry import BEVGeometry
from tracker import CarVisionTracker
from perception_engine import PanopticPerceptionEngine


def test_bev_geometry():
    print("[TEST 1/5] Testing BEV Geometry & 3D Projections...")
    geom = BEVGeometry(1280, 720, cam_height=1.35, pitch_deg=3.5, fov_deg=65.0)

    # Test tire contact point
    X, Z = geom.image_to_ground(640, 560)
    assert X is not None and Z is not None, "Failed to unproject valid image point"
    assert 1.0 <= Z <= 20.0, f"Unexpected distance: {Z}m"

    # Test 3D box corners
    corners = geom.get_3d_box_corners(X, Z, 'car')
    assert corners.shape == (8, 3), f"Unexpected corners shape: {corners.shape}"

    # Test projection back to 2D
    pts_2d = geom.project_3d_box(corners)
    assert pts_2d is not None and len(pts_2d) == 8, "Failed to project 3D box to 2D"
    print("  -> BEV Geometry test PASSED.")


def test_tracker():
    print("[TEST 2/5] Testing Multi-Object Tracker...")
    geom = BEVGeometry(1280, 720)
    tracker = CarVisionTracker(max_missed=3)

    # Frame 1: Detection at X=0, Z=30m
    corners1 = geom.get_3d_box_corners(0.0, 30.0, 'car')
    det1 = [{
        'box_2d': [580, 350, 700, 420],
        'class_name': 'car',
        'confidence': 0.88,
        'pos_3d': (0.0, 30.0),
        'corners_3d': corners1
    }]
    tracks1 = tracker.update(det1, dt=0.05)
    assert len(tracks1) == 1, "Expected 1 tracked object"
    assert tracks1[0].track_id == 1

    # Frame 2: Same vehicle moved to Z=28m (closing in at 40 m/s relative)
    corners2 = geom.get_3d_box_corners(0.0, 28.0, 'car')
    det2 = [{
        'box_2d': [575, 355, 705, 428],
        'class_name': 'car',
        'confidence': 0.91,
        'pos_3d': (0.0, 28.0),
        'corners_3d': corners2
    }]
    tracks2 = tracker.update(det2, dt=0.05)
    assert len(tracks2) == 1, "Expected 1 tracked object"
    assert tracks2[0].track_id == 1, "Track ID must persist"
    assert tracks2[0].vz < 0, f"Vehicle should have closing velocity, got vz={tracks2[0].vz}"
    print(f"  -> Tracker test PASSED. Track #{tracks2[0].track_id}: vz={tracks2[0].vz:.1f}m/s, TTC={tracks2[0].ttc}")


def test_samples():
    print("[TEST 3/5] Testing Sample Driving Videos...")
    sample_files = ["city_paris.mp4", "highway.mp4", "car-detection.mp4", "person-bicycle-car-detection.mp4"]
    for s in sample_files:
        path = os.path.join("samples", s)
        assert os.path.exists(path), f"Missing sample video: {path}"
        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), f"Cannot open sample: {path}"
        ret, frame = cap.read()
        assert ret and frame is not None, f"Cannot read frame from: {path}"
        cap.release()
        print(f"  -> Verified {s} ({frame.shape[1]}x{frame.shape[0]})")
    print("  -> Sample videos test PASSED.")


def test_perception_engine():
    print("[TEST 4/5] Testing Panoptic Perception Engine on Apple Silicon MPS...")
    engine = PanopticPerceptionEngine()
    assert engine.device.type == 'mps', f"Expected mps device, got {engine.device.type}"

    # Load test frame
    cap = cv2.VideoCapture("samples/highway.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to load highway frame"

    res = engine.process_frame(frame)
    assert res.drivable_mask is not None, "Drivable mask missing"
    assert res.lane_mask is not None, "Lane mask missing"
    assert res.drivable_mask.shape == frame.shape[:2], "Mask shape mismatch"
    assert res.inference_time_ms < 500.0, f"Inference too slow: {res.inference_time_ms}ms"

    print(f"  -> Panoptic Engine test PASSED: Latency={res.inference_time_ms:.1f}ms, Drivable area pixels={(res.drivable_mask > 0).sum()}, Lanes={(res.lane_mask > 0).sum()}")


def test_qt_init():
    print("[TEST 5/5] Testing PySide6 GUI initialization...")
    user_plugin_dir = os.path.expanduser("~/.local/share/carvision/plugins")
    if os.path.exists(user_plugin_dir):
        os.environ["QT_PLUGIN_PATH"] = user_plugin_dir
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(user_plugin_dir, "platforms")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide6 import QtWidgets
    from main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(sample_dir="samples")
    assert win is not None
    win.worker.stop()
    print("  -> PySide6 GUI test PASSED.")


if __name__ == "__main__":
    print("========================================")
    print("   CAR-VISION AUTOMATED VERIFICATION    ")
    print("========================================")
    t0 = time.time()
    test_bev_geometry()
    test_tracker()
    test_samples()
    test_perception_engine()
    test_qt_init()
    elapsed = time.time() - t0
    print("========================================")
    print(f" ALL 5 TESTS PASSED SUCCESSFULLY! ({elapsed:.1f}s)")
    print("========================================")
