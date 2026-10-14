# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""Tests for cameras (headless 3D scene renderer)."""

import numpy as np
import pytest
from PIL import Image

from amp_toy_robotics import cameras as render3d, core


def _scene():
    return core.new_scene(seed=11)


def test_render_scene_shape_and_dtype():
    s = _scene()
    img = render3d.render_scene(np.zeros(6), s.objects, 40, 28, size=128)
    assert isinstance(img, Image.Image)
    assert img.size == (128, 128)
    assert img.mode == "RGB"


def test_render_deterministic():
    s = _scene()
    q = np.array([0.3, -0.4, 0.2, 0.1, 0.5, -0.3])
    a = render3d.render_scene(q, s.objects, 40, 28, size=96)
    b = render3d.render_scene(q, s.objects, 40, 28, size=96)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_render_respects_size():
    s = _scene()
    for size in (64, 256, 512):
        img = render3d.render_scene(np.zeros(6), s.objects, 0, 45, size=size)
        assert img.size == (size, size)


def test_render_views_returns_three():
    s = _scene()
    v = render3d.render_views(np.zeros(6), s.objects, size=96)
    assert set(v) == {"top", "side", "iso"}
    for name, img in v.items():
        assert img.size == (96, 96)


def test_camera_project_framing():
    # all reference points must project inside the canvas
    s = _scene()
    ref = np.array([[0, 0, 40], [100, 100, 80], [-100, -100, 20], [0, 0, 0]])
    cam = render3d.Camera(40, 28, 128, ref)
    px = cam.project(ref)
    assert (px >= 0).all() and (px <= 128).all()


def test_camera_project_identity_orientation():
    pts = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 5.0]])
    size = 100
    cam = render3d.Camera(0.0, 0.0, size, pts)
    px = cam.project(pts)
    # pad=0.12 -> usable fraction (1 - 2*0.12) = 0.76
    scale = (size * 0.76) / (np.ptp(pts, axis=0).max() + 1e-6)
    # az=el=0: camera looks along +Y -> screen x tracks +world X, screen y tracks -world Z
    assert np.allclose(px[:, 0] - px[0, 0], scale * (pts[:, 0] - pts[0, 0]))
    assert np.allclose(px[:, 1] - px[0, 1], -scale * (pts[:, 2] - pts[0, 2]))


def test_depth_ordering_front_closer():
    # at 45 deg elevation the +z side of the arm is closer to the camera
    cam = render3d.Camera(0.0, 45.0, 100, np.array([[0, 0, -50.0], [0, 0, 50.0]]))
    r = render3d.rotate(np.array([[0, 0, 50.0], [0, 0, -50.0]]), 0.0, 45.0)
    near_depth, far_depth = float(r[0, 2]), float(r[1, 2])
    assert near_depth < far_depth
    assert cam.depth_t(near_depth) < cam.depth_t(far_depth)


def test_render_with_trail_and_grid_toggle():
    s = _scene()
    q = np.array([0.2, -0.3, 0.1, 0.2, 0.4, -0.2])
    trail = [core.tip_of(q).tolist(), [0, 0, 60]]
    with_grid = render3d.render_scene(q, s.objects, 30, 25, size=96, trail=trail,
                                      show_grid=True)
    no_grid = render3d.render_scene(q, s.objects, 30, 25, size=96, trail=trail,
                                    show_grid=False)
    assert not np.array_equal(np.asarray(with_grid), np.asarray(no_grid))


def test_render_grip_changes_image():
    s = _scene()
    q = np.zeros(6)
    open_ = render3d.render_scene(q, s.objects, 40, 28, size=96, grip=0.0)
    closed = render3d.render_scene(q, s.objects, 40, 28, size=96, grip=1.0)
    assert not np.array_equal(np.asarray(open_), np.asarray(closed))


def test_render_handles_degenerate_inputs():
    # empty object list, extreme joints — must not raise
    img = render3d.render_scene([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], [], 0, 0, size=64)
    assert img.size == (64, 64)
    img2 = render3d.render_scene(np.zeros(6), [], 179, 89, size=64)
    assert img2.size == (64, 64)
