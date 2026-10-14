# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""The benchmark must reproduce the original lab code bit for bit: the same
seed gives the same scene, pixels, teacher labels and training samples.

Runs only when AMP_LAB_SNAPSHOT points at a copy of the original project."""

import importlib
import os
import sys

import numpy as np
import pytest

from amp_toy_robotics import cameras, core, data

LAB = os.environ.get("AMP_LAB_SNAPSHOT")
pytestmark = [pytest.mark.parity,
              pytest.mark.skipif(not LAB, reason="AMP_LAB_SNAPSHOT not set")]


@pytest.fixture(scope="module")
def lab():
    sys.path.insert(0, LAB)
    try:
        return (importlib.import_module("app.core"), importlib.import_module("app.render3d"),
                importlib.import_module("vla.data"))
    finally:
        sys.path.remove(LAB)


SEEDS = [0, 3, 666, 9_000_000, 9_000_007, 9_050_001]


@pytest.mark.parametrize("seed", SEEDS)
def test_same_scene_and_pixels(lab, seed):
    lcore, lrender, _ = lab
    a, b = core.new_scene(seed), lcore.new_scene(seed)
    assert np.array_equal(a.q, b.q)
    assert [o["name"] for o in a.objects] == [o["name"] for o in b.objects]
    assert [o["pos"] for o in a.objects] == [o["pos"] for o in b.objects]
    assert [tuple(o["rgb"]) for o in a.objects] == [tuple(o["rgb"]) for o in b.objects]
    for grip in (0.0, 0.7):
        assert np.array_equal(cameras.observation(a.q, a.objects, grip),
                              lrender.observation(b.q, b.objects, grip))
    assert np.array_equal(np.asarray(cameras.render_scene(a.q, a.objects, size=256)),
                          np.asarray(lrender.render_scene(b.q, b.objects, size=256)))


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_same_teacher_and_closed_loop(lab, seed):
    lcore, _, ldata = lab
    a, b = core.new_scene(seed), lcore.new_scene(seed)
    text = " then ".join(data.program(np.random.default_rng(seed), a.object_names))
    assert text == " then ".join(ldata.program(np.random.default_rng(seed), b.object_names))
    ta = core.TeacherBackend().act(a, text)
    tb = lcore.TeacherBackend().act(b, text)
    assert np.array_equal(ta.q_seq, tb.q_seq) and np.array_equal(ta.grip_seq, tb.grip_seq)
    assert core.HOME_Q.tolist() == lcore.HOME_Q.tolist()


def test_same_training_samples(lab):
    _, _, ldata = lab
    a, b = data.samples_for_seed(41, n=3), ldata.samples_for_seed(41, n=3)
    for k in ("images", "state", "action", "goal"):
        assert np.array_equal(a[k], b[k]), k
    assert a["text"] == b["text"]
