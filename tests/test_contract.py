# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""The contract between the task and any policy (RULES.md)."""

import inspect

import numpy as np

from amp_toy_robotics import cameras, core, data, policy


def test_samples_are_determined_by_the_seed():
    a, b = data.samples_for_seed(3, n=4), data.samples_for_seed(3, n=4)
    for k in ("images", "state", "action"):
        assert np.array_equal(a[k], b[k])
    assert a["text"] == b["text"]
    assert a["images"].shape == (4, 3, 3, cameras.OBS_SIZE, cameras.OBS_SIZE)
    assert a["images"].dtype == np.uint8
    assert a["state"].shape == (4, core.STATE_DIM)
    assert a["action"].shape == (4, core.CHUNK, core.STATE_DIM)


def test_a_policy_is_given_only_images_state_and_text():
    # "action" is the label; "goal" is for the optional progress reward only
    assert set(data.samples_for_seed(1, n=1)) == {"images", "state", "text", "action", "goal"}
    assert list(inspect.signature(policy.Policy.predict).parameters) == [
        "self", "images", "state", "clause"]


def test_training_images_are_what_a_policy_sees():
    sc = core.new_scene(8)
    img = cameras.observation(sc.q, sc.objects, 0.0)
    views = cameras.render_views(sc.q, sc.objects, size=cameras.OBS_SIZE, grip=0.0)
    for i, v in enumerate(cameras.VIEW_NAMES):
        assert np.array_equal(img[i].transpose(1, 2, 0), np.asarray(views[v]))


def test_cameras_do_not_move_with_the_arm_or_the_objects():
    a, b = core.new_scene(8), core.new_scene(9)
    ia = cameras.observation(a.q, [], 0.0)
    ib = cameras.observation(b.q, [], 0.0)
    corner = (slice(None), slice(None), slice(0, 24), slice(0, 24))
    assert np.array_equal(ia[corner], ib[corner])


def test_invisible_joints_never_appear_in_the_labels():
    d = data.samples_for_seed(5, n=6)
    assert np.allclose(d["action"][:, :, [3, 5]], d["state"][:, None, [3, 5]])


def test_held_out_seeds_are_never_trained_on():
    assert data.EVAL_SEED0 == 9_000_000
    from amp_toy_robotics import evaluate
    assert evaluate.EVAL_SEED0 >= data.EVAL_SEED0 and evaluate.PROBE_SEED0 >= data.EVAL_SEED0
