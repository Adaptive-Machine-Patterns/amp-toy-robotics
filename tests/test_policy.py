# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""The policy interface, the closed loop and the scores."""

import numpy as np
import pytest

from amp_toy_robotics import core, evaluate, policy


class StayPut:
    """Never moves: the simplest legal policy."""
    def predict(self, images, state, clause):
        assert images.shape == (3, 3, 512, 512) and images.dtype == np.uint8
        assert state.shape == (7,) and isinstance(clause, str)
        return np.tile(state, (core.CHUNK, 1))


class Wrong:
    def predict(self, images, state, clause):
        return np.zeros((3, 7))


def test_a_policy_runs_in_the_closed_loop():
    backend = policy.PolicyBackend(StayPut(), "stay put")
    scene = core.new_scene(9_000_001)
    traj = backend.act(scene, "open the gripper then close the gripper")
    assert len(traj.meta["results"]) == 2


def test_a_wrong_shape_is_refused_with_a_reason():
    with pytest.raises(policy.PolicyError, match="expected 8 x 7"):
        policy.validate(Wrong(), "wrong")


def test_teacher_scores_near_perfect_and_stay_put_scores_low():
    t = evaluate.evaluate(core.TeacherBackend(), n_scenes=6)
    s = evaluate.evaluate(policy.PolicyBackend(StayPut()), n_scenes=6)
    assert t["success"] > 0.9
    assert s["success"] < t["success"]
    assert "|" in evaluate.report(t)


def test_grounding_of_a_blind_and_deaf_policy_is_zero():
    g = evaluate.grounding(StayPut(), n_scenes=3)
    assert g["image_swap"] == 0 and g["instruction_swap"] == 0 and g["blank_instruction"] == 0


def test_load_policy_from_module_callable():
    b = policy.load_policy("tests.test_policy:StayPut")
    assert isinstance(b, policy.PolicyBackend)


def test_a_folder_that_is_not_a_package_is_refused(tmp_path):
    with pytest.raises(policy.PolicyError, match="missing"):
        policy.load_package(tmp_path)


@pytest.mark.slow
def test_teacher_passes_its_own_evaluation():
    res = evaluate.evaluate(core.TeacherBackend(), n_scenes=15)
    assert res["success"] > 0.95
