# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""Tests for core (scene generation, FK, parsing, trajectory planning)."""

import math
import random

import numpy as np
import pytest

from amp_toy_robotics import core

HOME_Q = np.array([0.7, np.pi, -0.9, 0.0, 0.3, 0.0])   # a comfortable in-limits pose

# ── geometry sanity ──────────────────────────────────────────────────

def test_tip_at_home_is_deterministic():
    t1 = core.tip_of(np.zeros(6))
    t2 = core.tip_of(np.zeros(6))
    assert np.allclose(t1, t2)
    assert np.isfinite(t1).all()


def test_fk_matches_closed_form_straight_up():
    # robot_render FK convention: q0 = shoulder pitch from horizontal,
    # q1 = base yaw.  q0 = pi/2 with everything else zero -> arm straight up,
    # fingers continue upward, so the grasp point is GRASP_FRACTION of the
    # finger length above the wrist.
    t = core.tip_of(np.array([math.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0]))
    wrist_z = core.POST_HEIGHT + core.L1 + core.L2 + core.L3
    expected = wrist_z + core.GRASP_FRACTION * core.FINGER_LEN_DISPLAY
    assert np.allclose(t, [0.0, 0.0, expected], atol=1e-6)


def test_tip_is_grasp_point_two_thirds_between_fingers():
    # tip == grasp point == centerline point 2/3 down the drawn fingers
    q = HOME_Q
    assert np.allclose(core.tip_of(q), core.grasp_point(q), atol=1e-9)
    pts, last_pitch, base_yaw = core.geo.arm_points(q.tolist())
    gg = core.geo.gripper_points(tuple(pts[-1]), last_pitch, base_yaw,
                                          core.FINGER_LEN_DISPLAY,
                                          core.gripper_spacing(0.0))
    b = 0.5 * (np.asarray(gg["left_base"], float)
               + np.asarray(gg["right_base"], float))
    t = 0.5 * (np.asarray(gg["left_tip"], float)
               + np.asarray(gg["right_tip"], float))
    assert np.allclose(core.grasp_point(q), b + (2.0 / 3.0) * (t - b), atol=1e-9)


def test_grasp_point_independent_of_finger_spacing():
    q = HOME_Q
    assert np.allclose(core.grasp_point(q, 0.0), core.grasp_point(q, 1.0), atol=1e-9)


def test_q1_rotates_tip_in_horizontal_plane():
    # (fingers continue radially when the arm is horizontal, so the grasp
    # point keeps the same radius and height as the wrist)
    # q0 = 0 -> all links horizontal; base yaw (q1) sweeps the tip around the post
    t0 = core.tip_of(np.zeros(6))
    t1 = core.tip_of(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    assert abs(t1[2] - t0[2]) < 1e-6                                    # z unchanged
    assert abs(np.hypot(t1[0], t1[1]) - np.hypot(t0[0], t0[1])) < 1e-6 # same radius
    assert np.hypot(t1[0] - t0[0], t1[1] - t0[1]) > 50.0                # tip moved


def test_aimed_yaw_in_limits_and_covers_all_azimuths():
    lo, hi = core.JOINT_LIMITS[1]
    for az_deg in range(0, 360, 30):
        th = math.radians(az_deg)
        yaws = core.aimed_yaw(np.array([math.cos(th), math.sin(th), 50.0]))
        assert yaws, "no yaw candidate in limits"
        assert all(lo <= yaw <= hi for yaw in yaws)
        forward = [y for y in yaws
                   if math.isclose(math.cos(y), math.cos(th), abs_tol=1e-9)
                   and math.isclose(math.sin(y), math.sin(th), abs_tol=1e-9)]
        if forward:
            assert yaws[0] == forward[0], "forward yaw must come first"
        else:
            # only the folded solution is in limits: it faces the
            # opposite way and the arm reaches over the base
            for y in yaws:
                assert math.isclose(math.cos(y), -math.cos(th), abs_tol=1e-9)
                assert math.isclose(math.sin(y), -math.sin(th), abs_tol=1e-9)


def test_joint_limits_respected():
    q = core.clamp_q(np.array([10.0, -10.0, 10.0, -10.0, 10.0, -10.0]))
    for v, (a, b) in zip(q, core.JOINT_LIMITS):
        assert a - 1e-9 <= v <= b + 1e-9


# ── scene generation ─────────────────────────────────────────────────

def test_scene_seed_reproducible():
    s1 = core.new_scene(seed=42)
    s2 = core.new_scene(seed=42)
    assert s1.seed == s2.seed == 42
    assert s1.object_names == s2.object_names
    assert np.allclose(s1.q, s2.q)
    assert len(s1.object_names) == 4


def test_scene_objects_within_workspace():
    for seed in range(5):
        s = core.new_scene(seed=seed)
        for o in s.objects:
            x, y, z = o["pos"]
            assert -85 <= x <= 85 and -85 <= y <= 85
            assert 20 <= z <= 80


def test_scene_names_well_formed():
    s = core.new_scene(seed=3)
    for n in s.object_names:
        assert n.split()[0] in core.COLOR_NAMES
        assert n.split()[-1] in core.SHAPE_NAMES


# ── instruction parsing ──────────────────────────────────────────────

def test_parse_pick_up_named_object():
    s = core.new_scene(seed=1)
    name = s.object_names[0]
    p = core.parse_instruction(f"pick up the {name}", s)
    assert p.kind == "pick"
    assert p.target is not None
    assert np.allclose(p.target, s.objects[0]["pos"])
    assert p.instruction == f"Pick up the {name}"


def test_parse_is_case_and_punct_insensitive():
    s = core.new_scene(seed=1)
    name = s.object_names[0].upper()
    p = core.parse_instruction(f"!! PICK UP THE {name} ?!", s)
    assert p.kind == "pick"


def test_parse_move_to_is_lighter_action():
    s = core.new_scene(seed=1)
    name = s.object_names[1]
    p = core.parse_instruction(f"move to the {name}", s)
    assert p.kind == "move"
    assert p.grip is None


def test_parse_midpoint_between_two_objects():
    s = core.new_scene(seed=3)
    a, b = s.object_names[0], s.object_names[1]
    for text in (f"move to the midpoint between the {a} and the {b}",
                 f"midpoint between {a} and {b}"):
        p = core.parse_instruction(text, s)
        assert p.kind == "midpoint"
        assert p.instruction == f"Move to the midpoint between {a} and {b}"
        expected = (np.asarray(s.objects[0]["pos"], float)
                    + np.asarray(s.objects[1]["pos"], float)) / 2.0
        assert np.allclose(p.target, expected)


def test_parse_open_close_gripper():
    s = core.new_scene(seed=1)
    assert core.parse_instruction("open the gripper", s).grip == "open"
    assert core.parse_instruction("close the gripper", s).grip == "close"
    assert core.parse_instruction("close the gripper", s).target is None


def test_parse_unknown_text_falls_back_to_first_object():
    s = core.new_scene(seed=1)
    p = core.parse_instruction("do a little dance", s)
    assert p.kind == "move"
    assert p.target is not None
    assert p.instruction == f"Move to the {s.object_names[0]}"


# ── planning ─────────────────────────────────────────────────────────

def test_demo_trajectory_shape_and_end_pose():
    s = core.new_scene(seed=5)
    name = s.object_names[2]
    traj = core.TeacherBackend().act(s, f"pick up the {name}")
    assert traj.q_seq.shape[1] == 6
    assert traj.grip_seq.shape == (len(traj.q_seq),)
    # final tip must be at the target (aimed sampling lands within a few
    # world units — the old un-aimed teacher was ~8 units off on median)
    end_tip = core.tip_of(traj.q_seq[-1])
    assert np.linalg.norm(end_tip - traj.target) <= 5.0


def test_demo_trajectory_starts_at_scene_pose():
    s = core.new_scene(seed=5)
    traj = core.TeacherBackend().act(s, "pick up the " + s.object_names[0])
    assert np.allclose(traj.q_seq[0], s.q, atol=1e-9)


def test_demo_pick_closes_gripper():
    s = core.new_scene(seed=5)
    traj = core.TeacherBackend().act(s, "pick up the " + s.object_names[0])
    assert traj.grip_seq[0] < 0.05
    assert traj.grip_seq[-1] > 0.9


def test_demo_backend_is_reproducible_across_backend_instances():
    s = core.new_scene(seed=8)
    text = "pick up the " + s.object_names[0]
    a = core.TeacherBackend().act(s, text)
    b = core.TeacherBackend().act(s, text)
    assert np.allclose(a.q_seq, b.q_seq)


def test_demo_chains_from_start_q():
    s = core.new_scene(seed=5)
    b = core.TeacherBackend()
    first = b.act(s, "pick up the " + s.object_names[0])
    mid = first.q_seq[25]
    # a new command must start where the arm currently is, not at scene.q
    second = b.act(s, "pick up the " + s.object_names[1], start_q=mid)
    assert np.allclose(second.q_seq[0], mid, atol=1e-9)
    end_tip = core.tip_of(second.q_seq[-1])
    assert np.linalg.norm(end_tip - second.target) <= 5.0


def test_demo_grip_only_keeps_current_pose_when_chained():
    s = core.new_scene(seed=5)
    b = core.TeacherBackend()
    first = b.act(s, "pick up the " + s.object_names[0])
    pose = first.q_seq[-1]
    # a new command must start where the arm AND gripper currently are:
    # after 'pick up' the gripper is closed — 'open' must ramp 1 -> 0,
    # not snap open and re-close
    second = b.act(s, "open the gripper", start_q=pose,
                   start_grip=float(first.grip_seq[-1]))
    assert np.allclose(second.q_seq[0], pose, atol=1e-9)
    assert np.allclose(second.q_seq[-1], pose, atol=1e-9)   # arm stays put
    assert second.grip_seq[0] > 0.95        # starts closed (no snap open)
    assert second.grip_seq[-1] < 0.05       # ends open


def test_split_clauses_basic():
    assert core.split_clauses("pick up the cube") == ["pick up the cube"]
    assert core.split_clauses(
        "move to the red square then pick up the blue circle"
    ) == ["move to the red square", "pick up the blue circle"]
    # 'then' is a split word, not a substring match
    assert core.split_clauses("move to the thenus") == ["move to the thenus"]
    # three clauses
    assert core.split_clauses(
        "move to the red box then move to the blue box then pick up the red box"
    ) == ["move to the red box", "move to the blue box", "pick up the red box"]


# ── carrying ("pick up X then move to Y" moves the object) ──────────

def test_pick_then_move_carries_object():
    s = core.new_scene(seed=7)
    o0, o1 = s.object_names[0], s.object_names[1]
    traj = core.TeacherBackend().act(
        s, f"pick up the {o0} then move to the {o1}")
    h = traj.held
    assert h is not None
    assert h.name == o0
    # carried from the start of the SECOND clause (gripper already on it)
    assert 0 < h.hold_start < len(traj.q_seq)


# ── gripper state continuity across clauses ────────────────────────────────

def _max_grip_jump(traj) -> float:
    g = traj.grip_seq
    return float(np.max(np.abs(np.diff(g)))) if len(g) > 1 else 0.0


def test_grasp_succeeds_when_gripper_reaches_object():
    """The anti-teleport gate must not break GOOD grasps: the demo backend
    reaches the object, so its pick must still carry it."""
    s = core.new_scene(seed=7)
    o0, o1 = s.object_names[0], s.object_names[1]
    traj = core.TeacherBackend().act(
        s, f"pick up the {o0} then move to the {o1}")
    assert traj.held is not None and traj.held.name == o0
    assert traj.meta.get("grasp_failed") is None


def test_move_clause_keeps_gripper_state():
    # 'move to …' must not touch the gripper: closed stays closed,
    # open stays open (no gratuitous close-then-open)
    s = core.new_scene(seed=5)
    b = core.TeacherBackend()
    closed = b.act(s, "move to the " + s.object_names[1],
                   start_grip=1.0)
    assert closed.grip_seq.min() > 0.98
    open_ = b.act(s, "move to the " + s.object_names[1])
    assert open_.grip_seq.max() < 0.02


def test_frame_objects_rides_with_gripper():
    s = core.new_scene(seed=7)
    o0, o1 = s.object_names[0], s.object_names[1]
    traj = core.TeacherBackend().act(
        s, f"pick up the {o0} then move to the {o1}")
    h = traj.held
    # BEFORE the handoff the object is at its scene spot
    objs = core.frame_objects(s.objects, traj, h.hold_start - 1)
    orig = [o for o in objs if o["name"] == o0][0]
    assert np.allclose(orig["pos"], s.objects[0]["pos"])
    # DURING the carry it is drawn AT the grasp point
    i = min(h.hold_start + 5, len(traj.q_seq) - 1)
    objs = core.frame_objects(s.objects, traj, i)
    moving = [o for o in objs if o["name"] == o0][0]
    assert np.allclose(moving["pos"], core.grasp_point(traj.q_seq[i]), atol=1e-9)
    # other objects are never touched
    others = [o for o in objs if o["name"] != o0]
    assert np.allclose(others[0]["pos"], s.objects[1]["pos"])


def test_apply_hold_releases_at_final_pose():
    s = core.new_scene(seed=7)
    o0, o1 = s.object_names[0], s.object_names[1]
    traj = core.TeacherBackend().act(
        s, f"pick up the {o0} then move to the {o1}")
    assert traj.held is not None
    s2 = core.apply_hold(s, traj)
    moved = [o for o in s2.objects if o["name"] == o0][0]
    # released where the gripper ended (the goal of clause 2)
    assert np.allclose(moved["pos"], traj.held.release_pos)
    assert np.linalg.norm(np.array(moved["pos"]) - core.tip_of(traj.q_seq[-1])) < 1e-6
    # the ORIGINAL scene dict was not mutated (pure function)
    orig = [o for o in s.objects if o["name"] == o0][0]
    assert orig["pos"] != moved["pos"]


def test_move_chain_carries_nothing():
    s = core.new_scene(seed=7)
    o0, o1 = s.object_names[0], s.object_names[1]
    traj = core.TeacherBackend().act(
        s, f"move to the {o0} then move to the {o1}")
    assert traj.held is None


def test_pick_pick_carries_first_object_only():
    s = core.new_scene(seed=7)
    o0, o1 = s.object_names[0], s.object_names[1]
    traj = core.TeacherBackend().act(
        s, f"pick up the {o0} then pick up the {o1}")
    assert traj.held is not None
    assert traj.held.name == o0   # the FIRST grasp is what rides


def test_carry_chain_is_reproducible():
    s = core.new_scene(seed=7)
    text = f"pick up the {s.object_names[0]} then move to the {s.object_names[1]}"
    a = core.TeacherBackend().act(s, text)
    b = core.TeacherBackend().act(s, text)
    assert np.allclose(a.q_seq, b.q_seq)
    assert a.held.name == b.held.name
    assert a.held.hold_start == b.held.hold_start
    assert np.allclose(a.held.release_pos, b.held.release_pos)


def test_scene_reset_restores_object_position():
    s = core.new_scene(seed=7)
    o0 = s.object_names[0]
    traj = core.TeacherBackend().act(
        s, f"pick up the {o0} then move to the {s.object_names[1]}")
    moved = core.apply_hold(s, traj)
    # deterministic regeneration from the seed = the original scene
    s3 = core.new_scene(seed=7)
    orig = [o for o in s.objects if o["name"] == o0][0]
    back = [o for o in s3.objects if o["name"] == o0][0]
    assert np.allclose(orig["pos"], back["pos"])
    assert not np.allclose([o for o in moved.objects if o["name"] == o0][0]["pos"],
                           back["pos"])


def test_trajectory_invariants():
    s = core.new_scene(seed=2)
    traj = core.TeacherBackend().act(s, "pick up the " + s.object_names[3])
    for q in traj.q_seq:
        q = np.asarray(q)
        assert np.isfinite(q).all()
        for v, (a, b) in zip(q, core.JOINT_LIMITS):
            assert a - 1e-6 <= v <= b + 1e-6
    assert np.all(np.isfinite(traj.grip_seq))
    assert traj.q_seq.shape[1] == core.N_JOINTS
