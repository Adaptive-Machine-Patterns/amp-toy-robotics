# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""The teacher policy and the simulated world it acts in."""

import numpy as np
import pytest

from amp_toy_robotics import core

BE = core.TeacherBackend()


def _scene(seed=11):
    return core.new_scene(seed)


def test_solve_reach_is_exact_and_in_limits():
    for seed in range(40):
        sc = _scene(seed)
        for o in sc.objects:
            q = core.solve_reach(o["pos"], sc.q)
            assert np.linalg.norm(core.tip_of(q) - np.asarray(o["pos"])) < 1e-6
            assert np.array_equal(q, core.clamp_q(q))


def test_solve_reach_never_moves_the_invisible_joints():
    sc = _scene()
    q = core.solve_reach(sc.objects[0]["pos"], sc.q)
    assert q[3] == sc.q[3] and q[5] == sc.q[5]


def test_teacher_label_is_a_function_of_the_state():
    """Same state -> same label; a nearby state -> a nearby label."""
    sc = _scene()
    p = core.parse_instruction(f"pick up the {sc.object_names[0]}", sc)
    a = core.teacher_chunk(p, sc.q, 0.0)
    assert a.shape == (core.CHUNK, core.STATE_DIM)
    assert np.array_equal(a, core.teacher_chunk(p, sc.q.copy(), 0.0))
    b = core.teacher_chunk(p, sc.q + 1e-4, 0.0)
    assert np.abs(a - b).max() < 1e-3


def test_goal_does_not_change_along_the_way():
    sc = _scene()
    p = core.parse_instruction(f"move to the {sc.object_names[1]}", sc)
    goal, _ = core.teacher_goal(p, sc.q, 0.0)
    q = sc.q
    for _ in range(10):
        q, _g = core.teacher_step(p, q, 0.0)
        assert np.allclose(core.teacher_goal(p, q, 0.0)[0], goal, atol=1e-9)


def test_pick_reaches_closes_and_holds():
    sc = _scene()
    name = sc.object_names[0]
    tr = BE.act(sc, f"pick up the {name}")
    assert tr.tip_error < 1e-6 and tr.grip_seq[-1] == 1.0
    assert tr.grip_seq[: len(tr.grip_seq) // 3].max() == 0.0     # approaches open
    assert tr.meta["grasp_failed"] is None
    assert core.apply_hold(sc, tr).held == name


def test_gripper_only_clauses_do_not_move_the_arm():
    sc = _scene()
    tr = BE.act(sc, "close the gripper then open the gripper")
    assert np.allclose(tr.q_seq, sc.q)
    assert tr.grip_seq.max() == 1.0 and tr.grip_seq[-1] == 0.0


def test_carry_and_release():
    sc = _scene()
    a, b = sc.object_names[0], sc.object_names[-1]
    pos_b = np.asarray(sc.objects[-1]["pos"])
    tr = BE.act(sc, f"pick up the {a} then move to the {b} then open the gripper "
                    f"then return home")
    after = core.apply_hold(sc, tr)
    assert after.held is None
    # the object was left at b, not carried home
    assert np.linalg.norm(np.asarray(after.objects[0]["pos"]) - pos_b) < 2.0
    assert np.abs(tr.q_seq[-1] - core.HOME_Q)[[0, 1, 2, 4]].max() < 0.02
    h = tr.held
    mid = (h.hold_start + h.release_frame) // 2
    riding = core.frame_objects(sc.objects, tr, mid)[0]["pos"]
    assert np.allclose(riding, core.grasp_point(tr.q_seq[mid]))


def test_held_object_carries_over_to_the_next_command():
    sc = _scene()
    a, b = sc.object_names[0], sc.object_names[-1]
    tr1 = BE.act(sc, f"pick up the {a}")
    sc2 = core.apply_hold(sc, tr1)
    tr2 = BE.act(sc2, f"move to the {b}", start_q=tr1.q_seq[-1], start_grip=1.0)
    sc3 = core.apply_hold(sc2, tr2)
    assert sc3.held == a
    assert np.allclose(sc3.objects[0]["pos"], core.grasp_point(tr2.q_seq[-1]))


def test_picking_up_while_holding_another_object_sets_it_down_clear():
    """Carried onto the target, the held object is put down clear of it
    first; dropped on the target it would be the nearest when the gripper
    closes, and be grabbed again (before 0.3.0 the teacher looped on this)."""
    sc = core.new_scene(9010114)
    tr = BE.act(sc, "pick up the gold triangle then move to the teal square "
                    "then pick up the teal square")
    after = core.scene_at(sc, tr, len(tr.q_seq) - 1)
    assert after.held == "teal square"
    gold = next(o for o in after.objects if o["name"] == "gold triangle")
    teal = next(o for o in sc.objects if o["name"] == "teal square")
    assert np.linalg.norm(np.subtract(gold["pos"], teal["pos"])) >= core.SET_DOWN_CLEAR - 1e-6


def test_closing_away_from_any_object_grabs_nothing():
    """No teleporting: a gripper that closes on empty space stays empty."""
    sc = _scene()
    far = dict(sc.objects[0], pos=[120.0, 120.0, 150.0])
    sc = core.Scene(seed=sc.seed, q=sc.q, objects=[far])
    sim = core.Sim(sc, sc.q, 0.0)
    assert np.linalg.norm(core.grasp_point(sc.q) - np.asarray(far["pos"])) > core.GRASP_TOLERANCE
    sim.step(sc.q, 1.0)
    assert sim.held is None and sim.finish() == []


def test_backend_that_misses_is_reported():
    class Lazy:                       # closes the gripper without moving
        name = "lazy"

        def chunk(self, parsed, sim):
            g = min(1.0, sim.grip + 0.2)
            return np.tile(np.append(sim.q, g), (core.CHUNK, 1))

    sc = _scene()
    name = sc.object_names[0]
    tr = core.run_clauses(Lazy(), sc, f"pick up the {name}")
    assert tr.meta["grasp_failed"]["name"] == name
    assert tr.holds == []


def test_runaway_backend_hits_the_step_cap():
    class Spin:
        name = "spin"

        def chunk(self, parsed, sim):
            q = sim.q.copy()
            q[1] += 0.05 if q[1] < 3.5 else -1.0
            return np.tile(np.append(q, sim.grip), (core.CHUNK, 1))

    tr = core.run_clauses(Spin(), _scene(), "return home")
    assert tr.meta["step_cap_hit"]
    assert len(tr.q_seq) <= core.MAX_CLAUSE_STEPS + core.CHUNK + 1


def test_progress_pieces_grow_into_the_final_trajectory():
    sc = _scene()
    a, b = sc.object_names[0], sc.object_names[-1]
    parts = []
    tr = BE.act(sc, f"pick up the {a} then move to the {b}", progress=parts.append)
    assert len(parts) > 3
    for small, big in zip(parts, parts[1:] + [tr]):
        assert len(small.q_seq) <= len(big.q_seq)
        assert np.array_equal(small.q_seq, big.q_seq[:len(small.q_seq)])
    # a piece taken mid-carry already shows the object riding the gripper
    mid = next(p for p in parts if p.holds and len(p.q_seq) < len(tr.q_seq))
    last = len(mid.q_seq) - 1
    assert np.allclose(core.frame_objects(sc.objects, mid, last)[0]["pos"],
                       core.grasp_point(mid.q_seq[last]))


def test_scene_at_a_paused_frame_knows_what_is_in_the_gripper():
    sc = _scene()
    a, b = sc.object_names[0], sc.object_names[-1]
    tr = BE.act(sc, f"pick up the {a} then move to the {b} then open the gripper")
    h = tr.held
    during = core.scene_at(sc, tr, (h.hold_start + h.release_frame) // 2)
    assert during.held == a
    assert core.scene_at(sc, tr, 0).held is None
    assert core.scene_at(sc, tr, len(tr.q_seq) - 1).held is None
    # carrying on from the paused frame keeps the object in the gripper
    i = (h.hold_start + h.release_frame) // 2
    tr2 = BE.act(during, "return home", start_q=tr.q_seq[i], start_grip=float(tr.grip_seq[i]))
    assert core.apply_hold(during, tr2).held == a


def test_results_say_how_each_clause_went():
    sc = _scene()
    tr = BE.act(sc, f"pick up the {sc.object_names[0]} then return home")
    r = tr.meta["results"]
    assert [x["kind"] for x in r] == ["pick", "home"]
    assert r[0]["grasped"] is True and r[0]["distance"] < 1e-6
    assert not any(x["hit_step_limit"] for x in r)
