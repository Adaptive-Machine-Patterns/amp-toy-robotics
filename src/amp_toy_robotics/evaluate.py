# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""Scoring a policy.

1. Closed-loop success, through the same loop everything runs in
   (`core.run_clauses`).  For each held-out scene a random chain of clauses
   is executed one clause at a time, each starting from wherever the policy
   left the arm.  A clause succeeds if:

       pick up X          X is in the gripper afterwards
       move to ...        the grasp point is within GRASP_TOLERANCE of the target
       open / close       the gripper ends in that state and the arm stayed put
       return home        every visible joint is within 0.05 rad of the home pose

   The teacher is run from the same starting state, giving the ratio of
   grasp-point path lengths (below 1 = a shorter path than the teacher's).

2. Grounding tests: does the policy use the cameras and the words?  Open
   loop, on other held-out scenes, measured against the teacher.  Nothing
   here is given to the policy beyond its usual inputs.

       named_object       "move to the X": cosine between the policy's motion
                          and the teacher's motion toward X (1 = same direction)
       other_objects      the same cosine toward the other objects (should be low)
       image_swap         X and another object swap places, so only the
                          images change: how far the policy's motion changes,
                          as a share of how far the teacher's does
                          (0 = the cameras are ignored)
       instruction_swap   the instruction names another object, so only the
                          words change: the same share (0 = the words are ignored)
       blank_instruction  an empty clause: how far the motion moves from the
                          "move to the X" motion, as a share of that motion
                          (0 = the words make no difference)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from . import cameras, core, data

EVAL_SEED0 = data.EVAL_SEED0 + 10_000        # closed-loop scenes
PROBE_SEED0 = data.EVAL_SEED0 + 50_000       # grounding-test scenes


def path_length(q_seq) -> float:
    tips = np.array([core.grasp_point(q) for q in q_seq])
    return float(np.linalg.norm(np.diff(tips, axis=0), axis=1).sum())


def clause_ok(parsed: core.Parsed, scene: core.Scene, traj: core.Trajectory,
              start_q, start_grip: float) -> bool:
    if traj.meta["step_cap_hit"]:
        return False
    q, g = traj.q_seq[-1], float(traj.grip_seq[-1])
    if parsed.kind == "pick":
        return core.apply_hold(scene, traj).held == parsed.target_name
    if parsed.kind in ("move", "midpoint"):
        return traj.tip_error <= core.GRASP_TOLERANCE
    if parsed.kind == "grip":
        want = 1.0 if parsed.grip == "close" else 0.0
        return abs(g - want) < 0.1 and np.abs(q - start_q).max() < 0.05
    if parsed.kind == "home":
        return np.abs(q - core.HOME_Q)[[0, 1, 2, 4]].max() < 0.05
    return False


def evaluate(backend, n_scenes: int = 300, seed0: int = EVAL_SEED0,
             progress=None) -> dict:
    """Closed-loop success of `backend` (anything with `act`, e.g. a
    `policy.PolicyBackend` or `core.TeacherBackend`)."""
    teacher = core.TeacherBackend()
    rows = []
    for n, seed in enumerate(range(seed0, seed0 + n_scenes)):
        scene = core.new_scene(seed)
        clauses = data.program(np.random.default_rng(seed), scene.object_names)
        q, grip = scene.q.copy(), 0.0
        for clause in clauses:
            parsed = core.parse_instruction(clause, scene)
            traj = backend.act(scene, clause, start_q=q, start_grip=grip)
            ref = teacher.act(scene, clause, start_q=q, start_grip=grip)
            kind = parsed.grip if parsed.kind == "grip" else parsed.kind
            rows.append({"seed": seed, "clause": clause, "kind": kind,
                         "ok": bool(clause_ok(parsed, scene, traj, q, grip)),
                         "tip_error": traj.tip_error,
                         "path": path_length(traj.q_seq),
                         "teacher_path": path_length(ref.q_seq)})
            scene = replace(core.apply_hold(scene, traj), q=scene.q)
            q, grip = traj.q_seq[-1], float(traj.grip_seq[-1])
        if progress is not None:
            progress(n + 1, n_scenes)
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    moving = [r for r in rows if r["teacher_path"] > 1.0]
    return {
        "scenes": n_scenes, "seed0": seed0, "clauses": len(rows),
        "success": float(np.mean([r["ok"] for r in rows])),
        "success_by_kind": {k: float(np.mean([r["ok"] for r in v]))
                            for k, v in sorted(by.items())},
        "mean_tip_error": float(np.mean([r["tip_error"] for r in rows
                                         if r["tip_error"] is not None])),
        "path_ratio_vs_teacher": float(np.sum([r["path"] for r in moving])
                                       / np.sum([r["teacher_path"] for r in moving])),
        "rows": rows}


def _motion(chunk, q) -> np.ndarray:
    """Where a chunk takes the joints that can be seen, from pose q."""
    return (np.asarray(chunk)[-1, :6] - q)[[0, 1, 2, 4]]


def _cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def grounding(policy, n_scenes: int = 100, seed0: int = PROBE_SEED0) -> dict:
    """The grounding tests (module docstring) for a policy with
    `predict(images, state, clause)`."""
    named, other, image_swap, words_swap, blank = [], [], [], [], []
    for seed in range(seed0, seed0 + n_scenes):
        scene = core.new_scene(seed)
        q, state = scene.q, np.append(scene.q, 0.0)
        names = scene.object_names
        rng = np.random.default_rng(seed)
        a, b = (int(i) for i in rng.choice(len(names), 2, replace=False))

        def teacher(objects, name):
            parsed = core.parse_instruction(f"move to the {name}",
                                            replace(scene, objects=objects))
            return _motion(core.teacher_chunk(parsed, q, 0.0), q), parsed.instruction

        def model(objects, clause):
            return _motion(policy.predict(cameras.observation(q, objects, 0.0),
                                          state, clause), q)

        want, clause_a = teacher(scene.objects, names[a])
        want_b, clause_b = teacher(scene.objects, names[b])
        got = model(scene.objects, clause_a)
        named.append(_cos(got, want))
        other.append(np.mean([_cos(got, teacher(scene.objects, n)[0])
                              for i, n in enumerate(names) if i != a]))

        swapped = [dict(o) for o in scene.objects]
        swapped[a]["pos"], swapped[b]["pos"] = scene.objects[b]["pos"], scene.objects[a]["pos"]
        want2 = teacher(swapped, names[a])[0]
        image_swap.append(np.linalg.norm(model(swapped, clause_a) - got)
                          / (np.linalg.norm(want2 - want) + 1e-9))
        words_swap.append(np.linalg.norm(model(scene.objects, clause_b) - got)
                          / (np.linalg.norm(want_b - want) + 1e-9))
        blank.append(np.linalg.norm(model(scene.objects, "") - got)
                     / (np.linalg.norm(got) + 1e-9))
    return {"named_object": float(np.mean(named)),
            "other_objects": float(np.mean(other)),
            "image_swap": float(np.median(image_swap)),
            "instruction_swap": float(np.median(words_swap)),
            "blank_instruction": float(np.median(blank)),
            "probe_scenes": n_scenes, "probe_seed0": seed0}


def report(res: dict) -> str:
    """A short Markdown summary of `evaluate` (and `grounding`) results."""
    kinds = ["pick", "move", "midpoint", "open", "close", "home"]
    by = res["success_by_kind"]
    lines = [f"| clauses | {' | '.join(kinds)} | all | path vs teacher |",
             "|---" * (len(kinds) + 3) + "|",
             f"| {res['clauses']} | " + " | ".join(
                 f"{by[k]:.0%}" if k in by else "–" for k in kinds)
             + f" | **{res['success']:.1%}** | {res['path_ratio_vs_teacher']:.2f} |"]
    if "named_object" in res:
        lines += ["", "| named object | other objects | image swap | instruction swap | blank instruction |",
                  "|---|---|---|---|---|",
                  f"| {res['named_object']:.2f} | {res['other_objects']:.2f} | "
                  f"{res['image_swap']:.2f} | {res['instruction_swap']:.2f} | "
                  f"{res['blank_instruction']:.2f} |"]
    return "\n".join(lines)
