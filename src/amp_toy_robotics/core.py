# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""The task: scenes, instructions, the teacher, and the closed loop.

- Scene:            seed -> 6-DOF joint pose q + colored workspace objects
- parse_instruction: free text -> canonical instruction + world target +
                     gripper action  (same wording family as the training
                     templates: "move to the {color} {shape}")
- teacher:          a deterministic POLICY (state + clause -> next joint
                    angles + gripper).  The same state always gives the
                    same action, so it can label any state for training.
- Sim / run_clauses: the one closed loop every backend runs through
                    (observe -> action chunk -> execute -> repeat), with
                    physical grasp / carry / release.
- Trajectory:       q_seq (T,6), grip_seq (T,), tip_error, instruction

Pure numpy: no rendering, no machine-learning framework.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import geometry as geo

# ── Joint limits (identical to geometry.random_pose ranges) ──────────
JOINT_LIMITS: List[Tuple[float, float]] = [
    (0.1, 0.8 * math.pi),      # q0 shoulder pitch
    (math.pi / 2, 3 * math.pi / 2),  # q1 base yaw
    (-0.7 * math.pi, 0.7 * math.pi),   # q2 elbow pitch
    (-0.6 * math.pi, 0.6 * math.pi),   # q3 wrist twist (visual)
    (-math.pi, math.pi),       # q4 end-effector bend (visual)
    (-math.pi, math.pi),       # q5 tool twist (visual)
]

CHUNK = 8                     # action-chunk length: steps predicted per observation
STATE_DIM = 7                 # 6 joint angles (rad) + gripper aperture (0 open .. 1 closed)
MAX_CLAUSE_STEPS = 120        # safety cap on one clause's closed loop

# Geometry / vocabulary (single source of truth: geometry.py)
POST_HEIGHT = geo.POST_HEIGHT
L1, L2, L3 = geo.L1, geo.L2, geo.L3
COLOR_NAMES = list(geo.COLOR_NAMES)
SHAPE_NAMES = list(geo.SHAPE_NAMES)
N_JOINTS = 6
N_OBJECTS_DEFAULT = 4


# ── Scene ────────────────────────────────────────────────────────────

@dataclass
class Scene:
    seed: int
    q: np.ndarray                 # (6,) the pose this scene starts in
    objects: List[dict]           # geometry.random_objects() format
    held: Optional[str] = None    # name of the object in the gripper, if any
    instruction: str = ""
    source: str = "random"        # "random" | "val"

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=np.float64).reshape(6)

    @property
    def tip(self) -> np.ndarray:
        """Grasp point (between the fingers, 2/3 down the finger length)."""
        return tip_of(self.q)

    @property
    def object_names(self) -> List[str]:
        return [o["name"] for o in self.objects]


def new_scene(seed: int, n_objects: int = N_OBJECTS_DEFAULT) -> Scene:
    """Deterministic scene from a seed: start pose + objects (FK only)."""
    rng = random.Random(seed)
    q = np.asarray(geo.random_pose(rng), dtype=np.float64)
    objects = geo.random_objects(rng, n_objects)
    return Scene(seed=seed, q=q, objects=objects)


# 'return home' goes to one fixed pose, the same in every scene: the start
# pose of scene 666.  (Each scene's own start pose is only where it begins.)
HOME_Q = new_scene(666).q


# ── Instruction parsing (plain text -> canonical instruction) ────────

@dataclass
class Parsed:
    instruction: str
    target: Optional[np.ndarray] = None      # world xyz
    target_name: Optional[str] = None        # "blue square"
    grip: Optional[str] = None               # "open" | "close"
    kind: str = "unknown"                    # pick|move|midpoint|home|grip|free


def _find_object(text: str, objects: List[dict]) -> Optional[dict]:
    """First object whose '{color} {shape}' (or either word pair) appears."""
    # whole-name match first (in text order), then any object whose name
    # words both appear in the sentence.
    for o in objects:
        if o["name"].lower() in text:
            return o
    for o in objects:
        words = o["name"].lower().split()
        if all(w in text for w in words):
            return o
    return None


def parse_instruction(text: str, scene: Scene) -> Parsed:
    """Map free text to a canonical instruction + target.

    Rules (first match wins):
      * "midpoint between A and B"  -> midpoint target
      * object name present         -> "Pick up the {color} {shape}" (pick verb,
                                       gripper closes) or "Move to the …"
      * "open/close the gripper"    -> grip action (no target)
      * "home/reset/return"         -> the home pose (HOME_Q)
      * anything else               -> move to the first object (demo-safe)
    """
    t = " " + text.lower().strip() + " "
    objects = scene.objects
    names = [o["name"].lower() for o in objects]

    # midpoint between A and B  (fragment-based: names may be multi-word,
    # optionally prefixed by "the")
    a = b = None
    if " between " in t:
        rest = t.split(" between ", 1)[1]
        if " and " in rest:
            a = _find_object(rest.split(" and ", 1)[0], objects)
            b = _find_object(rest.split(" and ", 1)[1], objects)
        if a is not None and b is not None and a is not b:
            pa = np.asarray(a["pos"], dtype=np.float64)
            pb = np.asarray(b["pos"], dtype=np.float64)
            return Parsed(
                instruction=f"Move to the midpoint between {a['name']} and {b['name']}",
                target=(pa + pb) / 2.0, target_name="midpoint", kind="midpoint")

    obj = _find_object(t, objects)
    if obj is not None:
        name = obj["name"]
        if re.search(r"\b(pick|grab|grasp)\b", t):
            return Parsed(
                instruction=f"Pick up the {name}",
                target=np.asarray(obj["pos"], dtype=np.float64),
                target_name=name, grip="close", kind="pick")
        return Parsed(
            instruction=f"Move to the {name}",
            target=np.asarray(obj["pos"], dtype=np.float64),
            target_name=name, kind="move")

    if re.search(r"\bopen\s+(the\s+)?gripper\b", t) or "open fingers" in t:
        return Parsed(instruction="Open the gripper", target=None,
                      grip="open", kind="grip")
    if re.search(r"\bclose\s+(the\s+)?gripper\b", t) or "close fingers" in t:
        return Parsed(instruction="Close the gripper", target=None,
                      grip="close", kind="grip")
    if re.search(r"\b(home|reset|return|rest)\b", t):
        return Parsed(instruction="Return to the home pose",
                      target=None, kind="home")

    # fallback: demo-safe action on the first object
    o0 = objects[0]
    return Parsed(instruction=f"Move to the {o0['name']}",
                  target=np.asarray(o0["pos"], dtype=np.float64),
                  target_name=o0["name"], kind="move")


# ── clause chaining ───────────────────────────────────────────────────
_CLAUSE_SPLIT = re.compile(r"\s+then\s+", re.I)


def split_clauses(text: str) -> List[str]:
    """Split a multi-command instruction on the connector 'then'.

    Only 'then' is treated as a separator (not 'and', because 'and' is
    required inside 'midpoint between A and B').  Each clause is an
    independent, parseable command, executed in order, each one starting
    from where the previous one left off.

        "move to the red square then pick up the blue circle"
          -> ["move to the red square", "pick up the blue circle"]
    """
    t = text.strip()
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(t) if p.strip()]
    return parts or ([t] if t else [])


# ── Gripper geometry + grasp point (single source of truth) ─────────
FINGER_LEN_DISPLAY = 20.0   # finger length (world units), drawn and simulated
GRASP_FRACTION = 2.0 / 3.0  # grasp point: 2/3 down the fingers, between them
# the gripper only picks up an object if, at the moment it closes, the
# grasp point is within this many world units of the object — about one
# object-width (objects are ~8 units across).
GRASP_TOLERANCE = 8.0

def gripper_spacing(grip: float) -> float:
    """Center-to-center finger spacing for grip in [0,1] (open -> closed).

    Single source of truth: render3d draws the fingers with this spacing.
    (The grasp point sits on the centerline, so it does not depend on it.)
    """
    return 20.0 - 14.0 * min(1.0, max(0.0, float(grip)))


def grasp_point(q: np.ndarray, grip: float = 0.0) -> np.ndarray:
    """The GRASP point: on the gripper centerline (BETWEEN the two fingers),
    GRASP_FRACTION (2/3) of the way DOWN the drawn finger length from the
    wrist.  This is where an object sits when the robot grasps it — and
    where the trail ends.  `grip` is kept for API stability; the
    centerline point does not depend on the finger spacing.
    """
    pts, last_pitch, base_yaw = geo.arm_points(np.asarray(q).tolist())
    gg = geo.gripper_points(tuple(pts[-1]), last_pitch, base_yaw,
                            FINGER_LEN_DISPLAY, gripper_spacing(grip))
    b = 0.5 * (np.asarray(gg["left_base"], dtype=np.float64)
               + np.asarray(gg["right_base"], dtype=np.float64))
    t = 0.5 * (np.asarray(gg["left_tip"], dtype=np.float64)
               + np.asarray(gg["right_tip"], dtype=np.float64))
    return b + GRASP_FRACTION * (t - b)


# ── Kinematics helpers (FK only — no IK) ─────────────────────────────

def tip_of(q: np.ndarray) -> np.ndarray:
    """'Tip' of the robot = the GRASP point (between the fingers, 2/3 down
    the finger length) — NOT the wrist.  Teacher sampling, tip_error,
    Scene.tip and the GUI trail all aim this point at the object."""
    return grasp_point(q)


def clamp_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    lo = np.array([a for a, b in JOINT_LIMITS])
    hi = np.array([b for a, b in JOINT_LIMITS])
    return np.clip(q, lo, hi)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def aimed_yaw(target: np.ndarray) -> list:
    """q1 (base yaw) so the arm's plane faces the target.

    The arm is planar: every point of it — wrist, fingers and grasp
    point — lies in the vertical plane through the base at angle q1, so
    the tip's azimuth is exactly q1, or q1+pi when the arm is folded
    over the base.  Reaching a target is therefore only possible with
    the base aimed at it.
    """
    th = math.atan2(float(target[1]), float(target[0]))
    lo, hi = JOINT_LIMITS[1]                      # [pi/2, 3*pi/2], width pi
    th = (th - lo) % (2 * math.pi) + lo            # into [lo, lo + 2*pi)
    cands = [c for c in (th, th - math.pi) if lo <= c <= hi]
    cands.sort(key=lambda c: c > hi)               # forward first
    return [float(c) for c in cands]


# ── Teacher part 1: where to reach ───────────────────────────────────
# The arm is planar once the base yaw is set: three pitch joints (q0, q2,
# q4) place the grasp point in a 2-D plane, so every reachable target has a
# one-parameter family of solutions.  The teacher always picks the solution
# CLOSEST TO THE CURRENT POSE, which makes the goal a pure function of
# (target, current pose) — the property the training labels depend on.
# q3 and q5 neither move the grasp point nor change the rendered image, so
# the teacher never moves them.
LG = L3 + GRASP_FRACTION * FINGER_LEN_DISPLAY     # wrist joint -> grasp point
_A3_GRID = np.linspace(-math.pi, math.pi, 4096, endpoint=False)


def _wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def _planar_solutions(r: float, h: float) -> np.ndarray:
    """All in-limit (q0, q2, q4) that put the grasp point at (r, h) in the
    arm's plane (h measured from the shoulder), one per grid value of the
    last link's pitch and per elbow side.  Exact reach, no search."""
    a3 = _A3_GRID
    wr, wh = r - LG * np.cos(a3), h - LG * np.sin(a3)
    c2 = (wr ** 2 + wh ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    ok = np.abs(c2) <= 1.0
    elbow = np.arccos(np.clip(c2, -1.0, 1.0))
    (lo0, hi0), (lo2, hi2) = JOINT_LIMITS[0], JOINT_LIMITS[2]
    out = []
    for q2 in (elbow, -elbow):
        a1 = _wrap(np.arctan2(wh, wr)
                   - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2)))
        q4 = _wrap(a3 - a1 - q2)
        m = ok & (a1 >= lo0) & (a1 <= hi0) & (q2 >= lo2) & (q2 <= hi2)
        out.append(np.stack([a1, q2, q4], axis=1)[m])
    return np.concatenate(out)


_GRID_CACHE: dict = {}


def _closest_reach(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Fallback for targets the arm cannot touch: the in-limit pose whose
    grasp point is nearest (coarse grid, then a shrinking pattern search)."""
    if not _GRID_CACHE:
        g0 = np.linspace(*JOINT_LIMITS[0], 36)
        g2 = np.linspace(*JOINT_LIMITS[2], 48)
        g4 = np.linspace(*JOINT_LIMITS[4], 64)
        Q0, Q2, Q4 = np.meshgrid(g0, g2, g4, indexing="ij")
        A2, A3 = Q0 + Q2, Q0 + Q2 + Q4
        _GRID_CACHE["q"] = np.stack([Q0.ravel(), Q2.ravel(), Q4.ravel()], 1)
        _GRID_CACHE["r"] = (L1 * np.cos(Q0) + L2 * np.cos(A2) + LG * np.cos(A3)).ravel()
        _GRID_CACHE["h"] = (L1 * np.sin(Q0) + L2 * np.sin(A2) + LG * np.sin(A3)).ravel()
    best_q, best_d = None, float("inf")
    for yaw in aimed_yaw(target):
        q = ref.copy()
        q[1] = yaw
        r = math.hypot(target[0], target[1]) * math.cos(
            math.atan2(target[1], target[0]) - yaw)
        d2 = (_GRID_CACHE["r"] - r) ** 2 + (_GRID_CACHE["h"] - (target[2] - POST_HEIGHT)) ** 2
        q[[0, 2, 4]] = _GRID_CACHE["q"][int(np.argmin(d2))]
        d = float(np.linalg.norm(tip_of(q) - target))
        for step in (0.05, 0.02, 0.008, 0.003, 0.001):
            improved = True
            while improved:
                improved = False
                for j in (0, 2, 4):
                    for s in (step, -step):
                        c = q.copy()
                        c[j] += s
                        c = clamp_q(c)
                        dc = float(np.linalg.norm(tip_of(c) - target))
                        if dc < d - 1e-12:
                            q, d, improved = c, dc, True
        if d < best_d:
            best_q, best_d = q, d
    return best_q


def solve_reach(target, ref_q) -> np.ndarray:
    """Joint pose that puts the grasp point on `target`, choosing — among
    all poses that do — the one nearest `ref_q` in joint space.  q3 and q5
    are copied from `ref_q`.  Deterministic."""
    target = np.asarray(target, dtype=np.float64).reshape(3)
    ref = np.asarray(ref_q, dtype=np.float64).reshape(6)
    th = math.atan2(target[1], target[0])
    rho = math.hypot(target[0], target[1])
    best, best_c = None, float("inf")
    for yaw in aimed_yaw(target):
        sols = _planar_solutions(rho * math.cos(th - yaw), target[2] - POST_HEIGHT)
        if not len(sols):
            continue
        cost = ((sols[:, 0] - ref[0]) ** 2 + (yaw - ref[1]) ** 2
                + (sols[:, 1] - ref[2]) ** 2 + (sols[:, 2] - ref[4]) ** 2)
        i = int(np.argmin(cost))
        if cost[i] < best_c:
            best_c = float(cost[i])
            best = ref.copy()
            best[[0, 2, 4]] = sols[i]
            best[1] = yaw
    return best if best is not None else _closest_reach(target, ref)


# ── Teacher part 2: how to move (a policy, not a pre-planned path) ───
V_MAX = 0.12        # rad per step (joint-space norm) while cruising
V_MIN = 0.03        # rad per step at the end of the ease-out
GAIN = 0.5          # short ease-out: step = GAIN * remaining distance
GRIP_RATE = 0.125   # gripper aperture change per step
CLOSE_RADIUS = 0.05  # 'pick up': start closing once this near the goal (rad)


def teacher_goal(parsed: Parsed, q, grip: float,
                 held: Optional[str] = None) -> Tuple[np.ndarray, float]:
    """Goal (joint pose, gripper aperture) of a clause from state (q, grip).
    `held` names the object currently in the gripper, if any."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    if parsed.target is not None and parsed.kind in ("pick", "move", "midpoint"):
        q_goal = solve_reach(parsed.target, q)
    elif parsed.kind == "home":
        q_goal = q.copy()                 # the joints that cannot be seen stay put
        q_goal[[0, 1, 2, 4]] = HOME_Q[[0, 1, 2, 4]]
    else:
        q_goal = q.copy()
    if parsed.kind == "pick":
        # approach open and close on arrival.  Holding something else, or
        # already closed on nothing: open first, then close on the object.
        near = np.linalg.norm(q_goal - q) <= CLOSE_RADIUS
        if held == parsed.target_name:
            g_goal = 1.0
        else:
            g_goal = 1.0 if near and held is None and grip < GRIP_CLOSED else 0.0
    elif parsed.grip == "close":
        g_goal = 1.0
    elif parsed.grip == "open":
        g_goal = 0.0
    else:
        g_goal = float(grip)
    return q_goal, g_goal


def teacher_step(parsed: Parsed, q, grip: float,
                 held: Optional[str] = None) -> Tuple[np.ndarray, float]:
    """One step of the teacher policy: straight toward the goal in joint
    space, at a speed that depends only on the distance left."""
    q = np.asarray(q, dtype=np.float64).reshape(6)
    q_goal, g_goal = teacher_goal(parsed, q, grip, held)
    d = q_goal - q
    r = float(np.linalg.norm(d))
    if r <= V_MIN:
        q_next = q_goal
    else:
        q_next = q + d * (min(max(GAIN * r, V_MIN), V_MAX) / r)
    g_next = float(grip) + float(np.clip(g_goal - grip, -GRIP_RATE, GRIP_RATE))
    return q_next, g_next


def teacher_chunk(parsed: Parsed, q, grip: float,
                  held: Optional[str] = None, k: int = CHUNK) -> np.ndarray:
    """The next `k` teacher states from (q, grip): (k, 7) = 6 joint angles
    + gripper.  This is the training label for that state."""
    out = np.empty((k, STATE_DIM))
    for i in range(k):
        q, g = teacher_step(parsed, q, grip, held)
        if g < GRIP_OPEN:
            held = None
        elif parsed.kind == "pick" and held is None and grip < GRIP_CLOSED <= g:
            held = parsed.target_name       # closing on the object grasps it
        grip = g
        out[i, :6], out[i, 6] = q, grip
    return out


# ── Simulation: grasp / carry / release ──────────────────────────────
GRIP_CLOSED = 0.9   # aperture at which the fingers count as closed
GRIP_OPEN = 0.5     # aperture below which a held object is released


@dataclass
class HoldInfo:
    """One carry: object `name` rides the gripper (drawn at the grasp
    point) from frame `hold_start` until `release_frame`, the first frame
    it is NOT riding, where it is left at `release_pos`.  If the gripper
    never opens again, `release_frame == len(q_seq)`."""
    name: str
    hold_start: int
    release_frame: int
    release_pos: np.ndarray


class Sim:
    """State of the world while a trajectory runs.  Grasping is physical:
    an object is picked up only if the grasp point is within
    GRASP_TOLERANCE of it at the moment the fingers close, and it is left
    wherever the gripper is when the fingers open."""

    def __init__(self, scene: Scene, q, grip: float):
        self.objects = [dict(o) for o in scene.objects]
        self.q = np.asarray(q, dtype=np.float64).reshape(6).copy()
        self.grip = float(np.clip(grip, 0.0, 1.0))
        self.held = scene.held
        self.q_seq, self.grip_seq = [self.q.copy()], [self.grip]
        self.holds: List[HoldInfo] = []
        self._open_hold: Optional[HoldInfo] = None
        if self.held is not None:
            self._start_hold(self.held)

    def _obj(self, name):
        return next(o for o in self.objects if o["name"] == name)

    def _start_hold(self, name):
        self._open_hold = HoldInfo(name, len(self.q_seq) - 1, -1, np.zeros(3))
        self.holds.append(self._open_hold)

    def step(self, q, grip: float):
        q = clamp_q(q)
        grip = float(np.clip(grip, 0.0, 1.0))
        tip = grasp_point(q)
        if self.held is None and self.grip < GRIP_CLOSED <= grip:
            near = min(self.objects, key=lambda o: float(
                np.linalg.norm(tip - np.asarray(o["pos"]))))
            if np.linalg.norm(tip - np.asarray(near["pos"])) <= GRASP_TOLERANCE:
                self.held = near["name"]
        self.q, self.grip = q, grip
        self.q_seq.append(q.copy())
        self.grip_seq.append(grip)
        if self.held is not None:
            if self._open_hold is None:
                self._start_hold(self.held)
            self._obj(self.held)["pos"] = tip.tolist()
            if grip < GRIP_OPEN:
                self._open_hold.release_frame = len(self.q_seq) - 1
                self._open_hold.release_pos = tip.copy()
                self._open_hold, self.held = None, None

    def finish(self) -> List[HoldInfo]:
        if self._open_hold is not None:
            self._open_hold.release_frame = len(self.q_seq)
            self._open_hold.release_pos = grasp_point(self.q)
        return self.holds

    def holds_so_far(self) -> List[HoldInfo]:
        """The carries up to now, without ending the one in progress."""
        return [replace(h, release_frame=len(self.q_seq), release_pos=grasp_point(self.q))
                if h is self._open_hold else h for h in self.holds]


@dataclass
class Trajectory:
    q_seq: np.ndarray                 # (T, 6); frame 0 is the start pose
    grip_seq: np.ndarray              # (T,)
    instruction: str
    backend: str
    target: Optional[np.ndarray] = None
    tip_error: Optional[float] = None
    holds: List[HoldInfo] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def held(self) -> Optional[HoldInfo]:
        """The first carry of the trajectory, if any."""
        return self.holds[0] if self.holds else None


def frame_objects(objects: List[dict], traj: Optional[Trajectory],
                  idx: int) -> List[dict]:
    """Object list for frame `idx`: a carried object is drawn at the grasp
    point while held and at its release position afterwards."""
    if traj is None or not traj.holds:
        return objects
    pos = {}
    for h in traj.holds:
        if idx >= h.release_frame:
            pos[h.name] = np.asarray(h.release_pos, dtype=np.float64).tolist()
        elif idx >= h.hold_start:
            pos[h.name] = grasp_point(traj.q_seq[idx]).tolist()
    return [dict(o, pos=pos[o["name"]]) if o["name"] in pos else o
            for o in objects]


def scene_at(scene: Scene, traj: Optional[Trajectory], idx: int) -> Scene:
    """The scene as it is at frame `idx` of the trajectory: moved objects
    are where they are at that moment, and `held` names the object in the
    gripper then (if any)."""
    if traj is None or not traj.holds:
        return scene
    objects = [dict(o) for o in frame_objects(scene.objects, traj, idx)]
    held = next((h.name for h in traj.holds
                 if h.hold_start <= idx < h.release_frame), None)
    return replace(scene, objects=objects, held=held)


def apply_hold(scene: Scene, traj: Optional[Trajectory]) -> Scene:
    """The scene AFTER the trajectory: moved objects are where they were
    left, and `held` names the object still in the gripper (if any)."""
    if traj is None:
        return scene
    return scene_at(scene, traj, len(traj.q_seq) - 1)


def is_still(chunk: np.ndarray, q, grip: float) -> bool:
    """True when an action chunk asks for no further motion — the signal
    that the clause is finished."""
    chunk = np.asarray(chunk, dtype=np.float64)
    return (float(np.abs(chunk[:, :6] - np.asarray(q)).max()) < 0.03
            and float(np.abs(chunk[:, 6] - grip).max()) < 0.06)


def run_clauses(backend, scene: Scene, text: str,
                start_q=None, start_grip: float = 0.0,
                progress=None) -> Trajectory:
    """The closed loop every backend runs through.

    The instruction is split on 'then'.  For each clause, repeatedly: ask
    the backend for an action chunk from the CURRENT state, execute it,
    look again — until the backend asks for no further motion.  The
    backend decides everything about the motion, including the gripper and
    when it is done; this loop only executes and simulates.

    `progress`, if given, is called with the trajectory so far each time a
    chunk has been executed (so a GUI can animate while the backend works).
    """
    loop = closed_loop(scene, text, start_q, start_grip, backend.name, progress)
    try:
        parsed, sim = next(loop)
        while True:
            parsed, sim = loop.send(backend.chunk(parsed, sim))
    except StopIteration as done:
        return done.value


def closed_loop(scene: Scene, text: str, start_q=None, start_grip: float = 0.0,
                name: str = "policy", progress=None):
    """`run_clauses` as a generator, for callers that cannot block on the
    backend (e.g. a web page whose model runs asynchronously).  It yields
    `(parsed, sim)` whenever it needs a chunk, takes the chunk through
    `send()`, and returns the Trajectory (`StopIteration.value`)."""
    clauses = split_clauses(text)
    sim = Sim(scene, scene.q if start_q is None else start_q, start_grip)
    instructions, last, grasp_failed, capped, results = [], None, None, False, []

    def so_far(holds):
        return Trajectory(q_seq=np.asarray(sim.q_seq), grip_seq=np.asarray(sim.grip_seq),
                          instruction=" then ".join(instructions), backend=name,
                          holds=holds)

    for clause in clauses:
        parsed = parse_instruction(clause, replace(scene, objects=sim.objects))
        instructions.append(parsed.instruction)
        first_frame = len(sim.q_seq) - 1
        steps, clause_capped = 0, False
        while steps < MAX_CLAUSE_STEPS:
            chunk = yield parsed, sim
            still = is_still(chunk, sim.q, sim.grip)
            for a in chunk[:1] if still else chunk:   # still: just settle
                sim.step(a[:6], a[6])
            steps += len(chunk)
            if progress is not None:
                progress(so_far(sim.holds_so_far()))
            if still:
                break
        else:
            capped = clause_capped = True
        missed = parsed.kind == "pick" and sim.held != parsed.target_name
        if missed and grasp_failed is None:
            grasp_failed = {"name": parsed.target_name, "distance": float(
                np.linalg.norm(grasp_point(sim.q) - parsed.target))}
        results.append({
            "instruction": parsed.instruction, "kind": parsed.kind,
            "frames": (first_frame, len(sim.q_seq) - 1),
            "hit_step_limit": clause_capped,
            "distance": (float(np.linalg.norm(grasp_point(sim.q) - parsed.target))
                         if parsed.target is not None else None),
            "grasped": (not missed) if parsed.kind == "pick" else None})
        last = parsed
    holds = sim.finish()
    target = last.target if last is not None and last.target is not None else None
    return Trajectory(
        q_seq=np.asarray(sim.q_seq), grip_seq=np.asarray(sim.grip_seq),
        instruction=" then ".join(instructions), backend=name,
        target=target,
        tip_error=(float(np.linalg.norm(grasp_point(sim.q) - target))
                   if target is not None else None),
        holds=holds,
        meta={"kind": last.kind if last else "unknown",
              "target_name": last.target_name if last else None,
              "n_clauses": len(clauses), "clauses": instructions,
              "grasp_failed": grasp_failed, "step_cap_hit": capped,
              "results": results})


class TeacherBackend:
    """The teacher as a backend of the closed loop (no model, no network)."""

    name = "Teacher"

    def chunk(self, parsed: Parsed, sim: Sim) -> np.ndarray:
        return teacher_chunk(parsed, sim.q, sim.grip, sim.held)

    def act(self, scene: Scene, text: str, start_q=None,
            start_grip: float = 0.0, progress=None) -> Trajectory:
        return run_clauses(self, scene, text, start_q, start_grip, progress)
