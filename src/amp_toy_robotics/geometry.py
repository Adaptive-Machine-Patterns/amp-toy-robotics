# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""The arm and the objects: forward kinematics, gripper geometry, and how a
seed turns into a start pose and a set of objects.

The arm is a post with three pitch joints and three twist joints:

    q[0]  shoulder pitch, from horizontal      0.1 .. 0.8 pi
    q[1]  base yaw (the arm's plane)          pi/2 .. 3 pi/2
    q[2]  elbow pitch, relative               -0.7 pi .. 0.7 pi
    q[3]  wrist twist     (cannot be seen)    -0.6 pi .. 0.6 pi
    q[4]  wrist pitch, relative               -pi .. pi
    q[5]  tool twist      (cannot be seen)    -pi .. pi

Every point of the arm lies in the vertical plane at angle q[1], so the arm
is planar once the base yaw is set.

The order of the arithmetic and of the random draws is part of the
benchmark: a seed must give the same scene, to the last bit, everywhere.
"""

from __future__ import annotations

import random
from math import cos, pi, sin
from typing import Dict, List, Optional, Sequence, Tuple

POST_HEIGHT = 40       # the vertical post
L1 = 60                # shoulder to elbow
L2 = 50                # elbow to wrist
L3 = 35                # wrist to the gripper's crossbar

Point = Tuple[float, float, float]


def arm_points(q: Sequence[float]) -> Tuple[List[Point], float, float]:
    """The arm's joints in world coordinates: ground, top of post, elbow,
    wrist, crossbar.  Also returns the last link's pitch and the base yaw,
    which orient the gripper."""
    yaw = q[1]
    a1 = q[0]
    a2 = q[0] + q[2]
    a3 = q[0] + q[2] + q[4]
    pts: List[Point] = [(0, 0, 0), (0, 0, POST_HEIGHT)]
    for length, pitch in ((L1, a1), (L2, a2), (L3, a3)):
        reach = length * cos(pitch)
        rise = length * sin(pitch)
        x, y, z = pts[-1]
        pts.append((x + reach * cos(yaw), y + reach * sin(yaw), z + rise))
    return pts, a3, yaw


def gripper_points(base: Point, pitch: float, yaw: float,
                   finger_length: float, spacing: float) -> Dict[str, Point]:
    """The two-finger gripper: each finger starts half the spacing either
    side of `base`, across the arm's plane, and points along the last link."""
    along = (cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch))
    across = (-sin(yaw), cos(yaw), 0)
    h = spacing / 2
    left = tuple(b + a * h for b, a in zip(base, across))
    right = tuple(b - a * h for b, a in zip(base, across))
    return {
        "base": base,
        "left_base": left,
        "right_base": right,
        "left_tip": tuple(p + d * finger_length for p, d in zip(left, along)),
        "right_tip": tuple(p + d * finger_length for p, d in zip(right, along)),
    }


def random_pose(rng: Optional[random.Random] = None) -> List[float]:
    """A start pose, uniform within the joint limits (in joint order)."""
    rng = rng or random
    return [rng.uniform(0.1, 0.8 * pi), rng.uniform(pi / 2, 3 * pi / 2),
            rng.uniform(-0.7 * pi, 0.7 * pi), rng.uniform(-0.6 * pi, 0.6 * pi),
            rng.uniform(-pi, pi), rng.uniform(-pi, pi)]


# Object colours: the ColorBrewer "Set2" palette (Cynthia A. Brewer,
# colorbrewer2.org; see NOTICE), in this order.
COLORS = {
    "teal": "#66C2A5", "coral": "#FC8D62", "lavender": "#8DA0CB",
    "pink": "#E78AC3", "lime": "#A6D854", "gold": "#FFD92F",
    "tan": "#E5C494", "grey": "#B3B3B3",
}
COLOR_NAMES = list(COLORS)
SHAPE_NAMES = ["circle", "square", "triangle", "star", "hexagon"]


def rgb(name: str) -> Tuple[int, int, int]:
    h = COLORS[name]
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def random_objects(rng: Optional[random.Random] = None, n: int = 4) -> List[dict]:
    """`n` objects with distinct (colour, shape) pairs, placed at x, y in
    -80..80 and z in 20..80."""
    rng = rng or random
    combos = [(c, s) for c in COLOR_NAMES for s in SHAPE_NAMES]
    objects = []
    for color, shape in rng.sample(combos, n):
        pos = [rng.uniform(-80, 80), rng.uniform(-80, 80), rng.uniform(20, 80)]
        objects.append({"name": f"{color} {shape}", "pos": pos, "color": color,
                        "shape": shape, "rgb": rgb(color)})
    return objects
