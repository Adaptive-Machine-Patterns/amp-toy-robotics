# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""AMP Toy Robotics Benchmark.

A light, seed-generated benchmark for vision-language-action policies: a
6-joint arm, coloured shapes, three fixed cameras, and instructions in plain
English.  numpy and Pillow only.

    geometry   the arm and the objects
    core       scenes, instructions, the teacher, the closed loop
    cameras    the three fixed cameras (what a policy sees)
    data       unlimited training samples, generated from scene seeds
    policy     the policy interface and model packages
    evaluate   closed-loop success and grounding tests
"""

__version__ = "0.2.0"
