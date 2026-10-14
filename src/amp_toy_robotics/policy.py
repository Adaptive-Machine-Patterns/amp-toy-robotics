# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""What a policy is, and how one is plugged into the closed loop.

A policy is anything with

    predict(images, state, clause) -> chunk

    images   uint8 (3, 3, 512, 512): the top, side and iso cameras, exactly
             as `cameras.observation` renders them
    state    float (7,): 6 joint angles (rad) + gripper aperture (0 open, 1 closed)
    clause   one instruction clause, in canonical wording
             ("Pick up the teal star", "Return to the home pose", ...)
    chunk    float (8, 7): the next 8 joint states + gripper aperture

and nothing else: a policy is never given object positions or any other
scene data (RULES.md).  Any framework will do; numpy in, numpy out.

Policies can be loaded three ways (`load_policy`):

    "teacher"                      the reference (it sees the true scene)
    "my_module:make_policy"        a Python callable returning a policy
    path/to/package/               a model package (needs the `packages`
                                   extra: torch, safetensors, pyyaml)
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from . import cameras, core


@runtime_checkable
class Policy(Protocol):
    def predict(self, images: np.ndarray, state: np.ndarray, clause: str) -> np.ndarray: ...


class PolicyError(ValueError):
    """A policy (or package) is unusable; the message says why."""


def check_chunk(chunk, name: str = "policy") -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float64)
    want = (core.CHUNK, core.STATE_DIM)
    if chunk.shape != want:
        raise PolicyError(f"{name}: predict returned shape {chunk.shape}, "
                          f"expected {want[0]} x {want[1]}")
    if not np.all(np.isfinite(chunk)):
        raise PolicyError(f"{name}: predict returned values that are not finite")
    return chunk


class PolicyBackend:
    """A policy as a backend of `core.run_clauses`: render the cameras, give
    the policy the images, the state and the clause, return its chunk."""

    def __init__(self, policy: Policy, name: str | None = None):
        self.policy = policy
        self.name = name or getattr(policy, "name", type(policy).__name__)

    def chunk(self, parsed: core.Parsed, sim: core.Sim) -> np.ndarray:
        images = cameras.observation(sim.q, sim.objects, sim.grip)
        state = np.append(sim.q, sim.grip)
        return check_chunk(self.policy.predict(images, state, parsed.instruction), self.name)

    def act(self, scene: core.Scene, text: str, start_q=None,
            start_grip: float = 0.0, progress=None) -> core.Trajectory:
        return core.run_clauses(self, scene, text, start_q, start_grip, progress)


# ── model packages (format shared with the AMP Embodied AI Workbench) ──

PACKAGE_FILES = ("model.py", "weights.safetensors", "model.yaml")
PACKAGE_KEYS = ("name", "author", "license")
VALIDATION_SEED = 9_000_000
_count = itertools.count()


@dataclass
class Package:
    """A model package: a folder holding

        model.py              build() -> torch.nn.Module
                              predict(module, images, state, clause) -> chunk
        weights.safetensors   the module's state dict
        model.yaml            name, author, license, optional token_map
        trace.yaml            optional annotations for the model viewer
    """
    folder: Path
    name: str
    meta: dict
    module: Any
    predict_fn: Callable
    trace: dict | None = None
    token_map: Any = field(default=None)

    def predict(self, images, state, clause) -> np.ndarray:
        return check_chunk(self.predict_fn(self.module, images, state, clause), self.name)


def _yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise PolicyError("model packages need pyyaml: pip install 'amp-toy-robotics[packages]'") from None
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise PolicyError(f"{path.name} is not valid YAML: {e}") from None
    if not isinstance(data, dict):
        raise PolicyError(f"{path.name} must be a mapping of keys to values")
    return data


def load_package(folder, device: str = "cpu") -> Package:
    """Load a model package and check it on one observation.

    model.py is ordinary Python and runs with your permissions: load
    packages only from people you trust.  The weights themselves are
    safetensors, which cannot run code."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise PolicyError(f"not a folder: {folder}")
    missing = [f for f in PACKAGE_FILES if not (folder / f).is_file()]
    if missing:
        raise PolicyError(f"{folder.name}: missing {', '.join(missing)}")
    meta = _yaml(folder / "model.yaml")
    absent = [k for k in PACKAGE_KEYS if not str(meta.get(k) or "").strip()]
    if absent:
        raise PolicyError(f"model.yaml: missing {', '.join(absent)}")
    trace = _yaml(folder / "trace.yaml") if (folder / "trace.yaml").is_file() else None

    spec = importlib.util.spec_from_file_location(
        f"amp_model_package_{next(_count)}", folder / "model.py")
    code = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(code)
    except Exception as e:  # noqa: BLE001
        raise PolicyError(f"model.py failed to import: {type(e).__name__}: {e}") from e
    for fn in ("build", "predict"):
        if not callable(getattr(code, fn, None)):
            raise PolicyError(f"model.py does not define {fn}()")
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError:
        raise PolicyError("model packages need torch and safetensors: "
                          "pip install 'amp-toy-robotics[packages]'") from None
    module = code.build()
    if not isinstance(module, torch.nn.Module):
        raise PolicyError(f"build() returned {type(module).__name__}, not a torch.nn.Module")
    try:
        module.load_state_dict(load_file(str(folder / "weights.safetensors")), strict=True)
    except Exception as e:  # noqa: BLE001
        raise PolicyError(f"weights.safetensors does not fit build(): {e}") from e
    module = module.to(device).eval()
    pkg = Package(folder=folder, name=str(meta["name"]), meta=meta, module=module,
                  predict_fn=code.predict, trace=trace, token_map=meta.get("token_map"))
    validate(pkg, pkg.name)
    return pkg


def validation_input():
    """The one observation every policy is checked on when it is loaded."""
    scene = core.new_scene(VALIDATION_SEED)
    clause = core.parse_instruction(f"pick up the {scene.object_names[0]}", scene).instruction
    return cameras.observation(scene.q, scene.objects, 0.0), np.append(scene.q, 0.0), clause


def validate(policy: Policy, name: str = "policy") -> np.ndarray:
    images, state, clause = validation_input()
    try:
        return check_chunk(policy.predict(images, state, clause), name)
    except PolicyError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PolicyError(f"{name}: predict failed: {type(e).__name__}: {e}") from e


def load_policy(spec: str, device: str = "cpu"):
    """A backend for `core.run_clauses` from a spec: "teacher", a model
    package folder, or "module:callable" (the callable returns a policy)."""
    if spec == "teacher":
        return core.TeacherBackend()
    if Path(spec).expanduser().is_dir():
        pkg = load_package(spec, device)
        return PolicyBackend(pkg, pkg.name)
    if ":" not in spec:
        raise PolicyError(f"not a folder and not 'module:callable': {spec}")
    mod, attr = spec.split(":", 1)
    policy = getattr(importlib.import_module(mod), attr)()
    if not isinstance(policy, Policy):
        raise PolicyError(f"{spec} returned {type(policy).__name__}, which has no predict()")
    validate(policy, spec)
    return PolicyBackend(policy, spec)
