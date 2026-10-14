# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""Training data, generated on the fly from scene seeds (no torch here, so
worker processes stay light).

One sample is one moment in a teacher-driven episode:

    images   uint8 (3, 3, H, W)  the three fixed cameras at that moment
    state    (7,)   6 joint angles (rad) + gripper aperture
    text     the clause being executed, in canonical wording
    action   (CHUNK, 7)  the teacher's next CHUNK states from that moment
    goal     (3,)   where the teacher is taking the grasp point.  Used only
                    by the optional progress reward in the training loss
                    (see RULES.md); it is never shown to the model.

The model is given the images, the state and the text, and nothing else:
no object positions, no targets.

A seed fully determines its samples.
"""

from __future__ import annotations

import functools
import itertools
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import queue
import threading
import zlib
from dataclasses import replace

import numpy as np

from . import cameras, core

PER_SEED = 12            # samples drawn from each scene seed
EVAL_SEED0 = 9_000_000   # held-out scenes start here; training never goes near
JITTER = (0.08, 0.3)     # rad: small and large noise added to sampled poses


def program(rng: np.random.Generator, names: list[str]) -> list[str]:
    """A random chain of 2-4 clauses from the GUI's instruction grammar."""
    out = []
    for _ in range(int(rng.integers(2, 5))):
        kind = rng.choice(["pick", "move", "midpoint", "open", "close", "home"],
                          p=[0.30, 0.25, 0.10, 0.10, 0.10, 0.15])
        a, b = (str(n) for n in rng.choice(names, 2, replace=False))
        out.append({"pick": f"pick up the {a}",
                    "move": f"move to the {a}",
                    "midpoint": f"move to the midpoint between {a} and {b}",
                    "open": "open the gripper",
                    "close": "close the gripper",
                    "home": "return home"}[str(kind)])
    return out


def episode_states(scene: core.Scene, clauses: list[str]) -> list[tuple]:
    """Every state the teacher passes through while running `clauses`,
    including the finished state of each clause (where the label is
    'stay put').  Items: (parsed clause, q, grip, objects, held name)."""
    sim = core.Sim(scene, scene.q, 0.0)
    states = []
    for clause in clauses:
        parsed = core.parse_instruction(clause, replace(scene, objects=sim.objects))
        for _ in range(core.MAX_CLAUSE_STEPS):
            states.append((parsed, sim.q.copy(), sim.grip,
                           [dict(o) for o in sim.objects], sim.held))
            q, g = core.teacher_step(parsed, sim.q, sim.grip, sim.held)
            if np.abs(q - sim.q).max() < 1e-9 and abs(g - sim.grip) < 1e-9:
                break
            sim.step(q, g)
    return states


def samples_for_seed(seed: int, n: int = PER_SEED) -> dict:
    """The `n` training samples of scene `seed`."""
    scene = core.new_scene(seed)
    rng = np.random.default_rng(seed)
    states = episode_states(scene, program(rng, scene.object_names))
    out = {"images": [], "state": [], "action": [], "goal": [], "text": []}
    for i in rng.integers(0, len(states), n):
        parsed, q, grip, objects, held = states[int(i)]
        u = rng.random()
        if u < 0.6:                 # off-path pose: the teacher labels it too,
            q = q.copy()            # which teaches recovery from mistakes
            q[[0, 1, 2, 4]] += rng.normal(0.0, JITTER[0] if u < 0.35 else JITTER[1], 4)
            q = core.clamp_q(q)
            if held is not None:
                tip = core.grasp_point(q).tolist()
                objects = [dict(o, pos=tip) if o["name"] == held else o
                           for o in objects]
        if rng.random() < 0.5:      # a different instruction for the same view:
            # the only way to tell the two labels apart is to read the text
            parsed = core.parse_instruction(program(rng, scene.object_names)[0],
                                            replace(scene, objects=objects))
        out["images"].append(cameras.observation(q, objects, grip))
        out["state"].append(np.append(q, grip))
        out["action"].append(core.teacher_chunk(parsed, q, grip, held))
        out["goal"].append(core.grasp_point(core.teacher_goal(parsed, q, grip, held)[0]))
        out["text"].append(parsed.instruction)
    return {"images": np.stack(out["images"]),
            "state": np.asarray(out["state"], np.float32),
            "action": np.asarray(out["action"], np.float32),
            "goal": np.asarray(out["goal"], np.float32),
            "text": out["text"]}


def _packed(seed: int, n: int = PER_SEED) -> dict:
    """samples_for_seed with the images compressed, for sending between
    processes: the camera images are mostly flat colour and shrink a lot,
    and unpacking them does not hold up the training thread."""
    d = samples_for_seed(seed, n)
    d["images"] = (zlib.compress(d["images"].tobytes(), 1), d["images"].shape)
    return d


def _unpacked(d: dict) -> dict:
    data, shape = d["images"]
    d["images"] = np.frombuffer(zlib.decompress(data), np.uint8).reshape(shape)
    return d


def _merge(parts: list[dict], order: np.ndarray | None = None) -> dict:
    d = {k: (np.concatenate([p[k] for p in parts]) if k != "text"
             else list(itertools.chain.from_iterable(p["text"] for p in parts)))
         for k in parts[0]}
    if order is not None:
        d = {k: (v[order] if k != "text" else [v[i] for i in order])
             for k, v in d.items()}
    return d


def fixed_set(seed0: int, n_seeds: int, workers: int = 8) -> dict:
    """All samples of seeds seed0 .. seed0+n_seeds-1 (e.g. a validation set)."""
    with mp.get_context("spawn").Pool(workers) as pool:
        return _merge([_unpacked(d) for d in pool.map(_packed, range(seed0, seed0 + n_seeds))])


def stream(seed0: int, batch_size: int, workers: int = 16, block: int = 32,
           n_seeds: int | None = None, per_seed: int = PER_SEED,
           skip_blocks: int = 0):
    """Endless batches from seeds seed0, seed0+1, ...  Each block of `block`
    seeds is generated in parallel and shuffled with a seed-derived
    permutation, so the batch sequence depends only on (seed0, batch_size,
    block) — not on the number of workers or on timing.  `n_seeds` restricts
    the stream to that many seeds, repeated (for overfitting checks);
    `skip_blocks` starts later in the stream (to resume a run)."""
    q: queue.Queue = queue.Queue(maxsize=8)
    if n_seeds is not None:
        block = min(block, n_seeds)

    def produce():
        with mp.get_context("spawn").Pool(workers) as pool, ThreadPool(8) as unpackers:
            def launch(b):          # start generating block b
                seeds = [seed0 + (b * block + i) % (n_seeds or 2 ** 62) for i in range(block)]
                return pool.map_async(functools.partial(_packed, n=per_seed), seeds)

            ahead = max(1, workers // block)     # blocks being generated in advance
            jobs = {b: launch(b) for b in range(skip_blocks, skip_blocks + ahead)}
            for b in itertools.count(skip_blocks):
                jobs[b + ahead] = launch(b + ahead)
                d = _merge(unpackers.map(_unpacked, jobs.pop(b).get()))
                n = len(d["text"])
                order = np.random.default_rng(seed0 + b * block).permutation(n)
                for i in range(0, n - batch_size + 1, batch_size):
                    idx = order[i:i + batch_size]
                    q.put({k: (v[idx] if k != "text" else [v[j] for j in idx])
                           for k, v in d.items()})

    threading.Thread(target=produce, daemon=True).start()
    while True:
        yield q.get()
