# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""amp-bench: score a policy, or look at a scene.

    amp-bench eval --policy teacher                 # 300 held-out scenes, the standard
    amp-bench eval --policy path/to/package --json result.json
    amp-bench eval --policy my_module:make_policy
    amp-bench show --seed 9000007 --out scene.png
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _eval(args) -> int:
    from . import evaluate, policy
    backend = policy.load_policy(args.policy, args.device)
    print(f"policy: {backend.name}", file=sys.stderr)
    res = evaluate.evaluate(
        backend, args.scenes, args.seed0,
        progress=lambda i, n: print(f"\r  scene {i}/{n}", end="", file=sys.stderr))
    print(file=sys.stderr)
    if args.grounding and isinstance(backend, policy.PolicyBackend):
        res.update(evaluate.grounding(backend.policy, args.probe_scenes))
    print(evaluate.report(res))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"policy": args.policy, **res}, f, indent=1)
    return 0


def _show(args) -> int:
    from PIL import Image
    from . import cameras, core
    scene = core.new_scene(args.seed)
    main = cameras.render_scene(scene.q, scene.objects, size=512)
    views = cameras.render_views(scene.q, scene.objects, size=256)
    sheet = Image.new("RGB", (512 + 256, 768), (10, 12, 20))
    sheet.paste(main, (0, 128))
    for i, name in enumerate(cameras.VIEW_NAMES):
        sheet.paste(views[name], (512, 256 * i))
    sheet.save(args.out)
    print(f"scene {args.seed}: {', '.join(scene.object_names)} -> {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="amp-bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="closed-loop success and grounding tests")
    e.add_argument("--policy", required=True,
                   help='"teacher", a model package folder, or "module:callable"')
    e.add_argument("--scenes", type=int, default=300,
                   help="held-out scenes; 300 is the standard for reported results")
    e.add_argument("--seed0", type=int, default=None)
    e.add_argument("--probe-scenes", type=int, default=100)
    e.add_argument("--no-grounding", dest="grounding", action="store_false")
    e.add_argument("--device", default="cpu")
    e.add_argument("--json", default=None, help="write all results, per clause, here")
    s = sub.add_parser("show", help="render a scene and its three cameras")
    s.add_argument("--seed", type=int, default=9_000_007)
    s.add_argument("--out", default="scene.png")
    args = ap.parse_args(argv)
    if args.cmd == "eval":
        from . import evaluate
        args.seed0 = evaluate.EVAL_SEED0 if args.seed0 is None else args.seed0
        return _eval(args)
    return _show(args)


if __name__ == "__main__":
    sys.exit(main())
