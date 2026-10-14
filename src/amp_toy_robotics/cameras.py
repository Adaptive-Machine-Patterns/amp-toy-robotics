# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""The cameras: a headless PIL renderer of the arm and the objects.

Uses the FK and gripper geometry of `geometry.py` and adds:
  * free-orbit camera (azimuth/elevation) with automatic framing,
  * vertical gradient background + soft vignette,
  * painter's-algorithm depth sorting with depth shading,
  * cylindrical link highlights, spherical joints with specular dot,
  * ground shadows (base plate, objects, tip),
  * gripper open/close (finger spacing driven by `grip` in [0,1]),
  * fading grasp-point trail (ends between the fingers, 2/3 down the
    finger length, where objects are picked up; geometry shared with
    core.py),
  * top/side/iso strip views.

`observation` is the model's image input: the three fixed cameras.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from . import core  # shared gripper spacing (single source of truth)
from . import geometry as geo

# ── color helpers ────────────────────────────────────────────────────
def mix(c1: Sequence[int], c2: Sequence[int], t: float) -> Tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))  # type: ignore

def lighten(c: Sequence[int], t: float) -> Tuple[int, int, int]:
    return mix(c, (255, 255, 255), t)

def darken(c: Sequence[int], t: float) -> Tuple[int, int, int]:
    return mix(c, (10, 12, 18), t)

# ── projection ───────────────────────────────────────────────────────
def rotate(pts: np.ndarray, az_deg: float, el_deg: float) -> np.ndarray:
    """World (N,3) -> (x2, z2, depth) rows.  depth: larger = farther."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    az, el = math.radians(az_deg), math.radians(el_deg)
    ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    x1 = ca * x - sa * y
    y1 = sa * x + ca * y
    x2 = x1
    z2 = se * y1 + ce * z
    d = ce * y1 - se * z
    return np.stack([x2, z2, d], axis=1)


class Camera:
    """Free-orbit camera with automatic framing of a reference point set."""

    def __init__(self, az: float, el: float, size: int, ref_pts: np.ndarray,
                 pad: float = 0.12):
        self.az, self.el, self.size = az, el, size
        r = rotate(ref_pts, az, el)
        minx, maxx = r[:, 0].min(), r[:, 0].max()
        minz, maxz = r[:, 1].min(), r[:, 1].max()
        bw = max(maxx - minx, 1e-6)
        bh = max(maxz - minz, 1e-6)
        self.s = min(size / bw, size / bh) * (1.0 - 2 * pad)
        self.cx = 0.5 * (minx + maxx)
        self.cz = 0.5 * (minz + maxz)
        self.dmin, self.dmax = float(r[:, 2].min()), float(r[:, 2].max())

    def project(self, pts) -> np.ndarray:
        """(N,3) world -> (N,2) screen px (this camera's az/el)."""
        r = rotate(np.asarray(pts, dtype=np.float64).reshape(-1, 3), self.az, self.el)
        sx = self.size / 2.0 + self.s * (r[:, 0] - self.cx)
        sy = self.size / 2.0 - self.s * (r[:, 1] - self.cz)
        return np.stack([sx, sy], axis=1)

    def depth_t(self, depth: float) -> float:
        span = max(self.dmax - self.dmin, 1e-6)
        return min(1.0, max(0.0, (depth - self.dmin) / span))


def _link_color(t: float) -> Tuple[int, int, int]:
    """t: 0 near (light steel) -> 1 far (dark blue-grey)."""
    return mix((188, 202, 232), (74, 86, 118), t)


# ── ground-plane circle (perspective ellipse) ────────────────────────
def ground_circle(cam: Camera, cx: float, cy: float, radius: float,
                  n: int = 24) -> List[Tuple[float, float]]:
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th),
                    np.zeros(n)], axis=1)
    return [(float(x), float(y)) for x, y in cam.project(pts)]


# ── shapes ───────────────────────────────────────────────────────────
def _poly(draw: ImageDraw.ImageDraw, pts, **kw):
    draw.polygon(pts, **kw)


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: float, cy: float,
               r: float, fill, edge):
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=edge, width=max(2, int(r * 0.12)))
    elif shape == "square":
        h = r * 0.9
        draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=fill, outline=edge, width=max(2, int(r * 0.12)))
    elif shape == "triangle":
        h = r * 1.1
        pts = [(cx, cy - h), (cx - h * 0.95, cy + h * 0.7), (cx + h * 0.95, cy + h * 0.7)]
        _poly(draw, pts, fill=fill, outline=edge)
    elif shape == "star":
        pts = []
        for i in range(10):
            a = i * math.pi / 5 - math.pi / 2
            rad = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        _poly(draw, pts, fill=fill, outline=edge)
    elif shape == "hexagon":
        pts = []
        for i in range(6):
            a = i * math.pi / 3 - math.pi / 6
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        _poly(draw, pts, fill=fill, outline=edge)
    else:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=edge)


# ── main render ──────────────────────────────────────────────────────
def _ref_points(q: np.ndarray, objects: List[dict]) -> np.ndarray:
    pts3d, _, _ = geo.arm_points(np.asarray(q).tolist())
    ref = [[-125, -125, 0], [125, -125, 0], [125, 125, 0], [-125, 125, 0]]
    ref += [list(p) for p in pts3d]
    for o in objects or []:
        ref.append(list(o["pos"]))
    return np.asarray(ref, dtype=np.float64)


# Fixed framing for the model's cameras: a box that contains everything the
# arm can reach, so a world point always lands on the same pixel no matter
# where the arm or the objects are.
FIXED_REF = np.array([[x, y, z] for x in (-165.0, 165.0) for y in (-165.0, 165.0)
                      for z in (-20.0, 205.0)])


_BACKDROPS: dict = {}


def render_scene(q: Sequence[float], objects: Optional[List[dict]] = None,
                 az: float = 40.0, el: float = 28.0, size: int = 512,
                 grip: float = 0.0, trail: Optional[Sequence[Sequence[float]]] = None,
                 show_grid: bool = True, show_axes: bool = True,
                 fixed: bool = False) -> Image.Image:
    """Render the robot + objects from a free-orbit camera. Returns RGB PIL image.

    `fixed` frames the fixed workspace box instead of auto-framing the
    scene (used for the model's cameras).
    """
    q = np.asarray(q, dtype=np.float64).reshape(6)
    objects = objects or []
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    cam = (Camera(az, el, size, FIXED_REF, pad=0.03) if fixed
           else Camera(az, el, size, _ref_points(q, objects)))

    # background + grid: identical for every frame of a fixed camera, so it
    # is drawn once per camera and copied
    key = (size, az, el, show_grid, show_axes) if fixed else None
    if key in _BACKDROPS:
        img = _BACKDROPS[key].copy()
    else:
        # ── background: vertical gradient + vignette ──
        top_c, bot_c = (30, 38, 64), (10, 12, 20)
        grad = np.linspace(0, 1, size).reshape(-1, 1)
        rows = (np.array(top_c)[None, :] * (1 - grad) + np.array(bot_c)[None, :] * grad)
        bg = np.tile(np.round(rows).astype(np.uint8)[:, None, :], (1, size, 1))
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy = size / 2, size * 0.55
        rr_ = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size * 0.75)
        vig = np.clip(1.0 - 0.35 * np.clip(rr_ - 0.45, 0, 1) ** 2 * 2.2, 0.55, 1.0)
        img = Image.fromarray((bg * vig[..., None]).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        # ── grid ──
        if show_grid:
            grid_c = (44, 54, 82)
            ext, step = 125, 25
            for v in range(-ext, ext + 1, step):
                p1, p2 = cam.project([[v, -ext, 0], [v, ext, 0]])
                draw.line([tuple(p1), tuple(p2)], fill=grid_c, width=1)
                p1, p2 = cam.project([[-ext, v, 0], [ext, v, 0]])
                draw.line([tuple(p1), tuple(p2)], fill=grid_c, width=1)
            if show_axes:
                p0 = cam.project([[0, 0, 0]])[0]
                px = cam.project([[125, 0, 0]])[0]
                py = cam.project([[0, 125, 0]])[0]
                pz = cam.project([[0, 0, 90]])[0]
                draw.line([tuple(p0), tuple(px)], fill=(96, 58, 58), width=2)
                draw.line([tuple(p0), tuple(py)], fill=(58, 84, 60), width=2)
                draw.line([tuple(p0), tuple(pz)], fill=(58, 62, 96), width=2)
        if key is not None:
            _BACKDROPS[key] = img.copy()
    draw = ImageDraw.Draw(img)
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # ── ground shadows (always behind arm/objects) ──
    base_p = cam.project([[0, 0, 0]])[0]
    odraw.polygon(ground_circle(cam, 0, 0, 20), fill=(0, 0, 0, 95))
    odraw.polygon(ground_circle(cam, 0, 0, 14), fill=(24, 30, 46, 160))
    for o in objects:
        ox, oy, _oz = o["pos"]
        odraw.polygon(ground_circle(cam, ox, oy, 11), fill=(0, 0, 0, 70))
    if trail is not None:
        tl = np.asarray(trail, dtype=np.float64).reshape(-1, 3)
        if len(tl):
            lx, ly, _lz = tl[-1]
            odraw.polygon(ground_circle(cam, lx, ly, 7), fill=(0, 0, 0, 55))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # ── arm geometry ──
    pts3d, last_pitch, base_yaw = geo.arm_points(q.tolist())
    pts3d = np.asarray(pts3d, dtype=np.float64)
    R = rotate(pts3d, az, el)
    P = cam.project(pts3d)

    w_post = max(4, int(size * 0.028))
    w_link = max(3, int(size * 0.020))
    r_base = size * 0.032
    r_joint = size * 0.020

    # trail (behind arm links, above shadows)
    if trail is not None:
        tl = np.asarray(trail, dtype=np.float64).reshape(-1, 3)
        if len(tl) > 1:
            TP = cam.project(tl)
            n = len(TP) - 1
            for i in range(n):
                a = 40 + int(170 * i / max(1, n))
                wd = max(2, int(size * 0.005 * (0.4 + 0.6 * i / max(1, n))))
                odraw.line([tuple(TP[i]), tuple(TP[i + 1])],
                           fill=(96, 205, 255, a), width=wd)

    # gripper geometry (spacing driven by grip; shared with core.grasp_point)
    spacing = core.gripper_spacing(grip)                 # center-to-center
    gg = geo.gripper_points(
        tuple(pts3d[-1]), last_pitch, base_yaw,
        core.FINGER_LEN_DISPLAY, spacing)
    grip_pts = np.array([gg[k] for k in
                         ("base", "left_base", "right_base",
                          "left_tip", "right_tip")])
    GR = rotate(grip_pts, az, el)
    GP = cam.project(grip_pts)

    drawables = []  # (depth, painter_fn)

    def seg(i: int, w: int, base: Tuple[int, int, int], hi_t: float = 0.35):
        d = 0.5 * (R[i, 2] + R[i + 1, 2])
        t = cam.depth_t(d)
        col = mix(base, _link_color(t), 0.75) if base else _link_color(t)
        def _fn(d=d, i=i, w=w, col=col):
            draw.line([tuple(P[i]), tuple(P[i + 1])], fill=col, width=w)
            draw.line([tuple(P[i]), tuple(P[i + 1])],
                      fill=lighten(col, hi_t), width=max(1, w // 3))
            for j in (i, i + 1):
                draw.ellipse([P[j, 0] - w / 2, P[j, 1] - w / 2,
                              P[j, 0] + w / 2, P[j, 1] + w / 2], fill=col)
        drawables.append((d, _fn))

    def joint(i: int, r: float, base: Optional[Tuple[int, int, int]] = None):
        d = R[i, 2]
        t = cam.depth_t(d)
        col = base if base else _link_color(t)
        def _fn(i=i, r=r, col=col):
            c = P[i]
            draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r],
                         fill=col, outline=darken(col, 0.55), width=max(2, int(r * 0.18)))
            sr = r * 0.38
            sc = (c[0] - r * 0.3, c[1] - r * 0.35)
            draw.ellipse([sc[0] - sr, sc[1] - sr, sc[0] + sr, sc[1] + sr],
                         fill=lighten(col, 0.45))
        drawables.append((d, _fn))

    def grip_line(a: int, b: int, w: int, col, hi: bool = False):
        d = 0.5 * (GR[a, 2] + GR[b, 2])
        def _fn(a=a, b=b, w=w, col=col, hi=hi):
            draw.line([tuple(GP[a]), tuple(GP[b])], fill=col, width=w)
            if hi:
                draw.line([tuple(GP[a]), tuple(GP[b])],
                          fill=lighten(col, 0.45), width=max(1, w // 3))
        drawables.append((d, _fn))

    # base post + links
    seg(0, w_post, (110, 118, 138))
    for i in (1, 2, 3):
        seg(i, w_link, None)
    joints = [(0, r_base, (140, 148, 168))]
    for i in (1, 2, 3):
        joints.append((i, r_joint, None))
    for (i, r, c) in joints:
        joint(i, r, c)

    # gripper: crossbar + two LONG fingers (finger_base -> finger_tip)
    # grip_pts indices: 0=base 1=finger_L_base 2=finger_R_base
    #                   3=finger_L_tip  4=finger_R_tip
    gcol = (150, 190, 225)
    w_f = max(4, int(size * 0.016))
    d = GR[0, 2]
    grip_line(1, 2, max(3, int(size * 0.012)), gcol)          # crossbar
    grip_line(1, 3, w_f, gcol, hi=True)                       # left finger
    grip_line(2, 4, w_f, gcol, hi=True)                       # right finger

    # objects
    for o in objects:
        d = rotate(np.array(o["pos"]), az, el)[0, 2]
        c = cam.project([o["pos"]])[0]
        r = max(6.0, size * 0.016 * 1.25)
        rgb = o["rgb"]
        def _obj(c=c, r=r, rgb=rgb, shape=o["shape"]):
            draw_shape(draw, shape, c[0], c[1], r, fill=rgb,
                       edge=darken(rgb, 0.45))
            # specular: a smaller copy of the SAME shape, shifted up-left.
            # (A fixed ellipse pokes out of stars/triangles; a scaled
            # copy of the shape provably stays inside it.)
            hi_c = lighten(rgb, 0.5)
            draw_shape(draw, shape, c[0] - r * 0.16, c[1] - r * 0.20,
                       r * 0.34, fill=hi_c, edge=hi_c)
        drawables.append((d, _obj))

    # painter's algorithm: far first
    for _, fn in sorted(drawables, key=lambda x: -x[0]):
        fn()

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img


# ── the model's cameras ──────────────────────────────────────────────
VIEW_CAMS = {"top": (0.0, 89.0), "side": (0.0, 5.0), "iso": (45.0, 35.264)}
VIEW_NAMES = ("top", "side", "iso")
OBS_SIZE = 512      # rendered at the resolution SmolVLA's vision encoder uses


def render_views(q: Sequence[float], objects: Optional[List[dict]] = None,
                 size: int = 224, grip: float = 0.0) -> Dict[str, Image.Image]:
    """The three fixed cameras (top / side / iso).  These are exactly what
    the model sees."""
    return {name: render_scene(q, objects, az, el, size=size, grip=grip,
                               show_grid=True, show_axes=False, fixed=True)
            for name, (az, el) in VIEW_CAMS.items()}


def observation(q: Sequence[float], objects: List[dict], grip: float,
                size: int = OBS_SIZE) -> np.ndarray:
    """The model's image input: uint8 (3 views, 3 channels, H, W).  The one
    function used by data generation, evaluation and every policy."""
    views = render_views(q, objects, size=size, grip=grip)
    return np.stack([np.asarray(views[v], dtype=np.uint8).transpose(2, 0, 1)
                     for v in VIEW_NAMES])
