"""Read COLMAP/GLOMAP text models (cameras.txt, images.txt, points3D.txt).

Conventions (COLMAP):  X_cam = R @ X_world + t,  camera looks down +Z, +Y is
image-down.  Camera centre C = -R^T t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: list[float]

    def K(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) — for undistorted/pinhole use."""
        m, p = self.model, self.params
        if m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return p[0], p[0], p[1], p[2]
        return p[0], p[1], p[2], p[3]  # PINHOLE, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, ...


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # w x y z
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    def center(self) -> np.ndarray:
        return -self.R().T @ self.tvec


@dataclass
class Model:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    xyz: np.ndarray          # (N,3)
    rgb: np.ndarray          # (N,3) uint8
    error: np.ndarray        # (N,)
    ids: np.ndarray          # (N,) point3D ids

    def id_index(self) -> dict[int, int]:
        return {int(i): k for k, i in enumerate(self.ids)}


def qvec2rotmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def read_model(path: str | Path) -> Model:
    path = Path(path)
    cameras: dict[int, Camera] = {}
    for line in (path / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cameras[int(parts[0])] = Camera(int(parts[0]), parts[1], int(parts[2]), int(parts[3]),
                                        [float(x) for x in parts[4:]])
    images: dict[int, Image] = {}
    lines = [l for l in (path / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        h = lines[i].split()
        img = Image(int(h[0]), np.array([float(x) for x in h[1:5]]), np.array([float(x) for x in h[5:8]]),
                    int(h[8]), h[9])
        if i + 1 < len(lines):
            pts = lines[i + 1].split()
            if pts:
                arr = np.array([float(x) for x in pts]).reshape(-1, 3)
                img.xys = arr[:, :2]
                img.point3D_ids = arr[:, 2].astype(np.int64)
        images[img.id] = img
    ids, xyz, rgb, err = [], [], [], []
    p3 = path / "points3D.txt"
    if p3.exists():
        for line in p3.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ids.append(int(parts[0]))
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
            err.append(float(parts[7]))
    return Model(cameras, images,
                 np.array(xyz, dtype=np.float64).reshape(-1, 3),
                 np.array(rgb, dtype=np.uint8).reshape(-1, 3),
                 np.array(err, dtype=np.float64),
                 np.array(ids, dtype=np.int64))


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def floor_up(xyz: np.ndarray, prior_up: np.ndarray, tol: float, max_angle_deg: float = 60.0) -> tuple[np.ndarray | None, int]:
    """RANSAC the largest plane whose normal is within max_angle of prior_up;
    return its normal (pointing the prior's way) and inlier count.  Uses Open3D
    if present, else a small numpy RANSAC."""
    if len(xyz) < 200:
        return None, 0
    prior_up = prior_up / (np.linalg.norm(prior_up) + 1e-12)
    best_n, best_cnt = None, 0
    rest = xyz.copy()
    for _ in range(6):                       # peel a few planes, keep the best horizontal one
        if len(rest) < 200:
            break
        n, d, inl = _ransac_plane(rest, tol)
        if inl is None or inl.sum() < 100:
            break
        if np.dot(n, prior_up) < 0:
            n = -n
        ang = np.degrees(np.arccos(np.clip(np.dot(n, prior_up), -1, 1)))
        if ang <= max_angle_deg and inl.sum() > best_cnt:
            best_n, best_cnt = n, int(inl.sum())
        rest = rest[~inl]
    return best_n, best_cnt


def _ransac_plane(P: np.ndarray, tol: float, iters: int = 400):
    rng = np.random.default_rng(0)
    best = (None, None, None); best_cnt = 0
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False)
        a, b, c = P[i]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn; d = -np.dot(n, a)
        inl = np.abs(P @ n + d) < tol
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt; best = (n, d, inl)
    if best[0] is None:
        return None, None, None
    # refine on inliers (least squares)
    Q = P[best[2]]; c0 = Q.mean(0); w, v = np.linalg.eigh(np.cov((Q - c0).T)); n = v[:, 0]
    d = -np.dot(n, c0); inl = np.abs(P @ n + d) < tol
    return n, d, inl


def world_transform(model: Model, auto_orient: bool, scale: float, mode: str = "floor",
                    manual_up=None) -> dict:
    """Rotation M, centre c and scale s so that  X_out = s * M @ (X - c)  is a
    Z-up, roughly metric, roughly centred version of the COLMAP world.

    'Up' is estimated from the cameras: the average of each camera's -Y axis in
    world space.  This assumes the camera was held roughly level, which is true
    for almost all handheld / gimbal footage."""
    ups = np.array([img.R().T @ np.array([0.0, -1.0, 0.0]) for img in model.images.values()])
    cam_up = ups.mean(axis=0)
    up, source, n_inl = cam_up, "camera", 0
    if manual_up is not None and np.linalg.norm(manual_up) > 1e-6:
        up, source = np.asarray(manual_up, dtype=float), "manual"
    elif mode == "floor" and len(model.xyz) >= 200:
        centres = np.array([img.center() for img in model.images.values()])
        span = float(np.linalg.norm(centres.max(0) - centres.min(0))) + 1e-6
        n, n_inl = floor_up(model.xyz, cam_up, tol=0.01 * max(span, 1e-3))
        if n is not None:
            up, source = n, "floor plane"
    M = rotation_aligning(up, np.array([0.0, 0.0, 1.0])) if auto_orient and np.linalg.norm(up) > 1e-6 else np.eye(3)
    centre = np.median(model.xyz, axis=0) if len(model.xyz) else np.zeros(3)
    tilt = float(np.degrees(np.arccos(np.clip(np.dot(up / np.linalg.norm(up), cam_up / (np.linalg.norm(cam_up) + 1e-12)), -1, 1))))
    return {"M": M.tolist(), "centre": centre.tolist(), "scale": float(scale),
            "estimated_up": (up / np.linalg.norm(up)).tolist(), "up_source": source, "floor_points": int(n_inl),
            "camera_tilt_deg": round(tilt, 1)}


def load_transform(project) -> dict:
    """Effective world transform: mesh/orientation.json (refined from the
    splat planes) wins over the one in track/cameras.json."""
    import json
    o = project.dir("mesh") / "orientation.json"
    if o.exists():
        return json.loads(o.read_text())
    return json.loads((project.dir("track") / "cameras.json").read_text())["transform"]
