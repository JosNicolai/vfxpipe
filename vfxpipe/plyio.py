"""Tiny binary PLY reader (vertex properties only, faces optional) — enough to
read 3DGS .ply files and our own point clouds without a dependency."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
          "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
          "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
          "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (vertices structured array, faces (F,3) int array or None)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = next(l for l in header if l.startswith("format")).split()[1]
        if fmt == "ascii":
            raise ValueError("ASCII PLY not supported — export binary")
        endian = "<" if "little" in fmt else ">"
        elements: list[tuple[str, int, list]] = []
        for line in header:
            p = line.split()
            if p and p[0] == "element":
                elements.append((p[1], int(p[2]), []))
            elif p and p[0] == "property":
                if p[1] == "list":
                    elements[-1][2].append(("list", _TYPES[p[2]].replace("<", endian), _TYPES[p[3]].replace("<", endian), p[4]))
                else:
                    elements[-1][2].append((p[2], _TYPES[p[1]].replace("<", endian)))
        verts, faces = None, None
        for name, count, props in elements:
            if name == "vertex":
                dt = np.dtype([(n, t) for n, t in props])
                verts = np.frombuffer(f.read(dt.itemsize * count), dtype=dt)
            elif name == "face" and props and props[0][0] == "list":
                _, ct, it, _ = props[0]
                cnt_size, idx_size = np.dtype(ct).itemsize, np.dtype(it).itemsize
                out = []
                for _ in range(count):
                    n = int(np.frombuffer(f.read(cnt_size), dtype=ct)[0])
                    out.append(np.frombuffer(f.read(idx_size * n), dtype=it)[:3])
                faces = np.array(out, dtype=np.int64) if out else None
            else:  # skip unknown element
                size = sum(np.dtype(t).itemsize for n, t in props if n != "list")
                f.read(size * count)
    if verts is None:
        raise ValueError("PLY has no vertex element")
    return verts, faces


def splat_to_points(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From a 3DGS vertex array return xyz (N,3), rgb 0-1 (N,3), opacity (N,), scale (N,3) in world units."""
    names = verts.dtype.names or ()
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    if "f_dc_0" in names:
        dc = np.stack([verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"]], axis=1).astype(np.float64)
        rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0, 1)
    elif "red" in names:
        rgb = np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.float64) / 255.0
    else:
        rgb = np.full_like(xyz, 0.7)
    if "opacity" in names:
        op = 1.0 / (1.0 + np.exp(-verts["opacity"].astype(np.float64)))
    else:
        op = np.ones(len(xyz))
    if "scale_0" in names:
        sc = np.exp(np.stack([verts["scale_0"], verts["scale_1"], verts["scale_2"]], axis=1).astype(np.float64))
    else:
        sc = np.full_like(xyz, 0.01)
    return xyz, rgb, op, sc


def write_mesh_ply(path: str | Path, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray | None = None) -> None:
    n, m = len(verts), len(faces)
    with open(path, "wb") as f:
        hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
               "property float x", "property float y", "property float z"]
        if colors is not None:
            hdr += ["property uchar red", "property uchar green", "property uchar blue"]
        hdr += [f"element face {m}", "property list uchar int vertex_indices", "end_header"]
        f.write(("\n".join(hdr) + "\n").encode())
        if colors is not None:
            rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec["r"], rec["g"], rec["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
        else:
            rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        rec["x"], rec["y"], rec["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
        f.write(rec.tobytes())
        frec = np.empty(m, dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        frec["n"] = 3
        frec["a"], frec["b"], frec["c"] = faces[:, 0], faces[:, 1], faces[:, 2]
        f.write(frec.tobytes())
