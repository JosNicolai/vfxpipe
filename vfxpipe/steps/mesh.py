"""Step 4 — a proxy mesh of the environment.

Not a hero asset: this is the surface the character stands on, casts shadows
onto, is occluded by, and deforms (the couch cushion).  Poisson reconstruction
over the splat centres gives a watertight-ish surface that is good enough for
shadow catching; you clean up the parts that matter in Blender.

Runs on CPU only (Open3D).  A few minutes for a room.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..runner import Step, Context, register, StepError
from ..plyio import read_ply, splat_to_points, write_mesh_ply

PREVIEW_FACES = 20_000


@register
class Mesh(Step):
    name = "mesh"
    label = "Mesh"
    depends = ["track"]      # splat is optional: falls back to sparse points
    params = ["mesh_mode", "mesh_source", "mesh_min_opacity", "mesh_max_scale_pct", "poisson_depth", "mesh_density_trim",
              "mesh_target_faces", "plane_max", "plane_min_inliers", "plane_tolerance", "dense_max_size", "dense_trim"]
    outputs = ["mesh.ply", "mesh_preview.json", "planes.json", "orientation.json"]

    def extra_hash(self, project):
        from ..runner import load_status
        if project.settings.get("mesh_source") == "splat":
            return "splat:" + load_status(project, "splat").get("output_hash", "")
        return ""

    def preflight(self, project):
        try:
            import open3d  # noqa: F401
        except ImportError:
            return ["open3d not installed:  pip install open3d"]
        mode = project.settings.get("mesh_mode", "planes")
        if mode == "colmap_dense":
            if not project.tool("colmap"):
                return ["colmap not found"]
            if not project.has_nvidia():
                return ["colmap_dense needs COLMAP's CUDA build and an NVIDIA GPU (patch-match stereo has no CPU path)"]
            return []
        if project.settings.get("mesh_source") == "splat" and not (project.dir("splat") / "splat.ply").exists():
            return ["mesh_source is 'splat' but splat/splat.ply doesn't exist yet — run Splat first or set mesh_source to 'sparse'"]
        return []

    def run(self, ctx: Context):
        import open3d as o3d
        s = ctx.settings
        for f in ("planes.json", "orientation.json"):
            if (ctx.out / f).exists():
                (ctx.out / f).unlink()
        if s.get("mesh_mode", "planes") == "colmap_dense":
            return self._colmap_dense(ctx)
        cams = json.loads((ctx.project.dir("track") / "cameras.json").read_text())
        centres = np.array([f["C"] for f in cams["frames"] if f.get("solved")])

        if s.get("mesh_source") == "splat":
            verts, _ = read_ply(ctx.project.dir("splat") / "splat.ply")
            xyz, rgb, op, sc = splat_to_points(verts)
            keep = (op >= float(s.get("mesh_min_opacity", 0.6))) & \
                   (sc.max(axis=1) <= np.percentile(sc.max(axis=1), float(s.get("mesh_max_scale_pct", 90))))
            xyz, rgb = xyz[keep], rgb[keep]
            ctx.log(f"{keep.sum()} of {len(keep)} gaussians kept as surface samples")
        else:
            verts, _ = read_ply(ctx.project.dir("track") / "points.ply")
            xyz = np.stack([verts["x"], verts["y"], verts["z"]], 1).astype(np.float64)
            rgb = np.stack([verts["red"], verts["green"], verts["blue"]], 1) / 255.0
            ctx.log(f"{len(xyz)} sparse points")
        if len(xyz) < 500:
            raise StepError("too few points to reconstruct a surface")

        # keep only what's within a sensible distance of the camera path
        span = np.linalg.norm(centres.max(0) - centres.min(0)) + 1e-6
        d = np.min(np.linalg.norm(xyz[:, None, :] - centres[None, ::max(1, len(centres) // 50), :], axis=2), axis=1)
        keep = d < max(span * 4, np.percentile(d, 95))
        xyz, rgb = xyz[keep], rgb[keep]

        ctx.set_progress(0.1, "cleaning point cloud")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.voxel_down_sample(voxel_size=float(span) / 600.0)
        ctx.log(f"{len(pcd.points)} points after cleanup")
        if s.get("mesh_mode", "planes") == "planes":
            return self._planes(ctx, pcd, centres, span)

        ctx.set_progress(0.25, "estimating normals")
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        P = np.asarray(pcd.points)
        N = np.asarray(pcd.normals)
        # orient each normal towards the nearest camera (surfaces face the camera that saw them)
        step = max(1, len(centres) // 60)
        C = centres[::step]
        near = np.argmin(np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2), axis=1)
        to_cam = C[near] - P
        flip = np.einsum("ij,ij->i", to_cam, N) < 0
        N[flip] *= -1
        pcd.normals = o3d.utility.Vector3dVector(N)

        ctx.set_progress(0.4, "Poisson reconstruction")
        depth = int(s.get("poisson_depth", 9))
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        dens = np.asarray(dens)
        trim = float(s.get("mesh_density_trim", 0.02))
        if trim > 0:
            mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim))
        # crop to the point cloud's box (Poisson balloons beyond it)
        bb = pcd.get_axis_aligned_bounding_box()
        bb = o3d.geometry.AxisAlignedBoundingBox(bb.min_bound - 0.05 * span, bb.max_bound + 0.05 * span)
        mesh = mesh.crop(bb)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        ctx.log(f"poisson: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")

        ctx.set_progress(0.75, "simplifying")
        target = int(s.get("mesh_target_faces", 200000))
        if len(mesh.triangles) > target:
            mesh = mesh.simplify_quadric_decimation(target)
        V = np.asarray(mesh.vertices)
        F = np.asarray(mesh.triangles)
        col = None
        if mesh.has_vertex_colors():
            col = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        write_mesh_ply(ctx.out / "mesh.ply", V, F, col)

        ctx.set_progress(0.9, "writing preview")
        prev = mesh.simplify_quadric_decimation(PREVIEW_FACES) if len(F) > PREVIEW_FACES else mesh
        PV = np.asarray(prev.vertices)
        PF = np.asarray(prev.triangles)
        edges = np.concatenate([PF[:, [0, 1]], PF[:, [1, 2]], PF[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        (ctx.out / "mesh_preview.json").write_text(json.dumps({
            "xyz": PV.round(4).tolist(), "edges": edges.tolist(),
            "faces": int(len(F)), "verts": int(len(V)),
        }))
        return {"verts": int(len(V)), "faces": int(len(F)), "source": s.get("mesh_source")}


    # ---- plane fitting -----------------------------------------------------------
    def _planes(self, ctx: Context, pcd, centres: np.ndarray, span: float) -> dict:
        """RANSAC planes, largest first.  Each becomes a quad clipped to its
        inliers' extent, exported as its own object by the Blender script."""
        import open3d as o3d
        s = ctx.settings
        from ..colmap_io import load_transform, rotation_aligning
        T = load_transform(ctx.project)
        M = np.array(T["M"]); prior_up = np.array(T["estimated_up"])
        tol = float(s.get("plane_tolerance", 1.0)) / 100.0 * span
        min_frac = float(s.get("plane_min_inliers", 3.0)) / 100.0
        max_planes = int(s.get("plane_max", 8))
        rest = pcd
        total = len(rest.points)
        planes = []
        for k in range(max_planes):
            if len(rest.points) < max(200, total * min_frac):
                break
            model, inl = rest.segment_plane(distance_threshold=tol, ransac_n=3, num_iterations=2000)
            if len(inl) < total * min_frac:
                break
            P = np.asarray(rest.points)[inl]
            C = np.asarray(rest.colors)[inl] if rest.has_colors() else None
            n = np.array(model[:3]); n /= np.linalg.norm(n)
            # orient normal towards the cameras
            if np.dot(centres.mean(0) - P.mean(0), n) < 0:
                n = -n
            # in-plane axes from PCA of the inliers, quad = 2nd/98th percentile extents
            c0 = P.mean(0)
            Q = P - c0
            Qp = Q - np.outer(Q @ n, n)                     # flatten onto the plane
            w_, vec = np.linalg.eigh(Qp.T @ Qp / len(Qp))    # principal in-plane direction
            u = vec[:, np.argmax(w_)]; u = u - n * np.dot(u, n); u /= np.linalg.norm(u)
            v = np.cross(n, u)
            a = Q @ u; b = Q @ v
            a0, a1 = np.percentile(a, [1, 99]); b0, b1 = np.percentile(b, [1, 99])
            corners = [c0 + u * a0 + v * b0, c0 + u * a1 + v * b0, c0 + u * a1 + v * b1, c0 + u * a0 + v * b1]
            planes.append({"normal": n.tolist(), "centre": c0.tolist(),
                           "corners": [c.tolist() for c in corners], "points": int(len(inl)),
                           "size": [float(a1 - a0), float(b1 - b0)],
                           "color": (C.mean(0).tolist() if C is not None else [0.5, 0.5, 0.5])})
            ctx.log(f"plane {k + 1}: {len(inl)} points, {a1 - a0:.2f} x {b1 - b0:.2f} units")
            rest = rest.select_by_index(inl, invert=True)
            ctx.set_progress(0.2 + 0.6 * (k + 1) / max_planes, f"{len(planes)} planes")
        if not planes:
            raise StepError("no planes found — lower plane_min_inliers or raise plane_tolerance")

        # --- orientation: the largest plane within 45° of the current up is the floor candidate;
        #     refine 'up' to its normal (the splat is far denser than the sparse points)
        pu = prior_up / np.linalg.norm(prior_up)
        horiz = []
        for pl in planes:
            n = np.array(pl["normal"]); n = n if np.dot(n, pu) >= 0 else -n
            ang = float(np.degrees(np.arccos(np.clip(np.dot(n, pu), -1, 1))))
            if ang <= 45:
                horiz.append((pl, n, ang))
        refined = False
        manual = s.get("up_vector") or [0, 0, 0]
        if horiz and not any(manual):
            pl, n, ang = max(horiz, key=lambda t: t[0]["points"])
            if ang > 1.0:
                M = rotation_aligning(n, np.array([0.0, 0.0, 1.0]))
                T = {**T, "M": M.tolist(), "estimated_up": n.tolist(), "up_source": "mesh planes", "floor_points": pl["points"]}
                (ctx.out / "orientation.json").write_text(json.dumps(T))
                refined = True
                ctx.log(f"orientation refined: up rotated {ang:.1f}° to the largest horizontal plane ({pl['points']} points)")
        # --- naming: lowest big horizontal = floor, other horizontal = flat, vertical = wall
        up_w = np.array([0, 0, 1.0])
        for pl in planes:
            nw = M @ np.array(pl["normal"]); nw = nw if nw[2] >= 0 else -nw
            pl["nz"] = float(nw[2]); pl["height"] = float((M @ (np.array(pl["centre"]) - np.array(T["centre"])))[2])
        cam_h = [(M @ (c_ - np.array(T["centre"])))[2] for c_ in centres]
        cam_top = float(np.max(cam_h)) if len(cam_h) else 0.0
        horizontal = [pl for pl in planes if pl["nz"] > 0.85 and pl["height"] <= cam_top]
        floor = min(horizontal, key=lambda p: p["height"] - 1e-6 * p["points"]) if horizontal else None
        counters = {}
        for pl in planes:
            if pl is floor:
                kind = "floor"
            elif pl["nz"] > 0.85 and pl["height"] > cam_top:
                kind = "ceiling"
            elif pl["nz"] > 0.85:
                kind = "flat"          # seat, cushion, table top
            elif pl["nz"] < 0.3:
                kind = "wall"
            else:
                kind = "slope"
            counters[kind] = counters.get(kind, 0) + 1
            pl["kind"] = kind; pl["name"] = f"{kind}_{counters[kind]}" if kind != "floor" else "floor"
            ctx.log(f"  {pl['name']:8s} {pl['points']:6d} pts  height {pl['height']:+.2f}")
        # combined mesh (two triangles per plane) for the viewer and as a single-file fallback
        V, F = [], []
        for pl in planes:
            i = len(V); V += pl["corners"]; F += [[i, i + 1, i + 2], [i, i + 2, i + 3]]
        V = np.array(V); F = np.array(F)
        write_mesh_ply(ctx.out / "mesh.ply", V, F)
        (ctx.out / "planes.json").write_text(json.dumps({"planes": planes}))
        edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        (ctx.out / "mesh_preview.json").write_text(json.dumps({"xyz": V.round(4).tolist(), "edges": edges.tolist(),
                                                              "faces": int(len(F)), "verts": int(len(V))}))
        return {"planes": len(planes), "names": [p["name"] for p in planes], "faces": int(len(F)), "verts": int(len(V)),
                "orientation_refined": refined}


    # ---- COLMAP multi-view stereo ------------------------------------------------
    def _colmap_dense(self, ctx: Context) -> dict:
        """patch_match_stereo → stereo_fusion → poisson_mesher on the undistorted
        workspace the Track step already made.  GPU only.  Good on textured
        surfaces (fabric, floors, shelves); expect holes on plain walls."""
        import open3d as o3d
        s = ctx.settings
        colmap = ctx.project.tool("colmap")
        ws = ctx.project.dir("track") / "undistorted"
        if not (ws / "stereo").exists():
            raise StepError("track/undistorted/stereo missing — re-run the Track step")
        fused = ctx.out / "fused.ply"; poisson = ctx.out / "poisson.ply"
        ctx.set_progress(0.05, "patch-match stereo (slow)")
        ctx.run([colmap, "patch_match_stereo", "--workspace_path", ws, "--workspace_format", "COLMAP",
                 "--PatchMatchStereo.geom_consistency", "true",
                 "--PatchMatchStereo.max_image_size", str(int(s.get("dense_max_size", 1000)))],
                progress_parser=_pm_prog)
        ctx.check_cancel()
        ctx.set_progress(0.7, "fusing depth maps")
        ctx.run([colmap, "stereo_fusion", "--workspace_path", ws, "--workspace_format", "COLMAP",
                 "--input_type", "geometric", "--output_path", fused])
        ctx.set_progress(0.85, "Poisson meshing")
        ctx.run([colmap, "poisson_mesher", "--input_path", fused, "--output_path", poisson,
                 "--PoissonMeshing.trim", str(float(s.get("dense_trim", 7)))])
        mesh = o3d.io.read_triangle_mesh(str(poisson))
        target = int(s.get("mesh_target_faces", 200000))
        if len(mesh.triangles) > target:
            mesh = mesh.simplify_quadric_decimation(target)
        V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles)
        col = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
        write_mesh_ply(ctx.out / "mesh.ply", V, F, col)
        prev = mesh.simplify_quadric_decimation(PREVIEW_FACES) if len(F) > PREVIEW_FACES else mesh
        PV = np.asarray(prev.vertices); PF = np.asarray(prev.triangles)
        edges = np.unique(np.sort(np.concatenate([PF[:, [0, 1]], PF[:, [1, 2]], PF[:, [2, 0]]]), axis=1), axis=0)
        (ctx.out / "mesh_preview.json").write_text(json.dumps({"xyz": PV.round(4).tolist(), "edges": edges.tolist(),
                                                              "faces": int(len(F)), "verts": int(len(V))}))
        ctx.log(f"dense mesh: {len(V)} verts, {len(F)} faces")
        return {"verts": int(len(V)), "faces": int(len(F)), "source": "colmap_dense"}


def _pm_prog(line: str):
    import re
    m = re.search(r"Processing view (\d+) / (\d+)", line)
    if m:
        return 0.05 + 0.6 * int(m.group(1)) / max(1, int(m.group(2)))
    return None
