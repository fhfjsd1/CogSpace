"""Scene 3D reconstruction + ground alignment pipeline.

Combines Pi3X depth/pose estimation with RANSAC ground-plane alignment
to produce a ready-to-use reconstruction NPZ for downstream stages.

Usage as library:
    from reconstruct_scene import reconstruct_and_align
    ok = reconstruct_and_align("raw_frames", "recon/raw_frames.npz")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root is on sys.path so pi3 is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ------------------------------------------------------------------ #
#  Pi3X chunked inference                                             #
# ------------------------------------------------------------------ #

def _forward_head_chunked(model, hidden, pos, B, N, H, W, patch_h, patch_w, chunk_size=64):
    """Chunked forward_head to avoid OOM on long video sequences."""
    import torch
    import torch.nn.functional as F

    device = hidden.device
    hw = patch_h * patch_w + model.patch_start_idx

    ret_point = model.point_decoder(hidden, xpos=pos)
    ret_camera = model.camera_decoder(hidden, xpos=pos)

    pos_hw = pos.reshape(B, N * hw, -1)
    ret_metric = model.metric_decoder(
        model.metric_token.repeat(B, 1, 1),
        hidden.reshape(B, N * hw, -1),
        xpos=pos_hw[:, 0:1],
        ypos=pos_hw,
    )
    ret_conf = model.conf_decoder(hidden, xpos=pos)

    ret_point = ret_point.reshape(B, N, hw, -1)
    ret_camera = ret_camera.reshape(B, N, hw, -1)
    ret_conf = ret_conf.reshape(B, N, hw, -1)

    num_chunks = (N + chunk_size - 1) // chunk_size
    xy_list, z_list, conf_list, camera_poses_list = [], [], [], []

    for ci in range(num_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, N)
        cn = e - s

        cp = ret_point[:, s:e].reshape(B * cn, hw, -1)
        cc = ret_camera[:, s:e].reshape(B * cn, hw, -1)
        cf = ret_conf[:, s:e].reshape(B * cn, hw, -1)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            cxy, cz = model.point_head(
                cp[:, model.patch_start_idx:].float(), patch_h=patch_h, patch_w=patch_w,
            )
            cxy = cxy.permute(0, 2, 3, 1).reshape(B, cn, H, W, -1)
            cz = cz.permute(0, 2, 3, 1).reshape(B, cn, H, W, -1)

            c_cam = model.camera_head(
                cc[:, model.patch_start_idx:].float(), patch_h, patch_w,
            ).reshape(B, cn, 4, 4)

            c_conf = model.conf_head(
                cf[:, model.patch_start_idx:].float(), patch_h=patch_h, patch_w=patch_w,
            )[0]
            c_conf = c_conf.permute(0, 2, 3, 1).reshape(B, cn, H, W, -1)

        xy_list.append(cxy)
        z_list.append(cz)
        camera_poses_list.append(c_cam)
        conf_list.append(c_conf)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        xy = torch.cat(xy_list, dim=1)
        z = torch.cat(z_list, dim=1)
        camera_poses = torch.cat(camera_poses_list, dim=1)
        conf = torch.cat(conf_list, dim=1)

        z = torch.exp(z.clamp(max=15.0))
        local_points = torch.cat([xy * z, z], dim=-1)
        rays = F.normalize(torch.cat([xy, torch.ones_like(z)], dim=-1), dim=-1)

        metric = model.metric_head(ret_metric.float()).reshape(B).exp()

        from pi3.utils.geometry import homogenize_points
        points = torch.einsum(
            "bnij, bnhwj -> bnhwi",
            camera_poses,
            homogenize_points(local_points),
        )[..., :3] * metric.view(B, 1, 1, 1, 1)

        camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric.view(B, 1, 1)
        local_points = local_points * metric.view(B, 1, 1, 1, 1)

    return dict(
        points=points,
        local_points=local_points,
        rays=rays,
        conf=conf,
        camera_poses=camera_poses,
        metric=metric,
    )


def _inference_chunked(model, imgs, dtype, chunk_size=64):
    import torch

    B, N, _, H, W = imgs.shape
    patch_h, patch_w = H // 14, W // 14

    imgs_norm = (imgs - model.image_mean) / model.image_std
    hidden, poses_, use_depth_mask, use_pose_mask, norm_factor = model.encode(
        imgs_norm, with_prior=True, depths=None, intrinsics=None, poses=None, rays=None,
    )
    hidden = hidden.reshape(B, N, -1, model.dec_embed_dim)
    hidden, pos = model.decode(hidden, N, H, W, poses_, use_pose_mask)

    return _forward_head_chunked(model, hidden, pos, B, N, H, W, patch_h, patch_w, chunk_size)


def _run_pi3x_reconstruction(
    raw_frames_dir: str,
    output_npz_path: str,
    max_images: int = 500,
    chunk_size: int = 64,
    device_str: str = "cuda",
    model=None,
) -> bool:
    """Run Pi3X depth & pose estimation and save to NPZ.

    Args:
        model: Optional pre-loaded Pi3X model on device.
               If None, will load from pretrained (standalone mode).
    """
    import torch
    from pi3.utils.basic import load_multimodal_data
    from pi3.models.pi3x import Pi3X

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(output_npz_path) or ".", exist_ok=True)

    _own_model = False
    if model is None:
        print(f"[Recon] Loading Pi3X model on {device} ...")
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        model.disable_multimodal()
        model = model.to(device)
        _own_model = True
    else:
        print(f"[Recon] Using pre-loaded Pi3X model")

    print(f"[Recon] Loading images from: {raw_frames_dir}")
    conditions = dict(intrinsics=None, poses=None, depths=None)
    imgs, conditions = load_multimodal_data(
        raw_frames_dir, conditions, interval=1, device=device, verbose=True,
    )

    if imgs.numel() == 0:
        print("[Recon] ERROR: No images found.")
        return False

    # Record frame filename mapping (sorted .jpg stems)
    all_frame_names = sorted(
        n for n in os.listdir(raw_frames_dir)
        if n.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    num_images = imgs.shape[1]
    if num_images > max_images:
        print(f"[Recon] Limiting {num_images} images to {max_images}")
        indices = torch.linspace(0, num_images - 1, max_images, dtype=torch.long)
        imgs = imgs[:, indices]
        all_frame_names = [all_frame_names[int(i)] for i in indices]

    frame_names = json.dumps(all_frame_names)

    print(f"[Recon] Image tensor shape: {imgs.shape}")

    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print("[Recon] Running chunked inference ...")

    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=dtype):
            res = _inference_chunked(model, imgs, dtype, chunk_size=chunk_size)

    print(f"[Recon] Saving NPZ -> {output_npz_path}")
    save_dict = {
        "camera_poses": res["camera_poses"].cpu().numpy(),
        "local_points": res["local_points"].cpu().numpy(),
        "conf": res["conf"].cpu().numpy(),
        "frame_names": np.array(frame_names),
    }
    with open(output_npz_path, "wb") as f:
        np.savez_compressed(f, **save_dict)

    print(f"[Recon] Metric scale factor: {res['metric'].item():.4f}")
    if _own_model:
        del model
        torch.cuda.empty_cache()
    return True


# ------------------------------------------------------------------ #
#  Ground alignment                                                  #
# ------------------------------------------------------------------ #

def _rotation_from_a_to_b(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3, dtype=np.float32) if c > 0 else np.eye(3, dtype=np.float32)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s + 1e-12))).astype(np.float32)


def _fit_plane_ransac_with_prior(points, iters, thr, rng, normal_prior, prior_weight, min_align):
    n_pts = points.shape[0]
    if n_pts < 64:
        raise RuntimeError(f"Too few points for ground estimation: {n_pts}")

    prior = normal_prior.astype(np.float64)
    prior = prior / (np.linalg.norm(prior) + 1e-12)

    best_inliers = None
    best_score = -1.0
    best_n = None
    best_d = 0.0

    for _ in range(int(iters)):
        ids = rng.choice(n_pts, size=3, replace=False)
        p1, p2, p3 = points[ids]
        n = np.cross(p2 - p1, p3 - p1)
        nn = float(np.linalg.norm(n))
        if nn < 1e-8:
            continue
        n = n / nn
        align = abs(float(np.dot(n, prior)))
        if align < float(min_align):
            continue
        d = -float(np.dot(n, p1))
        dist = np.abs(points @ n + d)
        inliers = dist < float(thr)
        cnt = int(np.count_nonzero(inliers))
        score = float(cnt) + float(prior_weight) * align * float(n_pts)
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_n = n
            best_d = d

    if best_inliers is None or best_n is None:
        raise RuntimeError("RANSAC failed to find a valid plane")

    inlier_pts = points[best_inliers]
    c = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - c[None, :], full_matrices=False)
    n_refine = vh[-1]
    n_refine = n_refine / (np.linalg.norm(n_refine) + 1e-12)
    d_refine = -float(np.dot(n_refine, c))

    if float(np.dot(n_refine, prior)) < 0.0:
        n_refine = -n_refine
        d_refine = -d_refine

    inliers_final = np.abs(points @ n_refine + d_refine) < float(thr)
    return n_refine.astype(np.float32), float(d_refine), inliers_final


def _estimate_up_prior_from_poses(camera_poses, pose_convention):
    poses_flat = camera_poses.reshape(-1, 4, 4).astype(np.float64)
    if poses_flat.shape[0] == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rs = []
    for p in poses_flat:
        if pose_convention == "w2c":
            p = np.linalg.inv(p)
        rs.append(p[:3, :3])
    rs = np.stack(rs, axis=0)

    def axis_stability(axis_idx):
        vecs = rs[:, :, axis_idx]
        ref = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-12)
        aligned = vecs.copy()
        dots = np.sum(aligned * ref[None, :], axis=1)
        aligned[dots < 0.0] *= -1.0
        norms = np.linalg.norm(aligned, axis=1, keepdims=True) + 1e-12
        aligned = aligned / norms
        mean_v = aligned.mean(axis=0)
        mean_n = np.linalg.norm(mean_v)
        return (0.0, np.array([0.0, 0.0, 1.0], dtype=np.float64)) if mean_n < 1e-12 else (float(mean_n), mean_v / mean_n)

    best_s, best_v = -1.0, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for a in (0, 1, 2):
        s, v = axis_stability(a)
        if s > best_s:
            best_s, best_v = s, v
    return best_v.astype(np.float32)


def _align_ground(npz_path: str, pose_convention: str = "c2w", seed: int = 2026) -> bool:
    """In-place ground alignment on a single NPZ file."""
    rng = np.random.default_rng(seed)

    with np.load(npz_path, allow_pickle=True) as data:
        if "camera_poses" not in data or "local_points" not in data:
            print("[Align] ERROR: Missing camera_poses or local_points.")
            return False
        content = {k: data[k] for k in data.files}

    camera_poses = np.asarray(content["camera_poses"], dtype=np.float32)
    local_points = np.asarray(content["local_points"], dtype=np.float32)

    poses_flat = camera_poses.reshape(-1, 4, 4)
    points_flat = local_points.reshape(-1, local_points.shape[2], local_points.shape[3], 3)

    # Sample world points
    sampled = []
    pixel_stride, max_points, max_depth = 5, 250000, 20.0
    for i in range(points_flat.shape[0]):
        pts = points_flat[i][::pixel_stride, ::pixel_stride, :].reshape(-1, 3).astype(np.float32)
        valid = np.isfinite(pts).all(axis=1)
        if not np.any(valid):
            continue
        pts = pts[valid]
        z = pts[:, 2]
        vz = np.isfinite(z) & (z > 1e-6) & (z < max_depth)
        if not np.any(vz):
            continue
        pts = pts[vz]
        if pts.shape[0] > 12000:
            pts = pts[rng.choice(pts.shape[0], size=12000, replace=False)]
        # Transform to world
        p = poses_flat[i]
        if pose_convention == "w2c":
            p = np.linalg.inv(p)
        wpts = pts @ p[:3, :3].T + p[:3, 3]
        sampled.append(wpts)

    if not sampled:
        print("[Align] ERROR: No valid world points sampled.")
        return False

    all_pts = np.concatenate(sampled, axis=0).astype(np.float32)
    if all_pts.shape[0] > max_points:
        all_pts = all_pts[rng.choice(all_pts.shape[0], size=max_points, replace=False)]

    up_prior = _estimate_up_prior_from_poses(camera_poses, pose_convention)
    print(f"[Align] Up prior: {up_prior.tolist()}")

    n, d, inliers = _fit_plane_ransac_with_prior(
        points=all_pts, iters=1600, thr=0.03, rng=rng,
        normal_prior=up_prior, prior_weight=0.20, min_align=0.50,
    )

    inlier_ratio = float(np.mean(inliers))
    if inlier_ratio < 0.03:
        print(f"[Align] ERROR: Ground inlier ratio too low: {inlier_ratio:.4f}")
        return False

    # Orient normal upward (cameras on positive side)
    cam_centers = []
    for p in poses_flat:
        pc = p if pose_convention == "c2w" else np.linalg.inv(p)
        cam_centers.append(pc[:3, 3])
    cam_centers = np.array(cam_centers, dtype=np.float32)
    if cam_centers.shape[0] > 0:
        signed = cam_centers @ n + d
        if float(np.mean(signed > 0.0)) < 0.5:
            n, d = -n, -d

    # Build alignment transform: plane normal -> +Z, plane at z=0
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    r = _rotation_from_a_to_b(n.astype(np.float32), z_axis)
    p0 = (-d) * n
    p0_r = r @ p0.astype(np.float32)
    tz = -float(p0_r[2])

    a = np.eye(4, dtype=np.float32)
    a[:3, :3] = r
    a[2, 3] = tz

    # Apply to camera poses
    if pose_convention == "c2w":
        content["camera_poses"] = (a[None, None, :, :] @ camera_poses).astype(np.float32)
    else:
        a_inv = np.linalg.inv(a).astype(np.float32)
        content["camera_poses"] = (camera_poses @ a_inv[None, None, :, :]).astype(np.float32)

    prev = content.get("ground_align_matrix")
    if prev is not None and np.asarray(prev).shape == (4, 4):
        content["ground_align_matrix"] = (a @ np.asarray(prev, dtype=np.float32)).astype(np.float32)
    else:
        content["ground_align_matrix"] = a.astype(np.float32)

    # Save in-place
    tmp_path = Path(npz_path).with_suffix(Path(npz_path).suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **content)
    tmp_path.replace(npz_path)

    print(f"[Align] Done. inlier_ratio={inlier_ratio:.4f}")
    return True


# ------------------------------------------------------------------ #
#  Public interface                                                    #
# ------------------------------------------------------------------ #

def reconstruct_and_align(
    raw_frames_dir: str,
    output_npz_path: str,
    max_images: int = 500,
    chunk_size: int = 64,
    pose_convention: str = "c2w",
    seed: int = 2026,
    model=None,
) -> bool:
    """Full pipeline: Pi3X reconstruction -> ground alignment.

    Args:
        raw_frames_dir: Directory containing raw video frames (.jpg/.png).
        output_npz_path: Path to save the final aligned NPZ (e.g. "recon/raw_frames.npz").
        max_images: Maximum number of frames to process.
        chunk_size: Chunk size for Pi3X inference (reduce if OOM).
        pose_convention: "c2w" or "w2c".
        seed: Random seed for RANSAC.
        model: Optional pre-loaded Pi3X model (for Ray Serve deployed mode).

    Returns:
        True on success, False on failure.
    """
    if not os.path.isdir(raw_frames_dir):
        print(f"[Recon] ERROR: Raw frames directory not found: {raw_frames_dir}")
        return False

    # Step 1: Pi3X reconstruction
    print(f"\n{'='*60}")
    print("Step 1: Pi3X 3D Reconstruction")
    print(f"{'='*60}")

    ok = _run_pi3x_reconstruction(
        raw_frames_dir=raw_frames_dir,
        output_npz_path=output_npz_path,
        max_images=max_images,
        chunk_size=chunk_size,
        model=model,
    )
    if not ok:
        print("[Recon] Pi3X reconstruction failed.")
        return False

    # Step 2: Ground alignment
    print(f"\n{'='*60}")
    print("Step 2: Ground Alignment (RANSAC)")
    print(f"{'='*60}")

    ok = _align_ground(output_npz_path, pose_convention=pose_convention, seed=seed)
    if not ok:
        print("[Recon] Ground alignment failed.")
        return False

    print(f"\n[Recon] Pipeline complete -> {output_npz_path}")
    return True
