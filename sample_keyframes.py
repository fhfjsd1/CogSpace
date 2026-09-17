"""Geometric keyframe sampling from reconstructed scene.

Reads camera poses from the aligned NPZ (produced by reconstruct_scene.py),
applies the motion-aware uniform sampling strategy from geometric_sample_frames.py,
and copies the selected frames to the sampled_frames output directory.

Usage as library:
    from sample_keyframes import sample_keyframes
    selected_ids = sample_keyframes(
        recon_npz_path="recon/raw_frames.npz",
        raw_frames_dir="raw_frames",
        output_dir="sampled_frames/arkitscenes/41069025",
        target_frames=64,
        rotation_weight=5,
    )

Usage as script:
    python sample_keyframes.py \
        --recon-npz recon/raw_frames.npz \
        --raw-frames-dir raw_frames \
        --output-dir sampled_frames/arkitscenes/41069025 \
        --target-frames 64
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ #
#  Core sampling utilities (adapted from geometric_sample_frames.py)  #
# ------------------------------------------------------------------ #

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def _forward_dir_from_pose(pose: np.ndarray) -> np.ndarray:
    """Camera forward direction (z-axis of rotation matrix), roll-agnostic."""
    return _normalize(pose[:3, 2])


def _rotation_change_ignore_roll(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Rotation change between two frames, only measuring viewing direction."""
    fa = _forward_dir_from_pose(pose_a)
    fb = _forward_dir_from_pose(pose_b)
    c = float(np.clip(np.dot(fa, fb), -1.0, 1.0))
    return float(np.arccos(c))


def _mixed_motion_distance(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    rotation_weight: float,
    mean_trans_step: float,
    mean_rot_step: float,
) -> float:
    trans = float(np.linalg.norm(pose_b[:3, 3] - pose_a[:3, 3]))
    rot = _rotation_change_ignore_roll(pose_a, pose_b)
    return (trans / max(mean_trans_step, 1e-12)) + rotation_weight * (rot / max(mean_rot_step, 1e-12))


def _laplacian_variance(gray: np.ndarray) -> float:
    """No-dependency Laplacian variance sharpness metric."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    return float(np.var(lap, dtype=np.float64))


def _gradient_anisotropy(gray: np.ndarray) -> float:
    """Gradient anisotropy for detecting directional motion blur."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 1.0
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    vx = float(np.var(gx, dtype=np.float64))
    vy = float(np.var(gy, dtype=np.float64))
    lo, hi = min(vx, vy), max(vx, vy)
    if lo < 1e-12:
        return float("inf") if hi > 1e-12 else 1.0
    return hi / lo


def _sample_by_arc_length(
    valid_indices: List[int],
    poses: List[np.ndarray],
    target_n: int,
    rotation_weight: float,
) -> List[int]:
    """Uniform sampling on cumulative motion arc length."""
    if len(valid_indices) <= target_n:
        return list(valid_indices)

    trans_steps = [float(np.linalg.norm(poses[i][:3, 3] - poses[i - 1][:3, 3])) for i in range(1, len(valid_indices))]
    rot_steps = [_rotation_change_ignore_roll(poses[i - 1], poses[i]) for i in range(1, len(valid_indices))]

    mean_trans = float(np.mean(trans_steps)) if trans_steps else 1.0
    mean_rot = float(np.mean(rot_steps)) if rot_steps else 1.0
    if not np.isfinite(mean_trans) or mean_trans < 1e-12:
        mean_trans = 1.0
    if not np.isfinite(mean_rot) or mean_rot < 1e-12:
        mean_rot = 1.0

    cumulative = np.zeros(len(valid_indices), dtype=np.float64)
    for i in range(1, len(valid_indices)):
        d = _mixed_motion_distance(poses[i - 1], poses[i], rotation_weight, mean_trans, mean_rot)
        cumulative[i] = cumulative[i - 1] + d

    total = float(cumulative[-1])
    if total < 1e-9:
        # Degenerate: uniform time sampling
        pick = np.linspace(0, len(valid_indices) - 1, target_n)
        pick = np.round(pick).astype(int)
        pick = np.clip(pick, 0, len(valid_indices) - 1)
        return [valid_indices[i] for i in sorted(set(int(i) for i in pick))]

    targets = np.linspace(0.0, total, target_n)
    right = np.searchsorted(cumulative, targets, side="left")
    chosen_pos: List[int] = []
    for ti, r in enumerate(right):
        if r <= 0:
            chosen_pos.append(0)
        elif r >= len(cumulative):
            chosen_pos.append(len(cumulative) - 1)
        else:
            prev_i = r - 1
            idx = int(r if abs(cumulative[r] - targets[ti]) < abs(cumulative[prev_i] - targets[ti]) else prev_i)
            chosen_pos.append(idx)

    # Deduplicate while preserving order
    out = []
    last = -1
    for p in chosen_pos:
        if p != last:
            out.append(p)
            last = p

    result = [valid_indices[i] for i in out]

    # Pad with temporal uniform if under target
    if len(result) < target_n:
        temporal = np.linspace(0, len(valid_indices) - 1, target_n)
        temporal = np.round(temporal).astype(int)
        temporal = np.clip(temporal, 0, len(valid_indices) - 1)
        s = set(result)
        for t in temporal.tolist():
            if t not in s:
                result.append(t)
                s.add(t)
            if len(result) >= target_n:
                break

    # Pad from remaining valid indices
    if len(result) < target_n:
        s = set(result)
        for vi in valid_indices:
            if vi not in s:
                result.append(vi)
                s.add(vi)
            if len(result) >= target_n:
                break

    result = sorted(set(result))
    if len(result) > target_n:
        idx = np.linspace(0, len(result) - 1, target_n)
        idx = np.round(idx).astype(int)
        result = sorted(set(result[int(i)] for i in idx))
        if len(result) < target_n:
            s = set(result)
            for vi in valid_indices:
                if vi not in s:
                    result.append(vi)
                    s.add(vi)
                if len(result) >= target_n:
                    break
            result = sorted(result)

    return result[:target_n]


# ------------------------------------------------------------------ #
#  Public interface                                                    #
# ------------------------------------------------------------------ #

def sample_keyframes(
    recon_npz_path: str,
    raw_frames_dir: str,
    output_dir: str,
    target_frames: int = 64,
    rotation_weight: float = 5,
    enable_motion_blur_filter: bool = True,
    motion_blur_lap_threshold: float = 30.0,
    motion_blur_aniso_threshold: float = 4.0,
) -> Optional[List[str]]:
    """Geometric keyframe sampling from reconstructed scene.

    1. Load camera poses and frame name mapping from the aligned NPZ.
    2. Filter out motion-blurred frames (optional).
    3. Apply motion-aware uniform sampling on arc length.
    4. Copy selected frames to output_dir with preserved filenames.

    Args:
        recon_npz_path: Path to the aligned NPZ (from reconstruct_scene.py).
        raw_frames_dir: Directory containing the raw video frames.
        output_dir: Directory to save sampled keyframes.
        target_frames: Target number of keyframes.
        rotation_weight: Weight for rotation component in motion distance.
        enable_motion_blur_filter: Filter out frames with directional motion blur.
        motion_blur_lap_threshold: Laplacian variance upper bound for blur detection.
        motion_blur_aniso_threshold: Gradient anisotropy lower bound for blur detection.

    Returns:
        List of selected frame filenames (stems without extension), or None on error.
    """
    from PIL import Image as PILImage

    if not os.path.exists(recon_npz_path):
        print(f"[Sample] ERROR: NPZ not found: {recon_npz_path}")
        return None

    # Load NPZ
    with np.load(recon_npz_path, allow_pickle=True) as data:
        camera_poses = data["camera_poses"]  # (1, N, 4, 4)
        if "frame_names" in data:
            frame_names = json.loads(str(data["frame_names"]))
        else:
            # Fallback: infer frame names from raw_frames_dir
            all_names = sorted(
                n for n in os.listdir(raw_frames_dir)
                if n.lower().endswith((".jpg", ".png", ".jpeg"))
            )
            n_poses = camera_poses.shape[1]
            if len(all_names) == n_poses:
                frame_names = all_names
            else:
                # NPZ was downsampled during reconstruction — match by linspace indices
                indices = np.linspace(0, len(all_names) - 1, n_poses, dtype=int)
                frame_names = [all_names[i] for i in indices]

    n_frames = len(frame_names)
    n_poses = camera_poses.shape[1]
    if n_frames != n_poses:
        print(f"[Sample] WARNING: frame_names ({n_frames}) != poses ({n_poses}), using min")
        n_frames = min(n_frames, n_poses)
        frame_names = frame_names[:n_frames]
        camera_poses = camera_poses[:, :n_frames]

    poses_flat = camera_poses.reshape(n_frames, 4, 4).astype(np.float64)

    print(f"[Sample] Frames: {n_frames}, target: {target_frames}, rotation_weight: {rotation_weight}")

    # --- Motion blur pre-filtering ---
    from PIL import Image as _PIL
    prefilter_indices: List[int] = []
    for i in range(n_frames):
        img_path = os.path.join(raw_frames_dir, frame_names[i])
        if enable_motion_blur_filter and os.path.exists(img_path):
            try:
                with _PIL.open(img_path) as img:
                    gray = np.asarray(img.convert("L"), dtype=np.float32)
                lap = _laplacian_variance(gray)
                aniso = _gradient_anisotropy(gray)
                if lap <= motion_blur_lap_threshold and aniso >= motion_blur_aniso_threshold:
                    print(f"  [MOTION-DROP] {frame_names[i]}: lap={lap:.2f}, aniso={aniso:.2f}")
                    continue
            except OSError:
                pass
        prefilter_indices.append(i)

    if not prefilter_indices:
        print("[Sample] All frames filtered by motion blur check, using all frames.")
        prefilter_indices = list(range(n_frames))

    candidate_indices = prefilter_indices

    # --- Sampling ---
    if len(candidate_indices) <= target_frames:
        selected = list(candidate_indices)
    else:
        # Collect poses for valid candidates
        valid_poses = [poses_flat[i] for i in candidate_indices]
        selected = _sample_by_arc_length(candidate_indices, valid_poses, target_frames, rotation_weight)

    selected_names = [frame_names[i] for i in selected]
    print(f"[Sample] Selected {len(selected_names)} / {n_frames} frames")

    # --- Copy selected frames ---
    os.makedirs(output_dir, exist_ok=True)
    for name in selected_names:
        src = os.path.join(raw_frames_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"[Sample] WARNING: frame not found: {src}")

    # Also copy corresponding pose .txt files if they exist (for compatibility)
    for name in selected_names:
        stem = os.path.splitext(name)[0]
        src_txt = os.path.join(raw_frames_dir, f"{stem}.txt")
        dst_txt = os.path.join(output_dir, f"{stem}.txt")
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, dst_txt)

    # --- Save selected frame IDs JSON ---
    frame_ids = [os.path.splitext(n)[0] for n in selected_names]
    json_path = os.path.join(output_dir, "selected_frames.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_total": n_frames,
            "n_selected": len(frame_ids),
            "target_frames": target_frames,
            "rotation_weight": rotation_weight,
            "frame_ids": frame_ids,
            "frame_files": selected_names,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Sample] Saved selection JSON -> {json_path}")

    return frame_ids


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Geometric keyframe sampling")
    parser.add_argument("--recon-npz", required=True, help="Path to aligned NPZ")
    parser.add_argument("--raw-frames-dir", required=True, help="Raw frames directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for sampled frames")
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--rotation-weight", type=float, default=5)
    parser.add_argument("--disable-motion-blur-filter", action="store_true")
    args = parser.parse_args()

    result = sample_keyframes(
        recon_npz_path=args.recon_npz,
        raw_frames_dir=args.raw_frames_dir,
        output_dir=args.output_dir,
        target_frames=args.target_frames,
        rotation_weight=args.rotation_weight,
        enable_motion_blur_filter=not args.disable_motion_blur_filter,
    )
    if result is None:
        print("[Sample] Failed.")
    else:
        print(f"[Sample] Done. {len(result)} frames saved to {args.output_dir}")
