"""Build Spatial Mirror from SAM3 segmentation results + reconstruction NPZ.

Takes SAM3 detection JSONL (with optional cross-frame ``object_id`` from video
tracking) and a reconstruction NPZ (local_points / camera_poses / conf), back-
projects masks to world-coordinate 3D points, then merges detections into
consistent 3D instances.

Instance merging strategy:
  • Detections that share the same ``object_id`` (video tracking) are ALWAYS
    merged into one instance, regardless of spatial distance.
  • Detections without ``object_id`` (per-frame mode) are merged using the
    original centroid-distance threshold logic.
  • Both strategies can coexist in the same scene.

Output: ``mirror.json`` containing per-instance geometry (centroid, OBB, AABB).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ #
#  Instance tracking                                                  #
# ------------------------------------------------------------------ #

@dataclass
class InstanceState:
    instance_id: int
    label: str
    centroid: np.ndarray
    point_count: int
    frame_ids: List[str] = field(default_factory=list)
    xyz_min: Optional[np.ndarray] = None
    xyz_max: Optional[np.ndarray] = None
    points: List[np.ndarray] = field(default_factory=list)
    object_id: Optional[int] = None

    def add(self, frame_id: str, pts: np.ndarray) -> None:
        if pts.shape[0] == 0:
            return
        n_old = int(self.point_count)
        n_new = int(pts.shape[0])
        c_new = pts.mean(axis=0)
        if n_old <= 0:
            self.centroid = c_new.astype(np.float32)
            self.point_count = n_new
        else:
            self.centroid = ((self.centroid * n_old) + (c_new * n_new)) / float(n_old + n_new)
            self.point_count = n_old + n_new
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        self.xyz_min = pmin if self.xyz_min is None else np.minimum(self.xyz_min, pmin)
        self.xyz_max = pmax if self.xyz_max is None else np.maximum(self.xyz_max, pmax)
        self.frame_ids.append(frame_id)
        self.points.append(pts.astype(np.float32))


class InstanceTracker:
    """Cross-frame instance tracker with two merging modes.

    * **object_id merge**: detections that carry the same ``object_id`` are
      unconditionally merged into one instance (video tracking guarantee).
    * **distance merge**: detections *without* ``object_id`` are merged by
      centroid distance threshold (per-frame fallback).
    """

    def __init__(
        self,
        dist_threshold_m: float = 0.8,
        max_points_per_instance: int = 200000,
    ) -> None:
        self.dist_threshold_m = float(dist_threshold_m)
        self.max_points_per_instance = int(max_points_per_instance)
        self.by_label: Dict[str, List[InstanceState]] = {}
        self.next_id = 1

    # ---- object_id keyed index ----

    def _objid_index(self) -> Dict[Tuple[str, int], InstanceState]:
        idx: Dict[Tuple[str, int], InstanceState] = {}
        for lst in self.by_label.values():
            for inst in lst:
                if inst.object_id is not None:
                    idx[(inst.label, inst.object_id)] = inst
        return idx

    # ---- assign ----

    def assign(
        self,
        label: str,
        frame_id: str,
        points: np.ndarray,
        object_id: Optional[int] = None,
        used_ids_in_frame: Optional[set] = None,
    ) -> int:
        if used_ids_in_frame is None:
            used_ids_in_frame = set()

        centroid = points.mean(axis=0).astype(np.float32)
        bucket = self.by_label.setdefault(label, [])

        # Strategy 1: exact object_id match (video tracking)
        if object_id is not None:
            idx = self._objid_index()
            key = (label, object_id)
            if key in idx:
                inst = idx[key]
                if inst.instance_id not in used_ids_in_frame:
                    inst.add(frame_id, points)
                    return inst.instance_id

        # Strategy 2: distance-based (per-frame / fallback)
        best_idx = -1
        best_dist = float("inf")
        for i, inst in enumerate(bucket):
            if inst.instance_id in used_ids_in_frame:
                continue
            d = float(np.linalg.norm(centroid - inst.centroid))
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx >= 0 and best_dist < self.dist_threshold_m:
            bucket[best_idx].add(frame_id, points)
            return bucket[best_idx].instance_id

        # Create new instance
        new_inst = InstanceState(
            instance_id=self.next_id,
            label=label,
            centroid=centroid,
            point_count=int(points.shape[0]),
            frame_ids=[frame_id],
            xyz_min=points.min(axis=0),
            xyz_max=points.max(axis=0),
            points=[points.astype(np.float32)],
            object_id=object_id,
        )
        bucket.append(new_inst)
        self.next_id += 1
        return new_inst.instance_id

    # ---- export ----

    def _get_points_for_obb(self, inst: InstanceState) -> np.ndarray:
        if not inst.points:
            return np.empty((0, 3), dtype=np.float32)
        pts = np.concatenate(inst.points, axis=0)
        if pts.shape[0] <= self.max_points_per_instance:
            return pts
        idx = np.linspace(0, pts.shape[0] - 1, self.max_points_per_instance, dtype=np.int64)
        return pts[idx]

    def export_instances(self) -> List[Dict[str, Any]]:
        all_inst: List[InstanceState] = []
        for lst in self.by_label.values():
            all_inst.extend(lst)
        all_inst.sort(key=lambda x: x.instance_id)

        out: List[Dict[str, Any]] = []
        for inst in all_inst:
            pts = self._get_points_for_obb(inst)
            obb = estimate_yaw_obb(pts)
            out.append({
                "instance_id": int(inst.instance_id),
                "label": inst.label,
                "centroid": inst.centroid.astype(float).tolist(),
                "bbox_obb_z": obb,
                "bbox_aabb": {
                    "xyz_min": inst.xyz_min.astype(float).tolist() if inst.xyz_min is not None else None,
                    "xyz_max": inst.xyz_max.astype(float).tolist() if inst.xyz_max is not None else None,
                },
                "point_count": int(inst.point_count),
                "frame_ids": sorted(
                    list(set(inst.frame_ids)),
                    key=lambda x: int(x) if str(x).isdigit() else str(x),
                ),
            })
        return out


# ------------------------------------------------------------------ #
#  Geometry helpers                                                   #
# ------------------------------------------------------------------ #

def estimate_yaw_obb(points: np.ndarray) -> Dict[str, Any]:
    if points.shape[0] == 0:
        return {"center": [0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0], "yaw_rad": 0.0}
    center = points.mean(axis=0)
    if points.shape[0] < 3:
        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)
        return {
            "center": center.astype(float).tolist(),
            "size": (xyz_max - xyz_min).astype(float).tolist(),
            "yaw_rad": 0.0,
        }
    pts_xy = points[:, :2] - center[:2]
    cov = np.cov(pts_xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    yaw = float(np.arctan2(major[1], major[0]))
    c, s = float(np.cos(-yaw)), float(np.sin(-yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy_rot = (rot @ pts_xy.T).T
    min_xy = xy_rot.min(axis=0)
    max_xy = xy_rot.max(axis=0)
    min_z, max_z = float(points[:, 2].min()), float(points[:, 2].max())
    size = np.array([max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], max_z - min_z], dtype=np.float32)
    center_rot_xy = 0.5 * (min_xy + max_xy)
    c2, s2 = float(np.cos(yaw)), float(np.sin(yaw))
    rot_inv = np.array([[c2, -s2], [s2, c2]], dtype=np.float32)
    center_xy = (rot_inv @ center_rot_xy.reshape(2, 1)).reshape(2,) + center[:2]
    center_z = 0.5 * (min_z + max_z)
    return {
        "center": [float(center_xy[0]), float(center_xy[1]), float(center_z)],
        "size": size.astype(float).tolist(),
        "yaw_rad": yaw,
    }


def filter_outlier_points(
    points: np.ndarray,
    outlier_percentile: float = 10.0,
    mad_k: float = 3.5,
    min_points: int = 32,
) -> np.ndarray:
    if points.shape[0] < max(4, int(min_points)):
        return points
    pts = points
    p = float(outlier_percentile)
    if p > 0.0:
        p = min(max(p, 0.0), 49.0)
        lo = np.percentile(pts, p, axis=0)
        hi = np.percentile(pts, 100.0 - p, axis=0)
        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        if int(np.count_nonzero(keep)) >= max(8, int(0.2 * points.shape[0])):
            pts = pts[keep]
    if pts.shape[0] < max(4, int(min_points)) or mad_k <= 0.0:
        return pts
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med[None, :], axis=1)
    d_med = float(np.median(d))
    mad = float(np.median(np.abs(d - d_med)))
    if mad <= 1e-8:
        return pts
    robust_z = 0.6745 * (d - d_med) / mad
    keep = robust_z <= float(mad_k)
    if int(np.count_nonzero(keep)) >= max(8, int(0.2 * pts.shape[0])):
        return pts[keep]
    return pts


# ------------------------------------------------------------------ #
#  I/O helpers                                                        #
# ------------------------------------------------------------------ #

def load_recon_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load reconstruction NPZ.

    Returns:
        (poses_flat [F,4,4], points_flat [F,H,W,3], conf_flat [F,H,W,1] or None)
    """
    with np.load(npz_path) as data:
        camera_poses = np.asarray(data["camera_poses"], dtype=np.float32)
        local_points = np.asarray(data["local_points"], dtype=np.float32)
        conf = np.asarray(data["conf"], dtype=np.float32) if "conf" in data else None
    poses_flat = camera_poses.reshape(-1, 4, 4)
    points_flat = local_points.reshape(-1, local_points.shape[2], local_points.shape[3], 3)
    conf_flat = conf.reshape(-1, conf.shape[2], conf.shape[3], 1) if conf is not None else None
    return poses_flat, points_flat, conf_flat


def load_mask_from_npz(mask_npz_path: str, mask_key: Optional[str] = None) -> Optional[np.ndarray]:
    if not os.path.exists(mask_npz_path):
        print(f"[Mirror] [WARN] Mask file not found: {mask_npz_path}")
        return None
    try:
        npz = np.load(mask_npz_path)
    except Exception as exc:
        print(f"[Mirror] [WARN] Failed to read mask: {mask_npz_path} | {exc}")
        return None
    if mask_key and mask_key in npz:
        arr = npz[mask_key]
        return arr.astype(bool) if arr.ndim >= 2 else None
    for k in npz.files:
        arr = npz[k]
        if arr.ndim >= 2:
            return arr.astype(bool)
    print(f"[Mirror] [WARN] No 2D array in mask: {mask_npz_path}, keys={npz.files}")
    return None


# ------------------------------------------------------------------ #
#  Back-projection                                                    #
# ------------------------------------------------------------------ #

def local_points_mask_to_world(
    mask: np.ndarray,
    local_points_hw3: np.ndarray,
    pose_4x4: np.ndarray,
    pose_convention: str = "c2w",
    max_depth_m: float = 20.0,
    conf_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract world-coordinate points from local_points using mask.

    Supports mask → point-map size mismatch via proportional scaling.
    Optionally filters by confidence map (conf > 0).
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    mh, mw = mask.shape
    h, w = local_points_hw3.shape[:2]
    if (mh, mw) != (h, w):
        ys = np.floor(ys.astype(np.float32) * (float(h) / float(mh))).astype(np.int32)
        xs = np.floor(xs.astype(np.float32) * (float(w) / float(mw))).astype(np.int32)

    in_bounds = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    if not np.any(in_bounds):
        return np.empty((0, 3), dtype=np.float32)
    ys = ys[in_bounds]
    xs = xs[in_bounds]

    pts = local_points_hw3[ys, xs].astype(np.float32)
    valid = np.isfinite(pts).all(axis=1)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    pts = pts[valid]
    ys = ys[valid]
    xs = xs[valid]

    if conf_map is not None:
        ch, cw = conf_map.shape[:2]
        ys_c = ys.copy()
        xs_c = xs.copy()
        if (ch, cw) != (h, w):
            ys_c = np.floor(ys_c.astype(np.float32) * (float(ch) / float(h))).astype(np.int32)
            xs_c = np.floor(xs_c.astype(np.float32) * (float(cw) / float(w))).astype(np.int32)
        in_c = (ys_c >= 0) & (ys_c < ch) & (xs_c >= 0) & (xs_c < cw)
        if np.any(in_c):
            conf_vals = conf_map[ys_c[in_c], xs_c[in_c], 0]
            conf_mask = conf_vals > 0
            pts = pts[in_c][conf_mask]

    cam_depth = pts[:, 2]
    valid_depth = np.isfinite(cam_depth) & (cam_depth > 1e-6) & (cam_depth < float(max_depth_m))
    if not np.any(valid_depth):
        return np.empty((0, 3), dtype=np.float32)
    pts = pts[valid_depth]

    pose_use = pose_4x4.astype(np.float32)
    if pose_convention == "w2c":
        pose_use = np.linalg.inv(pose_use)
    rot = pose_use[:3, :3]
    trans = pose_use[:3, 3]
    return (pts @ rot.T + trans).astype(np.float32)


# ------------------------------------------------------------------ #
#  Main pipeline                                                      #
# ------------------------------------------------------------------ #

def build_spatial_mirror(
    sam3_output_dir: str,
    recon_npz_path: str,
    labels: List[str],
    pose_convention: str = "c2w",
    min_score: float = 0.0,
    dist_threshold_m: float = 0.8,
    max_depth_m: float = 20.0,
    outlier_percentile: float = 10.0,
    outlier_mad_k: float = 3.5,
    max_points_per_instance: int = 200000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build Spatial Mirror from SAM3 detections + recon NPZ.

    Args:
        sam3_output_dir: Directory containing ``sam3_detections.jsonl`` and ``masks/``.
        recon_npz_path: Path to reconstruction NPZ (camera_poses, local_points, conf).
        labels: Target object labels to include.
        pose_convention: "c2w" or "w2c" for the camera poses in the NPZ.
        min_score: Minimum detection score to consider.
        dist_threshold_m: Centroid distance threshold for non-tracked merging.
        max_depth_m: Discard points farther than this.
        outlier_percentile: Percentile for outlier trimming (0 to disable).
        outlier_mad_k: MAD-based robust z-score threshold (0 to disable).

    Returns:
        (instances_list, mirror_dict)
        mirror_dict contains centroid_map, entity_summary, stats.
    """
    print("[Mirror] Loading reconstruction NPZ ...")
    poses_flat, points_flat, conf_flat = load_recon_npz(recon_npz_path)
    num_recon_frames = int(poses_flat.shape[0])
    print(f"[Mirror] Recon frames: {num_recon_frames}, point shape: {points_flat.shape[1:]}")

    # Load SAM3 detections
    det_jsonl_path = os.path.join(sam3_output_dir, "sam3_detections.jsonl")
    if not os.path.exists(det_jsonl_path):
        raise FileNotFoundError(f"SAM3 detections not found: {det_jsonl_path}")

    target_labels_lower = {str(l).strip().lower() for l in labels}

    detections: List[Dict[str, Any]] = []
    with open(det_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = str(rec.get("label", "")).strip().lower()
            if label not in target_labels_lower:
                continue
            score = float(rec.get("score", 0.0))
            if score < min_score:
                continue
            detections.append(rec)

    print(f"[Mirror] Loaded {len(detections)} valid detections for labels {labels}")

    # Check if detections carry object_id (video tracking mode)
    has_object_id = any("object_id" in rec for rec in detections)
    if has_object_id:
        obj_ids = {rec.get("object_id") for rec in detections if "object_id" in rec}
        print(f"[Mirror] Video tracking mode: {len(obj_ids)} unique object_ids detected")

    # Group by frame
    det_by_frame: Dict[str, List[Dict[str, Any]]] = {}
    for rec in detections:
        fid = str(rec.get("frame_id", "")).strip()
        det_by_frame.setdefault(fid, []).append(rec)

    # Instance tracking
    tracker = InstanceTracker(
        dist_threshold_m=dist_threshold_m,
        max_points_per_instance=max_points_per_instance,
    )
    matched = 0
    valid = 0

    frame_ids = sorted(det_by_frame.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    for frame_id in frame_ids:
        recs = det_by_frame[frame_id]
        frame_idx = int(frame_id)

        if frame_idx < 0 or frame_idx >= num_recon_frames:
            print(f"[Mirror] Frame {frame_id} out of recon range [0, {num_recon_frames}), skip")
            continue

        local_pts = points_flat[frame_idx]        # (H, W, 3)
        pose = poses_flat[frame_idx]              # (4, 4)
        conf_map = conf_flat[frame_idx] if conf_flat is not None else None  # (H, W, 1)

        used_ids_by_label: Dict[str, set] = {}

        for rec in recs:
            matched += 1
            label = str(rec.get("label", "")).strip().lower()

            mask_rel = str(rec.get("mask_file", "")).strip()
            mask_key = str(rec.get("mask_key", "")).strip() or None
            if not mask_rel:
                continue

            mask_path = os.path.join(sam3_output_dir, mask_rel)
            mask = load_mask_from_npz(mask_path, mask_key)
            if mask is None or mask.ndim != 2:
                continue

            try:
                pts_world = local_points_mask_to_world(
                    mask=mask,
                    local_points_hw3=local_pts,
                    pose_4x4=pose,
                    pose_convention=pose_convention,
                    max_depth_m=max_depth_m,
                    conf_map=conf_map,
                )
            except Exception as exc:
                print(f"[Mirror] Frame {frame_id} label '{label}' back-projection failed: {exc}")
                continue

            pts_world = filter_outlier_points(pts_world, outlier_percentile, outlier_mad_k)
            if pts_world.shape[0] == 0:
                continue

            valid += 1
            used = used_ids_by_label.setdefault(label, set())

            # Pass object_id when available (video tracking mode)
            object_id = rec.get("object_id", None)
            if object_id is not None:
                object_id = int(object_id)

            inst_id = tracker.assign(
                label=label,
                frame_id=frame_id,
                points=pts_world,
                object_id=object_id,
                used_ids_in_frame=used,
            )
            used.add(inst_id)

    instances = tracker.export_instances()
    print(f"[Mirror] Built {len(instances)} instances (matched={matched}, valid_3d={valid})")

    # Build mirror dict (compact entity summary for agent reasoning)
    centroid_map: Dict[str, List[float]] = {}
    entity_summary: List[Dict[str, Any]] = []
    for inst in instances:
        centroid_map[inst["label"]] = inst["centroid"]
        entity_summary.append({
            "label": inst["label"],
            "instance_id": inst["instance_id"],
            "position": inst["centroid"],
            "size": inst["bbox_obb_z"]["size"],
            "frame_ids": inst["frame_ids"],
        })

    mirror_dict = {
        "centroid_map": centroid_map,
        "entity_summary": entity_summary,
        "stats": {
            "matched_detections": matched,
            "valid_detections": valid,
            "instance_count": len(instances),
            "frame_count": len(frame_ids),
            "has_object_id": has_object_id,
        },
    }

    return instances, mirror_dict


def print_spatial_mirror(mirror_dict: Dict[str, Any]) -> None:
    """Pretty-print the Spatial Mirror entity summary."""
    print("\n=== Spatial Mirror (Entity Summary) ===")
    for ent in mirror_dict.get("entity_summary", []):
        pos = ent["position"]
        sz = ent["size"]
        print(f"  [{ent['instance_id']}] {ent['label']}")
        print(f"      position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"      size:     ({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})")
        print(f"      frames:   {ent['frame_ids']}")
        ori = ent.get("orientation")
        if ori:
            dv = ori["direction_vector"]
            print(f"      orient:   azi={ori['azimuth_deg']:.0f} el={ori['elevation_deg']:.0f} "
                  f"ro={ori['rotation_deg']:.0f} alpha={ori['alpha']}")
            print(f"      forward:  ({dv['x']:.3f}, {dv['y']:.3f}, {dv['z']:.3f})")
    stats = mirror_dict.get("stats", {})
    print(f"\n  Stats: instances={stats.get('instance_count', 0)}, "
          f"valid_detections={stats.get('valid_detections', 0)}, "
          f"frames={stats.get('frame_count', 0)}, "
          f"video_tracking={stats.get('has_object_id', False)}")


# ------------------------------------------------------------------ #
#  Instance deduplication (VLM judging, appendix §2.2)                #
# ------------------------------------------------------------------ #

def find_duplicate_labels(instances: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Return labels that have more than one instance."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for inst in instances:
        lb = inst["label"].lower()
        groups.setdefault(lb, []).append(inst)
    return {k: v for k, v in groups.items() if len(v) > 1}


def rebuild_mirror_dict(instances: List[Dict[str, Any]], original_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild mirror_dict from a filtered instances list (no re-projection)."""
    centroid_map: Dict[str, List[float]] = {}
    entity_summary: List[Dict[str, Any]] = []
    for inst in instances:
        centroid_map[inst["label"]] = inst["centroid"]
        entry: Dict[str, Any] = {
            "label": inst["label"],
            "instance_id": inst["instance_id"],
            "position": inst["centroid"],
            "size": inst["bbox_obb_z"]["size"],
            "frame_ids": inst["frame_ids"],
        }
        if "orientation" in inst:
            entry["orientation"] = inst["orientation"]
        entity_summary.append(entry)
    new_stats = dict(original_stats)
    new_stats["instance_count"] = len(instances)
    return {
        "centroid_map": centroid_map,
        "entity_summary": entity_summary,
        "stats": new_stats,
    }


def deduplicate_instances(
    instances: List[Dict[str, Any]],
    sam3_output_dir: str,
    sampled_frames_dir: str,
    client,
    model_name: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Deduplicate instances using VLM judging (appendix_perpro.md §2.2).

    For each label that has >1 instance:
      1. Render candidate visualisations (bbox + instance-id badge).
      2. Ask VLM to pick the single best instance.
      3. Drop the rest.

    Returns:
        (filtered_instances, dropped_count)
    """
    from api_messages import judge_best_instance

    dup_labels = find_duplicate_labels(instances)
    if not dup_labels:
        return instances, 0

    print(f"[Dedup] Found {len(dup_labels)} labels with duplicates: {list(dup_labels.keys())}")

    detections = _load_detections_jsonl(sam3_output_dir)

    kept_ids: set = set()
    # Pre-populate with all non-duplicate instance IDs
    for inst in instances:
        if inst["label"].lower() not in dup_labels:
            kept_ids.add(inst["instance_id"])
    total_dropped = 0

    for label, group in dup_labels.items():
        cand_ids = {inst["instance_id"] for inst in group}

        # --- render candidates ---
        vis_dir = os.path.join(sam3_output_dir, "dedup_vis", label)
        image_paths: List[str] = []
        cand_summary: List[Dict[str, Any]] = []

        for inst in group:
            iid = inst["instance_id"]
            det = _select_best_detection(inst, detections)
            vis_path = os.path.join(vis_dir, f"instance_{iid}.jpg")
            rendered = False
            if det is not None:
                rendered = _render_candidate(inst, det, sampled_frames_dir, vis_path)

            if rendered and os.path.exists(vis_path):
                image_paths.append(vis_path)

            cand_summary.append({
                "instance_id": iid,
                "point_count": inst.get("point_count", 0),
                "frame_count": len(inst.get("frame_ids", [])),
                "rendered": rendered,
            })

        if not image_paths:
            print(f"  [Dedup] '{label}': no renderable candidates, keeping all")
            kept_ids.update(cand_ids)
            continue

        # --- call VLM judge ---
        result = judge_best_instance(
            client=client,
            model_name=model_name,
            label=label,
            candidate_image_paths=image_paths,
            candidate_summary=cand_summary,
        )

        if result is not None:
            keep_id = int(result.get("keep_instance_id", -1))
            if keep_id in cand_ids:
                kept_ids.add(keep_id)
                dropped = len(group) - 1
                total_dropped += dropped
                reason = result.get("reason", "")
                print(f"  [Dedup] '{label}': keep #{keep_id} ({reason}), "
                      f"drop {sorted(cand_ids - {keep_id})}")
                continue

        # fallback: keep all
        print(f"  [Dedup] '{label}': failed to judge, keeping all")
        kept_ids.update(cand_ids)

    filtered = [inst for inst in instances if inst["instance_id"] in kept_ids]
    return filtered, total_dropped


# ---- internal helpers ----

def _load_detections_jsonl(sam3_output_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(sam3_output_dir, "sam3_detections.jsonl")
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return rows


def _select_best_detection(inst: Dict[str, Any], detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick highest-score detection that matches instance frame_ids + label."""
    frame_ids = {str(x) for x in inst.get("frame_ids", [])}
    label = inst["label"].lower()
    cands: List[Tuple[float, Dict[str, Any]]] = []
    for det in detections:
        if det.get("label", "").lower() != label:
            continue
        if str(det.get("frame_id", "")) in frame_ids:
            cands.append((float(det.get("score", 0)), det))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def _render_candidate(
    inst: Dict[str, Any],
    det: Dict[str, Any],
    image_dir: str,
    save_path: str,
) -> bool:
    """Render bbox + instance_id badge on the source frame image."""
    from PIL import Image as _PILImage, ImageDraw as _ImageDraw, ImageFont as _ImageFont

    frame_id = str(det.get("frame_id", ""))
    img_path = os.path.join(image_dir, f"{frame_id}.jpg")
    if not os.path.exists(img_path):
        return False

    img = _PILImage.open(img_path).convert("RGB")
    w, h = img.size
    bbox = det.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False

    try:
        x1, y1, x2, y2 = [max(0, min(w, int(v))) for v in bbox]
    except (TypeError, ValueError):
        return False
    if x2 <= x1 or y2 <= y1:
        return False

    draw = _ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    # instance-id badge
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    radius = min(max(16, (x2 - x1 + y2 - y1) // 16), 42)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(255, 220, 0), outline=(120, 80, 0), width=2,
    )
    text = str(inst["instance_id"])
    font_size = max(18, int(radius * 1.2))
    font = None
    for fn in ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"]:
        try:
            font = _ImageFont.truetype(fn, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = _ImageFont.load_default()

    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((cx - tw // 2, cy - th // 2), text, fill=(220, 20, 0), font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path, quality=95)
    return True



# ------------------------------------------------------------------ #
#  Orientation estimation (Orient-Anything-V2)                        #
# ------------------------------------------------------------------ #

def _load_orient_model(ckpt_path: Optional[str] = None):
    """Load Orient-Anything-V2 model. Returns (model, device)."""
    import sys as _sys
    import torch as _torch

    _ori_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Orient-Anything-V2-main"
    )
    if _ori_dir not in _sys.path:
        _sys.path.insert(0, _ori_dir)

    from utils.paths import LOCAL_CKPT_PATH, HF_CKPT_PATH  # noqa: E402
    from vision_tower import VGGT_OriAny_Ref  # noqa: E402

    if ckpt_path is None:
        if os.path.exists(LOCAL_CKPT_PATH):
            ckpt_path = LOCAL_CKPT_PATH
        else:
            from huggingface_hub import hf_hub_download  # noqa: E402

            ckpt_path = hf_hub_download(
                repo_id="Viglong/OriAnyV2_ckpt",
                filename=HF_CKPT_PATH,
                repo_type="model",
                cache_dir=_ori_dir,
            )

    _dtype = (
        _torch.bfloat16
        if _torch.cuda.is_available()
        and _torch.cuda.get_device_capability()[0] >= 8
        else _torch.float16
    )
    _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    _model = VGGT_OriAny_Ref(out_dim=900, dtype=_dtype, nopretrain=True)
    _model.load_state_dict(_torch.load(ckpt_path, map_location="cpu"))
    _model.eval()
    _model = _model.to(_device)

    print(f"[Orient] Model loaded on {_device}")
    return _model, _device


def _infer_orientation_single(model, device, image_rgba):
    """Run Orient-Anything-V2 on an RGBA image (transparent background)."""
    import torch as _torch
    from PIL import Image as _PILImage

    import sys as _sys
    _ori_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Orient-Anything-V2-main"
    )
    if _ori_dir not in _sys.path:
        _sys.path.insert(0, _ori_dir)
    from utils.app_utils import preprocess_images, val_fit_alpha  # noqa: E402

    background = _PILImage.new("RGBA", image_rgba.size, (255, 255, 255, 255))
    image_rgb = _PILImage.alpha_composite(background, image_rgba).convert("RGB")

    image_tensors = preprocess_images([image_rgb], mode="pad").to(device)

    with _torch.no_grad():
        pose_enc = model(image_tensors.unsqueeze(0))
        pose_enc = pose_enc.view(-1)

        angle_az = _torch.argmax(pose_enc[0:360]).item()
        angle_el = _torch.argmax(pose_enc[360:540]).item() - 90
        angle_ro = _torch.argmax(pose_enc[540:900]).item() - 180

        distribute = _torch.sigmoid(pose_enc[0:360]).cpu().float().numpy().reshape(1, -1)
        alpha = val_fit_alpha(distribute=distribute)
        alpha = alpha[0] if _torch.is_tensor(alpha) else alpha

    return {
        "azimuth": angle_az,
        "elevation": angle_el,
        "rotation": angle_ro,
        "alpha": int(alpha.item()) if _torch.is_tensor(alpha) else int(alpha),
    }


def _angles_to_forward_camera(azimuth: float, elevation: float, rotation: float) -> np.ndarray:
    """Convert orientation angles to forward direction in camera space.

    Rotation order: R = R_rot @ R_ele @ R_azi  (matches Orient-Anything-V2).
    Forward direction = R[:, 0]  (first column).
    Camera convention: X-right, Y-down, Z-forward.
    """
    az = np.radians(azimuth)
    el = np.radians(elevation)
    ro = np.radians(rotation)

    cz, sz = np.cos(az), np.sin(az)
    cy, sy = np.cos(el), np.sin(el)
    cx, sx = np.cos(ro), np.sin(ro)

    R_azi = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R_ele = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_rot = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    R = R_rot @ R_ele @ R_azi
    return R[:, 0]


def _extract_object_rgba(image, mask):
    """Extract object region from image using binary mask, crop & pad to square."""
    from PIL import Image as _PILImage

    if image.size != mask.size:
        mask = mask.resize(image.size, _PILImage.NEAREST)

    img_arr = np.array(image)
    mask_arr = np.array(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_bin = (mask_arr > 127).astype(np.uint8)

    rgba = np.zeros((img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = img_arr
    rgba[:, :, 3] = mask_bin * 255
    rgba_img = _PILImage.fromarray(rgba, "RGBA")

    # crop to foreground bbox + padding
    alpha_ch = rgba[:, :, 3]
    if alpha_ch.max() == 0:
        return rgba_img

    rows = np.any(alpha_ch > 0, axis=1)
    cols = np.any(alpha_ch > 0, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    pad = 0.15
    ph = int((y1 - y0) * pad)
    pw = int((x1 - x0) * pad)
    y0 = max(0, y0 - ph)
    y1 = min(img_arr.shape[0] - 1, y1 + ph)
    x0 = max(0, x0 - pw)
    x1 = min(img_arr.shape[1] - 1, x1 + pw)
    cropped = rgba[y0 : y1 + 1, x0 : x1 + 1]
    cropped_img = _PILImage.fromarray(cropped, "RGBA")

    # resize_foreground (square pad, ratio=0.85) — required by the model
    arr = np.array(cropped_img)
    fg_idx = np.where(arr[..., 3] > 0)
    if len(fg_idx[0]) == 0:
        return rgba_img
    fy0, fy1 = fg_idx[0].min(), fg_idx[0].max()
    fx0, fx1 = fg_idx[1].min(), fg_idx[1].max()
    fg = arr[fy0 : fy1 + 1, fx0 : fx1 + 1]

    side = max(fg.shape[0], fg.shape[1])
    q0h, q0w = (side - fg.shape[0]) // 2, (side - fg.shape[1]) // 2
    q1h, q1w = side - fg.shape[0] - q0h, side - fg.shape[1] - q0w
    sq = np.pad(fg, ((q0h, q1h), (q0w, q1w), (0, 0)), mode="constant")

    new_side = int(sq.shape[0] / 0.85)
    p0h, p0w = (new_side - side) // 2, (new_side - side) // 2
    p1h, p1w = new_side - side - p0h, new_side - side - p0w
    padded = np.pad(sq, ((p0h, p1h), (p0w, p1w), (0, 0)), mode="constant")

    return _PILImage.fromarray(padded, "RGBA")


def _find_largest_mask_detection(
    inst: Dict[str, Any],
    detections: List[Dict[str, Any]],
    sam3_output_dir: str,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Find the detection with the largest mask area for a given instance.

    Returns:
        (best_detection, mask_pixel_count) or (None, 0).
    """
    label = inst["label"].lower()
    frame_ids = {str(x) for x in inst.get("frame_ids", [])}

    best_det = None
    best_area = 0

    for det in detections:
        if det.get("label", "").lower() != label:
            continue
        if str(det.get("frame_id", "")) not in frame_ids:
            continue

        mask_rel = str(det.get("mask_file", "")).strip()
        mask_key = str(det.get("mask_key", "")).strip() or None
        if not mask_rel:
            continue

        mask_path = os.path.join(sam3_output_dir, mask_rel)
        mask = load_mask_from_npz(mask_path, mask_key)
        if mask is None:
            continue

        area = int(np.count_nonzero(mask))
        if area > best_area:
            best_area = area
            best_det = det

    return best_det, best_area


def estimate_orientations(
    instances: List[Dict[str, Any]],
    sam3_output_dir: str,
    sampled_frames_dir: str,
    recon_npz_path: str,
    pose_convention: str = "c2w",
    orient_ckpt_path: Optional[str] = None,
    model=None,
) -> List[Dict[str, Any]]:
    """Estimate orientation for each instance using Orient-Anything-V2.

    For each instance:
      1. Select the frame with the largest mask area.
      2. Load image + mask, extract the object crop.
      3. Run Orient-Anything-V2 to predict (azimuth, elevation, rotation, alpha).
      4. Compute the forward direction in camera space from the angles.
      5. Rotate to world space using the camera-to-world pose from reconstruction.
      6. Register ``orientation`` into the instance dict.

    Args:
        instances: Built instance list (modified in-place).
        sam3_output_dir: Directory with ``sam3_detections.jsonl`` and ``masks/``.
        sampled_frames_dir: Directory with source frame images (``{frame_id}.jpg``).
        recon_npz_path: Path to reconstruction NPZ (camera_poses).
        pose_convention: "c2w" or "w2c".
        orient_ckpt_path: Optional checkpoint path for Orient-Anything-V2.
        model: Optional pre-loaded (model, device) tuple for Ray Serve deployed mode.

    Returns:
        The same instances list, now with an ``orientation`` field per instance.
    """
    from PIL import Image as _PILImage

    # --- Load / reuse Orient-Anything-V2 model ---
    if model is not None:
        _orient_model, _orient_device = model
        print("[Orient] Using pre-loaded Orient-Anything-V2 model")
    else:
        print("[Orient] Loading Orient-Anything-V2 model ...")
        _orient_model, _orient_device = _load_orient_model(orient_ckpt_path)

    # --- Load camera poses ---
    print("[Orient] Loading camera poses ...")
    poses_flat, _, _ = load_recon_npz(recon_npz_path)

    # --- Load detections ---
    detections = _load_detections_jsonl(sam3_output_dir)

    oriented = 0

    for inst in instances:
        iid = inst["instance_id"]
        label = inst["label"]
        frame_ids = inst.get("frame_ids", [])

        if not frame_ids:
            print(f"  [Orient] Instance {iid} '{label}': no frames, skip")
            continue

        # Pick detection with largest mask
        best_det, best_area = _find_largest_mask_detection(
            inst, detections, sam3_output_dir
        )
        if best_det is None:
            print(f"  [Orient] Instance {iid} '{label}': no valid detection, skip")
            continue

        frame_id = str(best_det.get("frame_id", ""))

        # Load source image
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = os.path.join(sampled_frames_dir, f"{frame_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"  [Orient] Instance {iid} '{label}': image not found for frame {frame_id}, skip")
            continue

        # Load mask as PIL Image (L mode, 0/255)
        mask_rel = str(best_det.get("mask_file", "")).strip()
        mask_key = str(best_det.get("mask_key", "")).strip() or None
        mask_path = os.path.join(sam3_output_dir, mask_rel)
        mask_np = load_mask_from_npz(mask_path, mask_key)
        if mask_np is None:
            print(f"  [Orient] Instance {iid} '{label}': mask load failed, skip")
            continue

        try:
            image = _PILImage.open(img_path).convert("RGB")
            mask_img = _PILImage.fromarray(
                (mask_np.astype(np.uint8) * 255), mode="L"
            )

            # Extract object RGBA crop (model input format)
            object_rgba = _extract_object_rgba(image, mask_img)

            # Infer orientation angles
            result = _infer_orientation_single(_orient_model, _orient_device, object_rgba)

            # Forward direction in camera space
            forward_cam = _angles_to_forward_camera(
                result["azimuth"], result["elevation"], result["rotation"]
            )

            # Rotate to world space using camera pose
            frame_idx = int(frame_id)
            if frame_idx < 0 or frame_idx >= poses_flat.shape[0]:
                print(f"  [Orient] Instance {iid} '{label}': frame {frame_id} out of range, skip")
                continue

            pose = poses_flat[frame_idx].astype(np.float32)
            if pose_convention == "w2c":
                c2w_rot = np.linalg.inv(pose)[:3, :3]
            else:
                c2w_rot = pose[:3, :3]

            forward_world = c2w_rot @ forward_cam.astype(np.float32)
            norm = np.linalg.norm(forward_world)
            if norm > 1e-8:
                forward_world = forward_world / norm

            inst["orientation"] = {
                "azimuth_deg": result["azimuth"],
                "elevation_deg": result["elevation"],
                "rotation_deg": result["rotation"],
                "alpha": result["alpha"],
                "direction_vector": {
                    "x": float(forward_world[0]),
                    "y": float(forward_world[1]),
                    "z": float(forward_world[2]),
                },
                "source_frame_id": frame_id,
                "mask_pixel_count": best_area,
            }
            oriented += 1

            dv = inst["orientation"]["direction_vector"]
            print(
                f"  [Orient] Instance {iid} '{label}': "
                f"azi={result['azimuth']:.0f} el={result['elevation']:.0f} "
                f"ro={result['rotation']:.0f} alpha={result['alpha']} "
                f"dir=({dv['x']:.3f}, {dv['y']:.3f}, {dv['z']:.3f})"
            )

        except Exception as e:
            print(f"  [Orient] Instance {iid} '{label}': error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"[Orient] Estimated orientations for {oriented}/{len(instances)} instances")
    return instances
