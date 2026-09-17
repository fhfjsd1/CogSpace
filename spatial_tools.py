"""Spatial cognitive & geometric tools for CogSpace agent reasoning.

Implements the 7 tools:
  1. get_entity(identifier)
  2. query_depth(pixel_xy)
  3. distance(A, B, mode="center")
  4. set_egocentric_view(origin, forward)
  5. turn(direction)
  6. look_top_down()
  7. relative_direction(entity)

Consumes the spatial mirror (mirror.json) produced by build_spatial_mirror.py
as the scene representation, and optionally a reconstruction NPZ for depth queries.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

Vec3 = Tuple[float, float, float]


# ================================================================== #
#  Scene state                                                       #
# ================================================================== #

@dataclass
class EntityRecord:
    """Internal entity record used by the tool runtime."""
    instance_id: int
    label: str
    position: Vec3
    orientation: Optional[Vec3]  # facing direction (unit vector)
    size: Vec3                   # bbox_obb_z size (length, width, height)
    centroid: Vec3
    point_count: int = 0
    frame_ids: List[str] = field(default_factory=list)


@dataclass
class EgocentricFrame:
    """Current egocentric reference frame."""
    origin: Optional[Vec3] = None
    forward: Optional[Vec3] = None   # +y axis of egocentric frame
    right: Optional[Vec3] = None     # +x axis
    up: Optional[Vec3] = None        # +z axis

    @property
    def is_set(self) -> bool:
        return self.origin is not None and self.forward is not None


@dataclass
class SceneState:
    """Runtime state shared by all spatial tools."""
    entities: Dict[str, EntityRecord] = field(default_factory=dict)   # label -> record
    entities_by_id: Dict[int, EntityRecord] = field(default_factory=dict)
    ego: EgocentricFrame = field(default_factory=EgocentricFrame)
    bev_built: bool = False

    # Optional depth data (loaded lazily)
    recon_npz_path: Optional[str] = None
    poses_flat: Optional[np.ndarray] = None   # (F, 4, 4)
    local_points: Optional[np.ndarray] = None  # (F, H, W, 3)
    conf_flat: Optional[np.ndarray] = None     # (F, H, W, 1) or None

    # DA3 (Depth Anything V3) for metric depth queries
    da3_model_dir: Optional[str] = None
    da3_model = None  # lazy-loaded
    da3_device: Optional[str] = None

    def _load_recon(self) -> None:
        if self.poses_flat is not None:
            return
        if self.recon_npz_path is None or not os.path.exists(self.recon_npz_path):
            return
        with np.load(self.recon_npz_path, allow_pickle=True) as data:
            camera_poses = np.asarray(data["camera_poses"], dtype=np.float32)
            local_points = np.asarray(data["local_points"], dtype=np.float32)
            conf = np.asarray(data["conf"], dtype=np.float32) if "conf" in data else None
        self.poses_flat = camera_poses.reshape(-1, 4, 4)
        self.local_points = local_points.reshape(
            -1, local_points.shape[2], local_points.shape[3], 3
        )
        if conf is not None:
            self.conf_flat = conf.reshape(-1, conf.shape[2], conf.shape[3], 1)

    def _load_da3(self) -> None:
        """Lazy-load the Depth Anything V3 model."""
        if self.da3_model is not None:
            return
        import sys
        import torch
        from importlib import import_module
        from pathlib import Path as _Path

        if self.da3_model_dir is None:
            raise RuntimeError(
                "DA3 model dir not set. Call "
                "set_da3_model_dir(path) or pass da3_model_dir to load_spatial_mirror()."
            )

        model_dir = _Path(self.da3_model_dir)
        for candidate in (
            model_dir, model_dir.parent, model_dir / "src",
            model_dir / "depth-anything-3", model_dir / "depth-anything-3" / "src",
        ):
            c = str(candidate)
            if c not in sys.path and candidate.exists():
                sys.path.insert(0, c)

        last_err = None
        for mod_name in ("depth_anything_3.api", "api"):
            try:
                DA3Class = import_module(mod_name).DepthAnything3
                break
            except ModuleNotFoundError as exc:
                last_err = exc
        else:
            raise ModuleNotFoundError(
                f"Cannot import DepthAnything3 from '{self.da3_model_dir}'"
            ) from last_err

        device_str = self.da3_device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        model = DA3Class.from_pretrained(str(model_dir), trust_remote_code=True)
        model = model.to(device=device).eval()

        self.da3_model = model
        self.da3_device = device_str
        print(f"[SpatialTools] DA3 model loaded on {device} from {model_dir}")


# Module-level state (singleton per process)
_state = SceneState()


# ================================================================== #
#  Vec3 math helpers                                                 #
# ================================================================== #

def _v3_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v3_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _v3_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v3_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _v3_norm(a: Vec3) -> float:
    return math.sqrt(_v3_dot(a, a))


def _v3_normalize(a: Vec3) -> Vec3:
    n = _v3_norm(a)
    if n < 1e-12:
        raise ValueError("cannot normalize zero-length vector")
    return _v3_scale(a, 1.0 / n)


def _v3_to_list(v: Vec3) -> List[float]:
    return [float(v[0]), float(v[1]), float(v[2])]


def _as_vec3(value: Any, default_z: float = 0.0) -> Vec3:
    """Convert a value to a 3D vector.

    Supports:
      - [x, y, z] or (x, y, z)  -> 3D
      - [x, y] or (x, y)        -> 2D, z filled with *default_z*
      - {"x": ..., "y": ..., "z": ...}  -> dict form
    """
    if isinstance(value, (list, tuple)):
        if len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        if len(value) == 2:
            return (float(value[0]), float(value[1]), default_z)
    if isinstance(value, dict):
        return (float(value["x"]), float(value["y"]), float(value.get("z", default_z)))
    raise TypeError(f"expected 2/3-element sequence or dict with x/y/z, got {type(value)}")


# ================================================================== #
#  Scene loading                                                       #
# ================================================================== #

def load_spatial_mirror(
    mirror_path: str,
    recon_npz_path: Optional[str] = None,
    da3_model_dir: Optional[str] = None,
) -> SceneState:
    """Load spatial mirror from mirror.json and initialize tool state.

    Args:
        mirror_path: Path to mirror.json produced by build_spatial_mirror.
        recon_npz_path: Optional path to reconstruction NPZ (for BEV / legacy depth).
        da3_model_dir: Optional path to Depth Anything V3 checkpoint directory.

    Returns:
        The initialized SceneState (also accessible via get_state()).
    """
    global _state

    with open(mirror_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _state = SceneState()
    _state.recon_npz_path = recon_npz_path
    _state.da3_model_dir = da3_model_dir

    for inst in data.get("instances", []):
        label = inst["label"].lower()
        centroid = _as_vec3(inst["centroid"])
        size_raw = inst.get("bbox_obb_z", {}).get("size", [0, 0, 0])
        size = _as_vec3(size_raw)

        orientation = None
        ori = inst.get("orientation")
        if ori and "direction_vector" in ori:
            dv = ori["direction_vector"]
            orientation = _as_vec3(dv)
            orientation = _v3_normalize(orientation)

        rec = EntityRecord(
            instance_id=inst["instance_id"],
            label=label,
            position=centroid,
            orientation=orientation,
            size=size,
            centroid=centroid,
            point_count=inst.get("point_count", 0),
            frame_ids=inst.get("frame_ids", []),
        )
        _state.entities[label] = rec
        _state.entities_by_id[inst["instance_id"]] = rec

    if recon_npz_path:
        _state._load_recon()

    print(f"[SpatialTools] Loaded {len(_state.entities)} entities from {mirror_path}")
    return _state


def get_state() -> SceneState:
    """Return the current global SceneState."""
    return _state


def reset_state() -> None:
    """Reset global state (useful between questions)."""
    global _state
    _state = SceneState()


# ================================================================== #
#  Entity lookup                                                       #
# ================================================================== #

def _normalize_text(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.strip().lower())


def _resolve_entity(identifier: str) -> EntityRecord:
    """Resolve an identifier to an EntityRecord.

    Accepts:
      - entity_id as string, e.g. "1"
      - label name, e.g. "stove", "Sofa"
      - camera view like "view1", "image_1" (raises for now — no camera entities)
    """
    target = _normalize_text(identifier)

    # Try integer instance_id
    try:
        iid = int(identifier)
        if iid in _state.entities_by_id:
            return _state.entities_by_id[iid]
    except (ValueError, TypeError):
        pass

    # Try label name
    for rec in _state.entities.values():
        if _normalize_text(rec.label) == target:
            return rec

    # Try partial match
    for rec in _state.entities.values():
        if target in _normalize_text(rec.label):
            return rec

    available = list(_state.entities.keys()) + [str(k) for k in _state.entities_by_id]
    raise KeyError(
        f"entity not found: '{identifier}'. Available: {available}"
    )


# ================================================================== #
#  Tool 1: get_entity                                                  #
# ================================================================== #

def get_entity(identifier: str) -> Dict[str, Any]:
    """Retrieve spatial information for an entity.

    Args:
        identifier: Unique entity_id (int as str), generic label_name (e.g. "chair"),
                    or a camera view reference like "view1" / "image_1".

    Returns:
        Entity dict with:
          - name: entity label
          - instance_id: numeric id
          - position: 3D center coordinates [x, y, z]
          - orientation: 3D facing direction unit vector [x, y, z] (None if unavailable)
          - size: 3D bounding box dimensions [length, width, height]
          - area: footprint area (size[0] * size[1])
    """
    rec = _resolve_entity(identifier)
    result: Dict[str, Any] = {
        "name": rec.label,
        "instance_id": rec.instance_id,
        "position": _v3_to_list(rec.centroid),
        "orientation": _v3_to_list(rec.orientation) if rec.orientation else None,
        "size": _v3_to_list(rec.size),
        "area": rec.size[0] * rec.size[1],
    }
    return result


# ================================================================== #
#  Tool 2: query_depth                                                 #
# ================================================================== #

def set_da3_model_dir(model_dir: str) -> None:
    """Set the DA3 (Depth Anything V3) model directory for depth queries."""
    _state.da3_model_dir = model_dir
    _state.da3_model = None  # force reload
    print(f"[SpatialTools] DA3 model dir set to: {model_dir}")


def _get_or_compute_da3_depth(image_path: str) -> np.ndarray:
    """Load cached DA3 depth or run inference on-demand.

    Depth is cached as {image_stem}_depth.npy next to the image file.

    Returns:
        Depth map as (H, W) numpy array in meters.
    """
    img_p = Path(image_path)
    npy_path = img_p.parent / f"{img_p.stem}_depth.npy"

    if npy_path.exists():
        return np.load(str(npy_path))

    # Run DA3 inference
    _state._load_da3()
    import torch

    prediction = _state.da3_model.inference([str(img_p)])
    depth = prediction.depth[0].cpu().numpy()  # (H, W)

    np.save(str(npy_path), depth)
    print(f"[query_depth] Saved depth map -> {npy_path}")
    return depth


def query_depth(
    pixel_xy: List[int],
    image_path: Optional[str] = None,
) -> float:
    """Query metric depth at a specific pixel using Depth Anything V3.

    On the first call for a given image, DA3 inference runs automatically
    and the result is cached as ``{stem}_depth.npy`` next to the image.

    Args:
        pixel_xy: [x, y] pixel coordinates (0-indexed, x=column, y=row).
        image_path: Path to the image file (.jpg/.png).

    Returns:
        Depth in meters at the specified pixel.

    Raises:
        RuntimeError if DA3 model dir is not configured.
        ValueError if pixel is out of bounds or depth is invalid.
    """
    if image_path is None:
        raise ValueError("query_depth requires an image_path argument")

    depth = _get_or_compute_da3_depth(image_path)
    h, w = depth.shape[:2]

    x, y = int(pixel_xy[0]), int(pixel_xy[1])
    if x < 0 or x >= w or y < 0 or y >= h:
        raise ValueError(f"pixel ({x}, {y}) out of image bounds ({w}, {h})")

    val = float(depth[y, x])
    if not np.isfinite(val) or val <= 1e-6:
        raise ValueError(f"no valid depth at pixel ({x}, {y})")

    return val


# ================================================================== #
#  Tool 3: distance                                                    #
# ================================================================== #

def distance(
    A: Union[str, Vec3, List[float]],
    B: Union[str, Vec3, List[float]],
    mode: str = "center",
) -> float:
    """Compute Euclidean distance between two entities or positions.

    Args:
        A: Entity identifier (str), 3D [x, y, z], or 2D [x, y] (z=0).
        B: Entity identifier (str), 3D [x, y, z], or 2D [x, y] (z=0).
        mode: "center" (centroid-to-centroid) or "closest" (OBB surface distance).

    Returns:
        Distance in meters.
    """
    def _to_pos(val: Union[str, Vec3, List[float]]) -> Vec3:
        if isinstance(val, str):
            return _resolve_entity(val).centroid
        return _as_vec3(val)

    pos_a = _to_pos(A)
    pos_b = _to_pos(B)

    if mode == "center":
        return _v3_norm(_v3_sub(pos_a, pos_b))

    if mode == "closest":
        # Approximate closest surface distance using AABB half-diagonal
        rec_a = _resolve_entity(A) if isinstance(A, str) else None
        rec_b = _resolve_entity(B) if isinstance(B, str) else None
        radius_a = _obb_radius(rec_a) if rec_a else 0.0
        radius_b = _obb_radius(rec_b) if rec_b else 0.0
        center_dist = _v3_norm(_v3_sub(pos_a, pos_b))
        return max(0.0, center_dist - radius_a - radius_b)

    raise ValueError(f"mode must be 'center' or 'closest', got '{mode}'")


def _obb_radius(rec: EntityRecord) -> float:
    """Approximate OBB bounding radius (half-diagonal of size)."""
    sx, sy, sz = rec.size
    return math.sqrt(sx * sx + sy * sy + sz * sz) / 2.0


# ================================================================== #
#  Tool 4: set_egocentric_view                                         #
# ================================================================== #

def set_egocentric_view(
    origin: Union[str, Vec3, List[float]],
    forward: Union[str, Vec3, List[float]],
) -> None:
    """Anchor a new egocentric coordinate reference frame.

    Defines "Where I am" (origin) and "Where I am looking" (forward).
    All subsequent directional operations (turn, relative_direction)
    are based on this frame.

    Supports both 3D and 2D coordinates. When a 2D [x, y] is passed,
    z is assumed to be 0 (ground plane). This makes the tool compatible
    with BEV projections from look_top_down().

    Coordinate convention:
      +x: Right
      +y: Forward (looking direction)
      +z: Up (gravity-aligned)

    Args:
        origin: Entity identifier, 3D [x, y, z], or 2D [x, y] (z=0).
        forward: Entity identifier (uses its orientation), 3D direction
                 vector, or 2D [dx, dy] (z=0, auto-normalized).
    """
    if isinstance(origin, str):
        origin = _resolve_entity(origin).centroid
    else:
        origin = _as_vec3(origin)

    if isinstance(forward, str):
        rec = _resolve_entity(forward)
        if rec.orientation is None:
            raise ValueError(
                f"entity '{forward}' has no orientation. "
                f"Provide a direction vector instead."
            )
        forward = rec.orientation
    else:
        forward = _v3_normalize(_as_vec3(forward))

    # Build orthonormal basis:
    #   +y = forward (horizontal projection for gravity alignment)
    #   +z = world up (0, 0, 1)
    #   +x = right = z × y
    forward_h = _v3_normalize((forward[0], forward[1], 0.0))
    world_up = (0.0, 0.0, 1.0)
    right = _v3_normalize(_v3_cross(world_up, forward_h))
    up = _v3_normalize(_v3_cross(forward_h, right))

    _state.ego = EgocentricFrame(
        origin=origin,
        forward=forward_h,
        right=right,
        up=up,
    )

    print(
        f"[set_egocentric_view] origin={origin}, forward={forward_h}, "
        f"right={right}, up={up}"
    )


# ================================================================== #
#  Tool 5: turn                                                        #
# ================================================================== #

def turn(direction: str) -> None:
    """Rotate the egocentric frame around the gravity axis (+z).

    Args:
        direction: "left" (+90°), "right" (-90°), or "back" (180°).

    Raises:
        RuntimeError if egocentric view is not set.
    """
    if not _state.ego.is_set:
        raise RuntimeError("call set_egocentric_view() before turn()")

    dir_norm = _normalize_text(direction)

    angle_map = {
        "left": math.pi / 2,
        "right": -math.pi / 2,
        "back": math.pi,
    }

    if dir_norm not in angle_map:
        raise ValueError(
            f"direction must be 'left', 'right', or 'back', got '{direction}'"
        )

    angle = angle_map[dir_norm]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    old_fwd = _state.ego.forward
    old_rgt = _state.ego.right

    new_fwd = (
        old_fwd[0] * cos_a - old_fwd[1] * sin_a,
        old_fwd[0] * sin_a + old_fwd[1] * cos_a,
        0.0,
    )
    new_rgt = (
        old_rgt[0] * cos_a - old_rgt[1] * sin_a,
        old_rgt[0] * sin_a + old_rgt[1] * cos_a,
        0.0,
    )
    # Recompute up for consistency
    new_up = _v3_normalize(_v3_cross(new_fwd, new_rgt))

    _state.ego.forward = _v3_normalize(new_fwd)
    _state.ego.right = _v3_normalize(new_rgt)
    _state.ego.up = new_up

    print(f"[turn] {direction} -> forward={_state.ego.forward}")


# ================================================================== #
#  Tool 6: look_top_down                                               #
# ================================================================== #

def look_top_down() -> Dict[str, Any]:
    """Project 3D scene onto a Top-Down (Bird's Eye View) plane.

    Uses the current egocentric frame if set, otherwise falls back to
    world XY plane (Z-up).

    Returns:
        BEV map dict with:
          - origin: [x, y] origin of the BEV frame
          - entities: list of projected entity dicts with
              name, position [x, y], orientation [x, y], size [l, w]
          - scale_m_per_unit: scale factor (meters per coordinate unit)
    """
    if _state.ego.is_set:
        ox, oy = _state.ego.origin[0], _state.ego.origin[1]
        fwd = _state.ego.forward
        rgt = _state.ego.right
    else:
        ox, oy = 0.0, 0.0
        fwd = (0.0, 1.0, 0.0)
        rgt = (1.0, 0.0, 0.0)

    # Project each entity onto the horizontal plane
    projected: List[Dict[str, Any]] = []
    for rec in _state.entities.values():
        dx = rec.centroid[0] - ox
        dy = rec.centroid[1] - oy
        # Local x = dot with right, local y = dot with forward
        lx = dx * rgt[0] + dy * rgt[1]
        ly = dx * fwd[0] + dy * fwd[1]

        ori_2d = None
        if rec.orientation:
            ori_2d = [
                rec.orientation[0] * rgt[0] + rec.orientation[1] * rgt[1],
                rec.orientation[0] * fwd[0] + rec.orientation[1] * fwd[1],
            ]

        projected.append({
            "name": rec.label,
            "instance_id": rec.instance_id,
            "position": [round(lx, 4), round(ly, 4)],
            "orientation": [round(ori_2d[0], 4), round(ori_2d[1], 4)] if ori_2d else None,
            "size": [round(rec.size[0], 4), round(rec.size[1], 4)],
        })

    _state.bev_built = True
    return {
        "origin": [ox, oy],
        "entities": projected,
    }


# ================================================================== #
#  Tool 7: relative_direction                                      #
# ================================================================== #

def relative_direction(
    entity: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the relative spatial placement of a target entity.

    Requires set_egocentric_view() to have been called first.

    Args:
        entity: Entity identifier (str) or an Entity dict from get_entity()
                or look_top_down() (may contain 2D "position").

    Returns:
        Direction dict:
          - left_or_right: "left", "right", or "center"
          - front_or_back: "front", "back", or "center"
          - up_or_down: "up", "down", or "level"
          - four_directions: "front", "back", "left", or "right"
          - angle: rotational angle in degrees relative to +y (forward)
    """
    if not _state.ego.is_set:
        raise RuntimeError("call set_egocentric_view() before relative_direction()")

    # Resolve entity
    if isinstance(entity, str):
        rec = _resolve_entity(entity)
        target_pos = rec.centroid
    elif isinstance(entity, dict):
        pos = entity.get("position")
        if pos is None:
            raise ValueError("entity dict must contain 'position'")
        # Support 2D [x, y] from BEV projection (z=0)
        if len(pos) == 2:
            target_pos = (float(pos[0]), float(pos[1]), 0.0)
        else:
            target_pos = _as_vec3(pos)
    else:
        raise TypeError(f"entity must be str or dict, got {type(entity)}")

    # Vector from ego origin to target
    delta = _v3_sub(target_pos, _state.ego.origin)

    # Project onto ego axes
    along_right = _v3_dot(delta, _state.ego.right)
    along_forward = _v3_dot(delta, _state.ego.forward)
    along_up = _v3_dot(delta, _state.ego.up)

    # Classify left/right
    lateral_thr = 0.05  # meters
    if along_right > lateral_thr:
        left_or_right = "right"
    elif along_right < -lateral_thr:
        left_or_right = "left"
    else:
        left_or_right = "center"

    # Classify front/back
    frontal_thr = 0.05
    if along_forward > frontal_thr:
        front_or_back = "front"
    elif along_forward < -frontal_thr:
        front_or_back = "back"
    else:
        front_or_back = "center"

    # Classify up/down
    vertical_thr = 0.05
    if along_up > vertical_thr:
        up_or_down = "up"
    elif along_up < -vertical_thr:
        up_or_down = "down"
    else:
        up_or_down = "level"

    # Four-directions (dominant horizontal direction)
    if abs(along_forward) > abs(along_right):
        four_directions = "front" if along_forward > 0 else "back"
    else:
        four_directions = "right" if along_right > 0 else "left"

    # Angle relative to forward axis (+y in ego frame), degrees, CW positive
    angle = math.degrees(math.atan2(along_right, along_forward))

    return {
        "left_or_right": left_or_right,
        "front_or_back": front_or_back,
        "up_or_down": up_or_down,
        "four_directions": four_directions,
        "angle": round(angle, 2),
        "projected": {
            "along_right": round(along_right, 4),
            "along_forward": round(along_forward, 4),
            "along_up": round(along_up, 4),
        },
    }


# ================================================================== #
#  Utility: scene summary for prompting                                #
# ================================================================== #

def format_scene_summary() -> str:
    """Format a text summary of all entities for injection into LLM prompts."""
    lines = ["[Scene Entities]"]
    for label, rec in _state.entities.items():
        pos = rec.centroid
        ori_str = (
            f"({rec.orientation[0]:.2f}, {rec.orientation[1]:.2f}, {rec.orientation[2]:.2f})"
            if rec.orientation
            else "N/A"
        )
        lines.append(
            f"  - {rec.label} (id={rec.instance_id})"
        )
    return "\n".join(lines)


# ================================================================== #
#  CLI test                                                            #
# ================================================================== #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spatial Tools CLI")
    parser.add_argument("--mirror", type=str, required=True, help="Path to mirror.json")
    parser.add_argument("--recon", type=str, default=None, help="Path to reconstruction NPZ")
    args = parser.parse_args()

    load_spatial_mirror(args.mirror, args.recon)

    print("\n" + "=" * 60)
    print("Scene Summary")
    print("=" * 60)
    print(format_scene_summary())

    print("\n" + "=" * 60)
    print("Tool Demo")
    print("=" * 60)

    # get_entity
    for label in list(_state.entities.keys())[:3]:
        ent = get_entity(label)
        print(f"\nget_entity('{label}'): {json.dumps(ent, indent=2)}")

    # distance
    labels = list(_state.entities.keys())
    if len(labels) >= 2:
        d = distance(labels[0], labels[1])
        print(f"\ndistance('{labels[0]}', '{labels[1]}'): {d:.4f} m")

    # set_egocentric_view
    if labels:
        first = labels[0]
        ent = get_entity(first)
        if ent["orientation"]:
            print(f"\nset_egocentric_view(origin='{first}', forward='{first}')")
            set_egocentric_view(first, first)

            # turn
            for t_dir in ["left", "right", "back"]:
                turn(t_dir)

            # relative_direction
            for other in labels[1:]:
                rd = relative_direction(other)
                print(f"\nrelative_direction('{other}'): {json.dumps(rd, indent=2)}")

            # look_top_down
            bev = look_top_down()
            print(f"\nlook_top_down(): {json.dumps(bev, indent=2)}")
