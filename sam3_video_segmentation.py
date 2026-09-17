"""SAM3 Video Instance Segmentation with cross-frame object tracking.

Uses Sam3VideoModel (PCS – Promptable Concept Segmentation on video) to
detect and track objects across consecutive frames.  Each label is processed
in its own video session so that object_id → label mapping is unambiguous.

Output format: JSONL detection records + per-frame NPZ masks, with an
``object_id`` field that is consistent across frames for the same tracked entity.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def _ensure_torch_version_string() -> None:
    v = getattr(torch, "__version__", None)
    if not isinstance(v, str) or not v.strip():
        torch.__version__ = "0.0.0"


def get_video_segmentor(model_id: str):
    _ensure_torch_version_string()
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    model = Sam3VideoModel.from_pretrained(model_id)
    return processor, model


def prepare_video_segmentor(segmentor, device: str):
    processor, model = segmentor
    use_cuda = device == "cuda" and torch.cuda.is_available()
    runtime_device = torch.device("cuda" if use_cuda else "cpu")
    model.to(runtime_device, dtype=torch.bfloat16)
    model.eval()
    return processor, model, runtime_device


def list_frame_ids(scene_dir: str) -> List[str]:
    """List frame IDs (numeric stems) from a directory of .jpg files."""
    frame_ids = []
    for name in os.listdir(scene_dir):
        if not name.lower().endswith(".jpg"):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            frame_ids.append(stem)
    frame_ids.sort(key=lambda x: int(x))
    return frame_ids


def load_frames_as_pil(scene_dir: str, frame_ids: List[str]) -> List[Image.Image]:
    """Load frames as PIL images in frame_ids order. Skips unreadable frames."""
    frames = []
    for fid in frame_ids:
        path = os.path.join(scene_dir, f"{fid}.jpg")
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"[SAM3-Video] Frame {fid}: failed to read, skip")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    return frames


@torch.no_grad()
def run_sam3_video_segmentation_for_label(
    segmentor,
    frames: List[Image.Image],
    label: str,
    seg_threshold: float = 0.1,
    min_pixels: int = 64,
    max_frame_num_to_track: Optional[int] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], int]:
    """Run SAM3 video segmentation for a single label across all frames.

    Uses Sam3VideoModel for cross-frame object tracking: the same physical
    object gets a consistent ``object_id`` across frames.

    Args:
        segmentor: ``(processor, model, device)`` tuple.
        frames: List of PIL images (consecutive video frames).
        label: Text prompt for the label to detect & track.
        seg_threshold: Minimum confidence score to keep a raw detection.
        min_pixels: Minimum mask pixel area.
        max_frame_num_to_track: Upper bound on frames to process.

    Returns:
        ``(per_frame_detections, next_object_id_offset)``

        *per_frame_detections* maps ``frame_idx`` (int, 0-based) to a list of
        detection dicts, each containing:
        ``{label, object_id, score, bbox, mask, mask_area}``.

        *next_object_id_offset* is ``max(object_id) + 1`` for this label,
        used to compute globally-unique IDs when processing multiple labels.
    """
    processor, model, device = segmentor

    inference_session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )

    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=label,
    )

    max_track = max_frame_num_to_track if max_frame_num_to_track is not None else len(frames)
    result: Dict[int, List[Dict[str, Any]]] = {}
    max_obj_id = -1

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=max_track,
    ):
        processed = processor.postprocess_outputs(inference_session, model_outputs)
        frame_idx = int(model_outputs.frame_idx)

        object_ids = processed.get("object_ids", [])
        scores = processed.get("scores", [])
        boxes = processed.get("boxes", [])
        masks = processed.get("masks", [])

        dets: List[Dict[str, Any]] = []
        n = len(object_ids)
        for i in range(n):
            # -- mask --
            mask = masks[i]
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask, dtype=bool)
            area = int(mask.sum())
            if area < min_pixels:
                continue

            # -- score --
            s = scores[i]
            score = float(s.detach().cpu().item()) if hasattr(s, "item") else float(s)
            if score < seg_threshold:
                continue

            # -- bbox (XYXY absolute) --
            box = boxes[i]
            if hasattr(box, "detach"):
                box = box.detach().cpu().numpy()
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]

            # -- object_id --
            oid = object_ids[i]
            obj_id = int(oid.detach().cpu().item()) if hasattr(oid, "item") else int(oid)
            max_obj_id = max(max_obj_id, obj_id)

            dets.append({
                "label": label,
                "object_id": obj_id,
                "score": score,
                "bbox": [x1, y1, x2, y2],
                "mask": mask,
                "mask_area": area,
            })

        result[frame_idx] = dets

    return result, max_obj_id + 1


def run_segmentation_for_scene(
    labels: List[str],
    sampled_frames_root: str,
    dataset: str,
    scene_name: str,
    output_root: str = "output",
    seg_model_id: str = "ckpt/sam3",
    seg_device: str = "cuda",
    seg_threshold: float = 0.5,
    seg_min_pixels: int = 64,
    min_score: float = 0.8,
    max_dets: int = 25,
    segmentor=None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Run SAM3 video segmentation on all frames of a scene for the given labels.

    Uses **Sam3VideoModel** (PCS – Promptable Concept Segmentation on video) for
    cross-frame object tracking.  Each label is processed in its own video session
    so that ``object_id`` → ``label`` mapping is unambiguous.  Global ``object_id``
    uniqueness across labels is ensured via offset accumulation.

    Output format is identical to the per-frame approach (JSONL + per-frame NPZ
    masks), with the addition of an ``object_id`` field in every detection record.
    Same ``object_id`` in different frames refers to the same tracked physical entity.

    Two-pass filtering:
      Pass 1 – raw detections collected at ``seg_threshold``.
      Pass 2 – keep detections with score >= ``min_score``; for labels that have
               detections but none >= ``min_score``, fallback to the single best one.

    Returns:
        (filtered_detections_list, scene_output_dir)
    """
    if float(seg_threshold) >= float(min_score):
        print(
            "[SAM3-Video] WARNING: seg_threshold >= min_score, low-score candidates "
            "will be filtered before fallback can kick in. Recommend seg_threshold < min_score."
        )

    scene_dir = os.path.join(sampled_frames_root, dataset, scene_name)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    frame_ids = list_frame_ids(scene_dir)
    if not frame_ids:
        raise RuntimeError(f"No .jpg frames found in {scene_dir}")

    # Output directory
    scene_out_dir = os.path.join(output_root, dataset, scene_name, "sam3_precompute")
    masks_dir = os.path.join(scene_out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Load model (or use pre-loaded)
    if segmentor is not None:
        print("[SAM3-Video] Using pre-loaded segmentor")
    else:
        print("[SAM3-Video] Loading model ...")
        segmentor = prepare_video_segmentor(get_video_segmentor(seg_model_id), seg_device)
    print(f"[SAM3-Video] Ready. Device={segmentor[2]}")

    # Load all frames as PIL images
    print("[SAM3-Video] Loading frames ...")
    frames = load_frames_as_pil(scene_dir, frame_ids)
    if not frames:
        raise RuntimeError(f"No frames could be loaded from {scene_dir}")
    # Trim frame_ids to match successfully loaded frames
    frame_ids = frame_ids[:len(frames)]
    print(f"[SAM3-Video] {len(frames)} frames loaded")

    # ----------------------------------------------------------
    # Pass 1: video segmentation per label → raw detections
    # ----------------------------------------------------------
    all_raw_detections: List[Dict[str, Any]] = []
    object_id_offset = 0

    for label_idx, label in enumerate(labels):
        print(f"[SAM3-Video] Label ({label_idx + 1}/{len(labels)}): '{label}' ...")

        per_frame_dets, next_offset = run_sam3_video_segmentation_for_label(
            segmentor=segmentor,
            frames=frames,
            label=label,
            seg_threshold=seg_threshold,
            min_pixels=seg_min_pixels,
            max_frame_num_to_track=len(frames),
        )

        for frame_idx, dets in per_frame_dets.items():
            if frame_idx >= len(frame_ids):
                continue
            frame_id = frame_ids[frame_idx]
            h, w = frames[frame_idx].size[1], frames[frame_idx].size[0]
            for det in dets:
                global_obj_id = object_id_offset + det["object_id"]
                all_raw_detections.append({
                    "frame_id": frame_id,
                    "label": det["label"],
                    "object_id": global_obj_id,
                    "score": det["score"],
                    "bbox": det["bbox"],
                    "mask_area": det["mask_area"],
                    "mask_file": f"masks/{frame_id}.npz",
                    "mask_key": f"mask_{global_obj_id:03d}",
                    "image_size_hw": [h, w],
                    "_mask": det["mask"],  # kept temporarily for NPZ saving
                })

        object_id_offset += next_offset
        n_det = sum(len(v) for v in per_frame_dets.values())
        print(f"[SAM3-Video]   '{label}': {n_det} raw detections, next_offset={next_offset}")

    print(f"[SAM3-Video] Pass 1 done. {len(all_raw_detections)} total raw detections")

    # ----------------------------------------------------------
    # Save masks per frame (NPZ)
    # ----------------------------------------------------------
    dets_by_frame: Dict[str, List[Dict[str, Any]]] = {}
    for det in all_raw_detections:
        dets_by_frame.setdefault(det["frame_id"], []).append(det)

    for frame_id, frame_dets in dets_by_frame.items():
        mask_pack: Dict[str, np.ndarray] = {}
        for det in frame_dets:
            mask_pack[det["mask_key"]] = det["_mask"].astype(np.uint8)
        if mask_pack:
            np.savez_compressed(os.path.join(masks_dir, f"{frame_id}.npz"), **mask_pack)

    # ----------------------------------------------------------
    # Pass 2: min_score filter + per-label fallback
    # ----------------------------------------------------------
    best_rec_by_label: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    labels_with_any_det: set = set()
    labels_with_high_score_det: set = set()
    filtered_detections: List[Dict[str, Any]] = []

    for det in all_raw_detections:
        score = float(det["score"])
        label = str(det["label"])
        if not label:
            continue
        labels_with_any_det.add(label)
        prev = best_rec_by_label.get(label)
        if prev is None or score > prev[0]:
            best_rec_by_label[label] = (score, det)
        if score >= float(min_score):
            labels_with_high_score_det.add(label)

    for det in all_raw_detections:
        if float(det["score"]) >= float(min_score):
            rec = {k: v for k, v in det.items() if not k.startswith("_")}
            filtered_detections.append(rec)

    # Fallback: labels with detections but none >= min_score → keep best-1
    fallback_labels = sorted(labels_with_any_det - labels_with_high_score_det)
    for label in fallback_labels:
        best_det = best_rec_by_label[label][1]
        rec = {k: v for k, v in best_det.items() if not k.startswith("_")}
        filtered_detections.append(rec)
        print(
            f"[SAM3-Video] Fallback: '{label}' has no high-score det, "
            f"kept best (score={best_det['score']:.4f})"
        )

    # ----------------------------------------------------------
    # Write JSONL + manifest
    # ----------------------------------------------------------
    det_jsonl_path = os.path.join(scene_out_dir, "sam3_detections.jsonl")
    with open(det_jsonl_path, "w", encoding="utf-8") as f:
        for rec in filtered_detections:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "version": "sam3_precompute_v1",
        "mode": "video_tracking",
        "dataset": dataset,
        "scene_id": scene_name,
        "labels": labels,
        "frame_count": len(frame_ids),
        "detection_count": len(filtered_detections),
        "detections_file": "sam3_detections.jsonl",
        "masks_dir": "masks",
        "mask_format": "npz(uint8 0/1)",
    }
    with open(os.path.join(scene_out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[SAM3-Video] Done. {len(filtered_detections)} filtered detections across {len(frame_ids)} frames.")
    print(f"[SAM3-Video] Output -> {scene_out_dir}")

    # Summary per label
    label_det_counts: Dict[str, int] = {}
    for rec in filtered_detections:
        lbl = rec["label"]
        label_det_counts[lbl] = label_det_counts.get(lbl, 0) + 1
    print("[SAM3-Video] Detection counts per label:")
    for lbl, cnt in sorted(label_det_counts.items()):
        print(f"  {lbl}: {cnt}")

    missing = [lbl for lbl in labels if lbl not in label_det_counts]
    if missing:
        print(f"[SAM3-Video] WARNING - labels with zero detections: {missing}")

    # Tracked object summary
    obj_ids_seen: set = set()
    for rec in filtered_detections:
        obj_ids_seen.add(rec.get("object_id", -1))
    print(f"[SAM3-Video] Unique tracked objects: {len(obj_ids_seen)}")

    return filtered_detections, scene_out_dir


# ------------------------------------------------------------------ #
#  Point-prompt segmentation (Sam3Tracker) for missing detection recall #
# ------------------------------------------------------------------ #

def get_point_tracker(model_id: str):
    """Load Sam3TrackerProcessor + Sam3TrackerModel for point-prompt segmentation."""
    _ensure_torch_version_string()
    try:
        from transformers import Sam3TrackerProcessor, Sam3TrackerModel
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import Sam3TrackerProcessor / Sam3TrackerModel. "
            "Ensure transformers >= 5.0.0.dev0 with SAM3 Tracker support."
        ) from exc
    processor = Sam3TrackerProcessor.from_pretrained(model_id)
    model = Sam3TrackerModel.from_pretrained(model_id)
    return processor, model


def prepare_point_tracker(tracker, device: str):
    """Move point tracker to device and set eval mode."""
    processor, model = tracker
    use_cuda = device == "cuda" and torch.cuda.is_available()
    runtime_device = torch.device("cuda" if use_cuda else "cpu")
    model.to(runtime_device)
    model.eval()
    return processor, model, runtime_device


def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Compute XYXY bounding box from a binary mask."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@torch.no_grad()
def run_point_segmentation(
    processor,
    model,
    device: torch.device,
    image_pil: Image.Image,
    point_xy: List[int],
    mask_threshold: float = 0.0,
    min_pixels: int = 64,
) -> Optional[Dict[str, Any]]:
    """Single-frame point-prompt segmentation using Sam3TrackerModel.

    Args:
        processor: Sam3TrackerProcessor.
        model: Sam3TrackerModel.
        device: Torch device.
        image_pil: PIL image.
        point_xy: [x, y] pixel coordinate inside the target object.
        mask_threshold: Minimum mask logit to keep (0 = keep all positive).
        min_pixels: Minimum mask area to accept.

    Returns:
        Dict with {mask, bbox, mask_area} or None if no valid mask.
    """
    x, y = int(point_xy[0]), int(point_xy[1])
    input_points = [[[[x, y]]]]
    input_labels = [[[1]]]

    inputs = processor(
        images=image_pil,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)

    amp_enabled = device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        outputs = model(**inputs, multimask_output=False)

    orig_sizes = inputs.get("original_sizes")
    orig_sizes_list = orig_sizes.tolist() if orig_sizes is not None else None

    masks_list = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        orig_sizes_list,
    )
    pred_masks = masks_list[0]

    if pred_masks.ndim == 4:
        comp_mask = pred_masks[0, 0]
    elif pred_masks.ndim == 3:
        comp_mask = pred_masks[0]
    else:
        comp_mask = pred_masks

    if isinstance(comp_mask, torch.Tensor):
        comp_mask = comp_mask.numpy()
    if comp_mask.dtype in (np.float32, np.float64):
        comp_mask = (comp_mask > mask_threshold).astype(np.uint8)
    else:
        comp_mask = comp_mask.astype(np.uint8)

    comp_mask_bool = comp_mask.astype(bool)
    area = int(comp_mask_bool.sum())
    if area < min_pixels:
        return None

    bbox = _mask_to_bbox(comp_mask_bool)
    if bbox is None:
        return None

    return {
        "mask": comp_mask_bool,
        "bbox": list(bbox),
        "mask_area": area,
    }


def append_point_detections(
    scene_out_dir: str,
    new_detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Append point-supplemented detections to existing JSONL + NPZ files.

    Each detection dict must contain:
        frame_id (str), label (str), mask (np.ndarray bool),
        bbox (list [x1,y1,x2,y2]), mask_area (int), image_size_hw (list [h,w]).
    Optional fields: point (list [x,y]).

    New mask arrays are saved into per-frame NPZ files; JSONL records are
    appended.  Returns the list of appended JSONL records (without numpy data).
    """
    det_jsonl_path = os.path.join(scene_out_dir, "sam3_detections.jsonl")
    masks_dir = os.path.join(scene_out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    jsonl_records: List[Dict[str, Any]] = []

    for det in new_detections:
        frame_id = str(det["frame_id"])
        mask = det.pop("mask")
        mask_npz_path = os.path.join(masks_dir, f"{frame_id}.npz")

        # Load existing mask pack for this frame
        existing_pack: Dict[str, np.ndarray] = {}
        if os.path.isfile(mask_npz_path):
            existing_pack = dict(np.load(mask_npz_path))

        # Compute next available mask_key index
        max_idx = -1
        for k in existing_pack:
            try:
                idx = int(k.replace("mask_", "").replace("obj_", ""))
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
        mask_key = f"mask_{max_idx + 1:03d}"

        # Add new mask and re-save
        existing_pack[mask_key] = mask.astype(np.uint8)
        np.savez_compressed(mask_npz_path, **existing_pack)

        # Build JSONL record (no numpy data)
        rec = {
            "frame_id": frame_id,
            "label": det["label"],
            "score": 1.0,
            "bbox": det["bbox"],
            "mask_area": det["mask_area"],
            "mask_file": f"masks/{frame_id}.npz",
            "mask_key": mask_key,
            "image_size_hw": det.get("image_size_hw", [480, 640]),
            "prompt_type": "point",
        }
        if "point" in det:
            rec["point"] = det["point"]
        jsonl_records.append(rec)

    # Append to JSONL
    with open(det_jsonl_path, "a", encoding="utf-8") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return jsonl_records
