import os
import json
from typing import Dict, Any

from workflow.state import GraphState

import ray
from ray import serve
from llm_config import get_llm_client, get_llm_config, resolve_model_name

# Core pipeline imports (non-GPU dependent or orchestrated indirectly)
from sample_keyframes import sample_keyframes
from api_messages import (
    get_perception_proposal,
    infer_point_for_missing_label,
    generate_pseudocode,
    rewrite_and_execute
)
from build_spatial_mirror import (
    build_spatial_mirror,
    find_duplicate_labels,
    deduplicate_instances,
    rebuild_mirror_dict
)
from sam3_video_segmentation import list_frame_ids, append_point_detections

def reconstruct_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 0: Scene 3D Reconstruction (Ray Serve)\n{'='*60}")
    
    # Get Ray Serve handle for Pi3XServer (app name: pi3x_app)
    handle = serve.get_app_handle("pi3x_app")
    ok = handle.run_reconstruct.remote(
        raw_frames_dir="raw_frames",
        output_npz_path=state["recon_npz_path"],
        max_images=500,
        chunk_size=64,
        pose_convention="c2w"
    ).result()
    
    if not ok:
        raise RuntimeError("[Recon] Scene reconstruction failed.")
        
    return {} # No direct state update needed beyond saving to disk

def sample_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 0.5: Geometric Keyframe Sampling\n{'='*60}")
    selected_ids = sample_keyframes(
        recon_npz_path=state["recon_npz_path"],
        raw_frames_dir="raw_frames",
        output_dir=state["sampled_frames_dir"],
        target_frames=64,
        rotation_weight=5,
    )
    if not selected_ids:
        raise RuntimeError("[Sample] Keyframe sampling failed.")
    return {}

def proposal_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 1: Perception Proposal (Question ID {state['question_data']['id']})\n{'='*60}")
    labels = get_perception_proposal(
        state["question_data"], 
        state["api_key"], 
        sampled_frames_dir=state["sampled_frames_dir"]
    )
    print(f"Extracted Labels: {labels}")
    if not labels:
        raise RuntimeError("No labels extracted.")
    return {"labels": labels}

def segment_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 2: SAM3 Segmentation (Ray Serve)\n{'='*60}")
    handle = serve.get_app_handle("sam3_app")
    result = handle.run_segmentation.remote(
        labels=state["labels"],
        sampled_frames_root="sampled_frames",
        dataset=state["dataset"],
        scene_name=state["scene_name"],
        output_root="output",
        seg_model_id="ckpt/sam3",
        seg_device="cuda"
    ).result()
    all_detections, scene_out_dir = result
    
    return {"scene_out_dir": scene_out_dir}

def build_mirror_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 3: Build Spatial Mirror\n{'='*60}")
    instances, mirror_dict = build_spatial_mirror(
        sam3_output_dir=state["scene_out_dir"],
        recon_npz_path=state["recon_npz_path"],
        labels=state["labels"],
        pose_convention="c2w",
        min_score=0.0,
        dist_threshold_m=0.8,
        max_depth_m=20.0,
    )
    return {"instances": instances, "mirror_dict": mirror_dict}

def check_missing_node(state: GraphState) -> Dict[str, Any]:
    # Corresponds to check_missing_labels
    labels = state["labels"]
    instances = state["instances"]
    
    found = set()
    for inst in instances:
        found.add(inst["label"])
        if "label_aliases" in inst:
            for al in inst["label_aliases"]:
                found.add(al)

    missing = []
    for lbl in labels:
        if lbl not in found:
            missing.append(lbl)
            
    return {"missing_labels": missing}

def recall_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 3.5: Missing Detection Recall\n{'='*60}")
    missing_labels = state["missing_labels"]
    print(f"Missing labels: {missing_labels}")

    # Use unified LLM client (supports both API and local vLLM)
    cfg = get_llm_config()
    client = get_llm_client(api_key=state["api_key"])
    vlm_model = resolve_model_name(cfg.default_vlm_model)

    handle = serve.get_app_handle("sam3_app")
    
    fids = sorted(list_frame_ids(state["sampled_frames_dir"]))
    total_new = 0
    max_frames_per_label = 5
    
    for label in missing_labels:
        print(f"\n[Recall] Attempting to recall: '{label}'")
        sampled_fids = fids[::max(1, len(fids) // max_frames_per_label)][:max_frames_per_label]
        new_dets = []
        
        for fid in sampled_fids:
            img_path = os.path.join(state["sampled_frames_dir"], f"{fid}.jpg")
            if not os.path.exists(img_path):
                continue
            
            point_px = infer_point_for_missing_label(client, vlm_model, img_path, label)
            if not point_px:
                continue
                
            res = handle.run_point_tracking.remote(
                img_path=img_path,
                point_px=point_px,
                model_id="ckpt/sam3",
                device="cuda"
            ).result()
            
            if res:
                new_dets.append({
                    "frame_id": fid,
                    "label": label,
                    "point": point_px,
                    "mask": res["mask"],
                    "bbox": res["bbox"],
                    "mask_area": res["mask_area"],
                    "image_size_hw": res.get("image_size_hw", [1024, 1024]) # Mock or return from model
                })
        
        if new_dets:
            appended = append_point_detections(state["scene_out_dir"], new_dets)
            total_new += len(appended)
            
    return {"recall_retries": state.get("recall_retries", 0) + 1}

def dedup_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 4: Multi-Instance Deduplication\n{'='*60}")
    dup_labels = find_duplicate_labels(state["instances"])
    if dup_labels:
        print(f"Duplicate labels: {list(dup_labels.keys())}")
        cfg = get_llm_config()
        dedup_client = get_llm_client(api_key=state["api_key"])
        vlm_model = resolve_model_name(cfg.default_vlm_model)
        instances, dropped = deduplicate_instances(
            instances=state["instances"],
            sam3_output_dir=state["scene_out_dir"],
            sampled_frames_dir=state["sampled_frames_dir"],
            client=dedup_client,
            model_name=vlm_model,
        )
        if dropped > 0:
            mirror_dict = rebuild_mirror_dict(instances, state["mirror_dict"]["stats"])
            mirror_dict["stats"]["dedup_dropped"] = dropped
            return {"instances": instances, "mirror_dict": mirror_dict}
    return {}

def orient_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 4.5: Orientation Estimation (Ray Serve)\n{'='*60}")
    handle = serve.get_app_handle("orient_app")
    instances = handle.run_estimation.remote(
        instances=state["instances"],
        sam3_output_dir=state["scene_out_dir"],
        sampled_frames_dir=state["sampled_frames_dir"],
        recon_npz_path=state["recon_npz_path"],
        pose_convention="c2w"
    ).result()
    
    mirror_dict = rebuild_mirror_dict(instances, state["mirror_dict"]["stats"])
    
    # Save mirror.json
    out_path = state["instance_output_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": state["dataset"],
            "scene_name": state["scene_name"],
            "question_id": state["question_data"]["id"],
            "labels": state["labels"],
            "instances": instances,
            "stats": mirror_dict["stats"],
        }, f, ensure_ascii=False, indent=2)
        
    return {"instances": instances, "mirror_dict": mirror_dict}

def codegen_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 5: Pseudocode Generation\n{'='*60}")
    pseudocode = generate_pseudocode(
        question_data=state["question_data"],
        instances=state["instances"],
        api_key=state["api_key"],
        model_name="qwen3.5-plus",
    )
    
    if pseudocode:
        out_path = os.path.join(
            "output", state["dataset"], state["scene_name"], "pseudocode.py"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pseudocode)
            
    return {"pseudocode": pseudocode}

def execute_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n{'='*60}\nStage 6: Code Execution\n{'='*60}")
    script_output_dir = os.path.join(
        "output", state["dataset"], state["scene_name"], "scripts"
    )
    
    try:
        pred = rewrite_and_execute(
            pseudocode=state["pseudocode"],
            instances=state["instances"],
            question_data=state["question_data"],
            api_key=state["api_key"],
            script_output_dir=script_output_dir,
            mirror_json_path=state["instance_output_path"],
            model_name="qwen3.5-plus",
        )
        if pred:
            print(f"\nFINAL RESULT: {pred}\n")
            return {"final_result": str(pred)}
    except Exception as e:
        print(f"Error during execution: {e}")
        
    return {}
