import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional

import torch
import ray
from ray import serve

# ---------------------------------------------------------------------------
# Silence noisy HF/tqdm progress bars (e.g. "Loading weights" spam from
# transformers.core_model_loading).  Only our own print() output is shown.
# ---------------------------------------------------------------------------
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

# ---------------------------------------------------------------------------
# Centralised model weight directory (all weights live under ckpt/)
# ---------------------------------------------------------------------------
CKPT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpt"
)

# We import the heavy logic here so they are loaded onto the GPU workers
from reconstruct_scene import reconstruct_and_align
from sam3_video_segmentation import (
    run_segmentation_for_scene,
    get_video_segmentor,
    prepare_video_segmentor,
    get_point_tracker,
    prepare_point_tracker,
    run_point_segmentation,
)
from build_spatial_mirror import estimate_orientations, _load_orient_model

@serve.deployment(ray_actor_options={"num_gpus": 0.3})
class Pi3XServer:
    def __init__(self):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Pi3XServer] Loading Pi3X model into GPU ...")
        from pi3.models.pi3x import Pi3X
        _pi3x_path = os.path.join(CKPT_ROOT, "pi3x")
        self.model = Pi3X.from_pretrained(_pi3x_path).eval()
        self.model.disable_multimodal()
        self.model = self.model.to(self.device)
        print("[Pi3XServer] Pi3X model loaded ✓")

    def __call__(self, request: Any) -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "ok", "model": "Pi3X", "device": str(self.device)}

    def run_reconstruct(self, raw_frames_dir: str, output_npz_path: str, max_images: int, chunk_size: int, pose_convention: str) -> bool:
        return reconstruct_and_align(
            raw_frames_dir=raw_frames_dir,
            output_npz_path=output_npz_path,
            max_images=max_images,
            chunk_size=chunk_size,
            pose_convention=pose_convention,
            model=self.model,
        )

@serve.deployment(ray_actor_options={"num_gpus": 0.3})
class SAM3Server:
    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _sam3_path = os.path.join(CKPT_ROOT, "sam3")
        self.seg_model_id = os.environ.get("SAM3_SEG_MODEL_ID", _sam3_path)
        self.tracker_model_id = os.environ.get("SAM3_TRACKER_MODEL_ID", _sam3_path)

        print(f"[SAM3Server] Loading SAM3 video segmentor ({self.seg_model_id}) ...")
        self.segmentor = get_video_segmentor(self.seg_model_id)
        self.segmentor = prepare_video_segmentor(self.segmentor, self.device)
        print("[SAM3Server] Video segmentor loaded ✓")

        print(f"[SAM3Server] Loading SAM3 point tracker ({self.tracker_model_id}) ...")
        self.tracker = get_point_tracker(self.tracker_model_id)
        self.tracker_proc, self.tracker_model, _ = prepare_point_tracker(self.tracker, self.device)
        print("[SAM3Server] Point tracker loaded ✓")

    def __call__(self, request: Any) -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "ok", "model": "SAM3", "device": str(self.device)}

    def run_segmentation(self, labels: List[str], sampled_frames_root: str, dataset: str, scene_name: str, output_root: str, seg_model_id: str, seg_device: str):
        return run_segmentation_for_scene(
            labels=labels,
            sampled_frames_root=sampled_frames_root,
            dataset=dataset,
            scene_name=scene_name,
            output_root=output_root,
            seg_model_id=seg_model_id,
            seg_device=seg_device,
            segmentor=self.segmentor,
        )
        
    def run_point_tracking(self, img_path: str, point_px: List[int], model_id: str, device: str):
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        return run_point_segmentation(
            self.tracker_proc, self.tracker_model,
            torch.device(self.device), img, point_px
        )

@serve.deployment(ray_actor_options={"num_gpus": 0.3})
class OrientV2Server:
    def __init__(self):
        print("[OrientV2Server] Loading Orient-Anything-V2 model ...")
        _orient_ckpt = os.path.join(
            CKPT_ROOT, "orianyv2", "demo_ckpts", "rotmod_realrotaug_best.pt"
        )
        self.model, self.device = _load_orient_model(_orient_ckpt)
        print(f"[OrientV2Server] Orient model loaded on {self.device} ✓")

    def __call__(self, request: Any) -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "ok", "model": "OrientV2", "device": str(self.device)}

    def run_estimation(self, instances: List[Dict], sam3_output_dir: str, sampled_frames_dir: str, recon_npz_path: str, pose_convention: str):
        return estimate_orientations(
            instances=instances,
            sam3_output_dir=sam3_output_dir,
            sampled_frames_dir=sampled_frames_dir,
            recon_npz_path=recon_npz_path,
            pose_convention=pose_convention,
            model=(self.model, self.device),
        )

# Application builder - use .bind() + serve.run() for modern Ray Serve
def build_app():
    # Bind all deployments into a single Serve application graph
    # Each deployment is independently accessible
    return {
        "pi3x": Pi3XServer.bind(),
        "sam3": SAM3Server.bind(),
        "orient_v2": OrientV2Server.bind(),
    }

# Deploy all models
def deploy_all():
    ray.init(ignore_reinit_error=True)
    # Clean up any leftover apps from previous deployments (e.g. old "default" app)
    for app_name in ["default", "pi3x_app", "sam3_app", "orient_app"]:
        try:
            serve.delete(app_name)
        except Exception:
            pass  # ignore if app doesn't exist

    # Each deployment needs a unique name + unique route_prefix to coexist
    serve.run(Pi3XServer.bind(), name="pi3x_app", route_prefix="/pi3x")
    serve.run(SAM3Server.bind(), name="sam3_app", route_prefix="/sam3")
    serve.run(OrientV2Server.bind(), name="orient_app", route_prefix="/orient")
    print("All 3 Ray Serve models deployed successfully:")
    print("  - Pi3XServer  → http://127.0.0.1:8000/pi3x")
    print("  - SAM3Server  → http://127.0.0.1:8000/sam3")
    print("  - OrientV2Server → http://127.0.0.1:8000/orient")

if __name__ == "__main__":
    deploy_all()
    # keep alive
    import time
    while True:
        time.sleep(1000)
