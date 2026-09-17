from typing import TypedDict, Dict, Any, List, Optional

class GraphState(TypedDict):
    """
    GraphState holds the structured data passed across LangGraph nodes.
    """
    question_data: Dict[str, Any]
    api_key: str
    
    # Paths and configurations
    dataset: str
    scene_name: str
    recon_npz_path: str
    sampled_frames_dir: str
    scene_out_dir: str
    instance_output_path: str
    
    # Pipeline data
    labels: List[str]
    missing_labels: List[str]
    instances: List[Dict[str, Any]]
    mirror_dict: Dict[str, Any]
    
    # Generated outputs
    pseudocode: str
    final_result: str
    
    # Control flow
    recall_retries: int
