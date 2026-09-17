import os
import ray

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_config import print_llm_status
from workflow.graph import build_workflow

def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("Please set DASHSCOPE_API_KEY environment variable. Cannot call API unconditionally without it.")
        return

    # Try connecting to Ray (assumes `ray start --head` or `serve.start()` is already running)
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except Exception as e:
        print("Ray is not initialized, falling back to local isolated ray initialization.")
        ray.init(ignore_reinit_error=True)

    # Initialize Graph
    print_llm_status()
    app = build_workflow()

    # Define initial state matching demo.py ID 957
    sample_data = {
        "id": 957,
        "dataset": "arkitscenes",
        "scene_name": "41069025",
        "question_type": "object_rel_direction_hard",
        "question": "If I am standing by the stove and facing the sofa, is the tv to my front-left, front-right, back-left, or back-right?\nThe directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis).",
        "options": [
            "A. front-left",
            "B. back-right",
            "C. back-left",
            "D. front-right"
        ]
    }
    
    dataset = sample_data["dataset"]
    scene_name = sample_data["scene_name"]

    initial_state = {
        "question_data": sample_data,
        "api_key": api_key,
        "dataset": dataset,
        "scene_name": scene_name,
        "recon_npz_path": os.path.join("recon", "raw_frames.npz"),
        "sampled_frames_dir": os.path.join("sampled_frames", dataset, scene_name),
        "scene_out_dir": os.path.join("output", dataset, scene_name),
        "instance_output_path": os.path.join("output", dataset, scene_name, "mirror.json"),
        "recall_retries": 0,
        "labels": [],
        "missing_labels": [],
        "instances": [],
        "mirror_dict": {},
        "pseudocode": "",
        "final_result": ""
    }

    print("Running LangGraph Pipeline via Ray Serve...")
    
    # Run the graph
    try:
        # LangGraph invoke returns the final state
        final_state = app.invoke(initial_state)
        print("\n\nPipeline execution complete!")
        print("Final Answer Extracted:", final_state.get("final_result", "Not Found"))
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
