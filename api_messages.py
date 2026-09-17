"""API message construction and LLM call interfaces for CogSpace pipeline.


Supports two LLM backends:
  - "api"  : Cloud API (DashScope / OpenAI) — requires DASHSCOPE_API_KEY or OPENAI_API_KEY
  - "local": Local vLLM deployment   — requires VLLM_BASE_URL pointing to vLLM server

Switch via environment variable: export LLM_MODE=local
Or programmatically: from llm_config import get_llm_client, get_llm_config
"""

import json
import os
import re
import subprocess
import time
from typing import Optional

from openai import OpenAI
from llm_config import get_llm_client, get_llm_config, resolve_model_name


# ============================================================
# Shared helpers
# ============================================================


def _extract_code_block(text: str) -> Optional[str]:
    """Extract pseudocode from <code>...</code> tags, fallback to ```python```."""
    if not text:
        return None
    m = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ============================================================
# Stage 1: Perception Proposal (LLM)
# ============================================================

def _count_sampled_frames(frames_dir: str) -> int:
    """Count image files in the sampled frames directory."""
    if not os.path.isdir(frames_dir):
        return 0
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return sum(
        1 for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    )


def get_perception_proposal(
    question_data: dict,
    api_key: str,
    sampled_frames_dir: str = "",
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> list:
    """Ask LLM to propose perception targets (entity labels) for the question.

    Args:
        question_data: Question dict with 'question', 'options'.
        api_key: LLM API key.
        sampled_frames_dir: Path to sampled frames directory (used to count views).
        api_base_url: API base URL (legacy param; overridden by llm_config in local mode).

    Returns:
        List of entity label strings to be segmented by SAM3.
    """
    cfg = get_llm_config()
    # Use unified client factory (respects LLM_MODE env var)
    client = get_llm_client(mode=None, api_key=api_key or None, base_url=api_base_url)
    model = resolve_model_name(cfg.default_chat_model)

    system_prompt = """You are a spatial perception planner for 3D reasoning. Your only job is to propose what should be perceived for solving the 3D reasoning task. Do NOT answer the VQA question directly. The goal is to extract the spatial evidence required by dispatching tool calls.

[API Specification]
Tool: OP_perceive(instance_id, entity)
- instance_id (integer): A unique numerical identifier for the perception call. Distinguishes different instances of the same label if multiple exist, and differentiates independent parallel calls.
- entity: The core entity of interest. This includes objects (e.g., "black sneaker"), regions (e.g., area, rooms), and cameras (e.g., a specific frame index or a viewpoint).
  - name (string): The object category label or the view token (e.g., "View 1").
  - kind (string): Must be either "object" or "view".

[Planning Rules]
1. Comprehensiveness: Extract all entities explicitly mentioned or implicitly needed to solve the question, including target objects, reference objects, and necessary view anchors. Do not add inferred/new labels unnecessarily, unless they are explicitly observed in the visual input and deemed necessary.
2. Prioritization: Prioritize object entities. Add view targets only when viewpoint transitions or camera poses are explicitly involved in the query. 
3. Dispatch Alignment: Dispatch one target per function call. Generate multiple parallel independent calls if several entities need parsing.
4. Efficiency:  Keep tool calls minimal but complete; thoroughly avoid redundant or duplicate proposals.

[Output Format Constraints]
Generate perception proposals strictly as simulated tool calls in standard JSON format. Return JSON only. No markdown formatting outside the JSON, and no extra reasoning text or explanations.

[Output JSON Schema Example]
{
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "OP_perceive",
        "arguments": {
          "instance_id": 1,
          "entity": {
            "name": "black sneaker",
            "kind": "object"
          }
        }
      }
    }
  ]
}"""

    options_text = "\n".join(question_data.get("options", []))
    num_views = max(_count_sampled_frames(sampled_frames_dir), 1)
    views_tokens = " ".join(f"<view_{i+1}>" for i in range(num_views))
    user_prompt = f"""[Views]
{views_tokens}
|[Question]
{question_data['question']}
|[Options]
{options_text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )

        msg_content = response.choices[0].message.content.strip()

        # Clean up possible markdown code blocks
        if msg_content.startswith("```json"):
            msg_content = msg_content[7:]
        elif msg_content.startswith("```"):
            msg_content = msg_content[3:]
        if msg_content.endswith("```"):
            msg_content = msg_content[:-3]

        parsed_json = json.loads(msg_content.strip())

        print(">>> Raw API Response Received:")
        print(json.dumps(parsed_json, indent=2))

        labels = []
        if "tool_calls" in parsed_json:
            for call in parsed_json["tool_calls"]:
                try:
                    args = call["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    entity_name = args.get("entity", {}).get("name")
                    if entity_name:
                        labels.append(entity_name)
                except Exception as e:
                    print(f"Error parsing call: {e}")

        return labels

    except Exception as e:
        print(f"API Error: {e}")
        return []


# ============================================================
# Stage 3.5: Missing Detection Recall — VLM point inference
# ============================================================

def infer_point_for_missing_label(
    client: OpenAI,
    model_name: str,
    image_path: str,
    label: str,
) -> Optional[list]:
    """Use VLM to infer a 2D point for a missing label in a single frame.

    Returns:
        [x, y] in [0, 1000] range, or None if object not found in frame.
    """
    import base64

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""[Views]
<view_1>

[Task]
Given the query '{label}', for each frame, detect and localize the visual content described by the given textual query. If the visual content does not exist in a frame, skip that frame. Output ONLY a targeted point coordinate `[x, y]` inside the object in standard JSON format where x and y are integers in [0, 1000].

[Output JSON Schema Example]
{{
  "tool_calls": [
    {{
      "type": "function",
      "function": {{
        "name": "OP_perceive",
        "arguments": {{
          "instance_id": ,
          "entity": {{
            "name": "{label}",
            "kind": "object"
          }},
          "hint2d": {{
            "type": "point",
            "value": [100, 200],
            "frame_index": "view_1"
          }}
        }}
      }}
    }}
  ]
}}"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=256,
    )

    text = response.choices[0].message.content.strip()

    # Try parsing as tool_calls JSON first
    block = _extract_json_block(text)
    if block:
        try:
            obj = json.loads(block)
            for call in obj.get("tool_calls", []):
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                hint2d = args.get("hint2d", {})
                if hint2d.get("type") == "point":
                    val = hint2d.get("value")
                    if isinstance(val, list) and len(val) == 2:
                        return val
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return None


# ============================================================
# Stage 5: Pseudocode Generation (Tool-augmented reasoning)
# ============================================================

def build_entity_summary(instances: list) -> str:
    """Format spatial mirror instances as entity candidate summary for LLM."""
    lines = []
    for inst in instances:
        name = inst["label"]
        iid = inst["instance_id"]
        lines.append(f'- Entity(id={iid}, name="{name}")')
    return "\n".join(lines)


def generate_pseudocode(
    question_data: dict,
    instances: list,
    api_key: str,
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "qwen3.5-plus",
    category_hint: Optional[str] = None,
) -> Optional[str]:
    """Stage 5: Generate tool-augmented pseudocode for spatial reasoning.

    Args:
        question_data: Question dict with 'question', 'options', 'question_type'.
        instances: Spatial mirror instance list.
        api_key: LLM API key.
        api_base_url: API base URL.
        model_name: Model name.
        category_hint: Optional category-specific guidance text.

    Returns:
        Extracted pseudocode string, or None on failure.
    """
    cfg = get_llm_config()
    client = get_llm_client(mode=None, api_key=api_key or None, base_url=api_base_url)
    model = resolve_model_name(model_name)

    # System persona
    system_prompt = (
        "You are a spatial reasoning algorithm assistant. Your task is to produce rigorous pseudocode plans with explicit tool calls. Your goal is not to guess the answer directly, but to design a pseudocode sequence using the provided tools. Prioritize using the given tools and maintain rigorous, reproducible steps."
    )

    # Tool specification
    tool_spec = """\
[Usage Instructions]
You are given an abstract scene summary and must plan a tool-driven solution. You are NOT allowed to answer the VQA question directly. You have access to the following spatial tools. Treat them as "callable black-box functions". Do not assume hidden capabilities beyond these definitions. Plan strictly according to the inputs and outputs; implement any additional geometric logic using simple code if necessary.

[Coordinate System Setup]
Egocentric Reference Frame: Once initialized, the system relies on an egocentric coordinate system where:
+x axis: Right
+y axis: Forward (view facing direction)
+z axis: Upward (gravity-aligned)


[Geometric APIs]

[Tool 1] get_entity(identifier)

Input: identifier (string, either a unique entity_id, a generic label_name like "chair", or a camera view like "view1" or "image_1" indicating the camera at a specific frame).
Output: A single Entity object or a list of Entity objects containing:
position: 3D center coordinates.
orientation: 3D unit vector indicating facing direction.
size: 3D bounding box dimensions (length, width, height) (applicable only to objects).
area: Footprint area (specific to rooms).
Usage: Must be called first to retrieve information before referencing any entity.

[Tool 2] query_depth(pixel_xy)

Input: pixel_xy (list of two integers [x, y]).
Output: Depth value at the specified pixel coordinates in meters.
Usage: Fetch 2D-to-3D depth information for target pixels from the visual observation (used only for single-image inputs).

[Tool 3] distance(A, B, mode="center")

Input: A, B (Entity or 3D positions) and mode ("center" or "closest").
Output: Euclidean distance between A and B in meters.
Usage: Calculate the distance between two Entity or locations. Useful for deciding "closer/further" relationships, distance comparison, or thresholding. If the mode parameter is not provided, it defaults to "center".


[Cognitive APIs]

[Tool 4] set_egocentric_view(origin, forward)

Input: origin (3D coordinate, e.g., an entity's position), forward (3D direction vector, e.g., an entity's orientation).
Output: None.
Usage: Simulates the human perspective by anchoring a new egocentric coordinate reference frame representing "Where I am" (origin) and "Where I am looking" (forward). You must use this tool to align the reasoning frame before performing relative direction checks. All subsequent directional operations are based on this frame.

[Tool 5] turn(direction)

Input: direction ("left", "right", or "back").
Output: None.
Usage: Executes a rotational transformation around the gravity axis within the current perspective. "left" represents +90°, "right" represents -90°, and "back" represents 180°.


[Tool 6] look_top_down()

Input: None.
Output: None.
Usage: Projects the current 3D spatial representation onto a Top-Down (Bird's Eye View) plane to construct a 2D cognitive map for planar spatial relationship queries.


[Tool 7] get_relative_direction(entity)

Input: entity (an Entity object).
Output: A structured Direction dictionary containing:
left_or_right: "left", "right", or "center"
front_or_back: "front", "back", or "center"
up_or_down: "up", "down", or "level"
four_directions: "front", "back", "left", or "right"
angle: Rotational angle relative to the current egocentric x-axis.
Usage: Calculates the relative spatial placement and orientation angle constraints of the target entity explicitly reliant on the current egocentric view."""

    # Build entity candidates summary
    entity_summary = build_entity_summary(instances)

    options_text = "\n".join(question_data.get("options", []))

    category_hint ="forward direction is from stove to sofa"

    parts = [
        "[Task]",
        "Generate python-style pseudocode for solving the spatial reasoning question.",
        "",
        "[Question]",
        question_data["question"],
        "",
        "[Options]",
        options_text,
        "",
        "[Entity Candidates Summary]",
        entity_summary,
        "",
        "[Tool Specification]",
        tool_spec,
    ]

    if category_hint:
        parts += [
            "",
            "[Category-Specific Guidance]",
            category_hint,
        ]

    # Append merged quality checklist & output constraints
    parts += [
        "",
        "[Output Constraints & Quality Checklist]",
        "- Output pseudocode ONLY and wrap it in `<code>...</code>` tags. Do not output any conversational prose or reasoning text outside the `<code>` block.",
        "- Do not use standard Markdown code fences (avoid ```). Do not provide the final answer choice directly.",
        "- If options are provided, ensure the calculated result is explicitly mapped to A/B/C/D.",
        "- Keep comments concise while writing the pseudocode.",
        "- Ensure required tool calls are explicit. Do not call any tools outside the specified list.",
        "- Ensure view orientations are established (via set_egocentric_view) before relative directions are queried (via get_relative_direction).",
        "- Ensure reference frame handling (world coordinate vs. local coordinate) is logically valid.",
        "- Ensure view rotations and coordinate movements are handled separately and correctly.",
        "- Keep the exact scene naming convention for views and objects as provided.",
    ]

    user_prompt = "\n".join(parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            extra_body={"enable_thinking": False},
        )

        raw_text = response.choices[0].message.content.strip()
        print(f"\n>>> Raw pseudocode response ({len(raw_text)} chars):")
        print(raw_text[:2000])
        if len(raw_text) > 2000:
            print(f"  ... (truncated, total {len(raw_text)} chars)")

        pseudocode = _extract_code_block(raw_text)
        if pseudocode is None:
            print("[Pseudocode] WARNING: Could not extract <code> block from response.")
            print("[Pseudocode] Using full response as pseudocode (may need manual review).")
            pseudocode = raw_text

        return pseudocode

    except Exception as e:
        print(f"[Pseudocode] API Error: {e}")
        return None


# ============================================================
# Stage 6: Code Execution (Pseudocode → Executable Script)
# ============================================================

MAX_REWRITE_ATTEMPTS = 4


def _run_isolated_script(script_path: str, timeout: int = 30):
    """Run a Python script in an isolated subprocess.

    Returns (return_code, stdout, stderr).
    """
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        res = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "Execution timed out."
    except Exception as e:
        return 1, "", str(e)


def _extract_pred_from_stdout(stdout: str) -> Optional[str]:
    """Extract the predicted answer from 'Pred: XXX' line."""
    for line in stdout.split("\n"):
        if line.startswith("Pred:"):
            return line.replace("Pred:", "").strip()
    return None


def rewrite_and_execute(
    pseudocode: str,
    instances: list,
    question_data: dict,
    api_key: str,
    script_output_dir: str,
    mirror_json_path: str,
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name: str = "qwen3.5-plus",
    max_attempts: int = MAX_REWRITE_ATTEMPTS,
) -> Optional[str]:
    """Stage 6: Rewrite pseudocode into executable Python and run it.


    Args:
        pseudocode: Pseudocode string from Stage 5.
        instances: Spatial mirror instance list (used to build cogmap JSON).
        question_data: Question dict with 'question', 'options'.
        api_key: LLM API key.
        script_output_dir: Directory to save the generated script.
        mirror_json_path: Path to mirror.json for the generated script.
        api_base_url: API base URL.
        model_name: Code-generation model name.
        max_attempts: Max rewrite attempts (1 initial + retries).

    Returns:
        Predicted answer string, or None on failure.
    """
    cfg = get_llm_config()
    client = get_llm_client(mode=None, api_key=api_key or None, base_url=api_base_url)
    model = resolve_model_name(model_name)

    system_prompt = (
        "You are an expert Python developer specialized in converting spatial reasoning pseudocode into robust, executable Python scripts."
    )

    # Read spatial_tools.py source code
    tools_src_path = os.path.join(os.path.dirname(__file__), "spatial_tools.py")
    with open(tools_src_path, "r", encoding="utf-8") as f:
        spatial_tools_code = f.read()

    base_prompt = f"""
[Task]
Your task is to rewrite a piece of pseudocode into an executable Python script.

Here is the exact source code of `spatial_tools.py`. You MUST use these functions and strictly respect their implementations and return data structures:
```python
{spatial_tools_code}
```

[Available Libraries]
You can use "numpy", "scipy", "math", and other standard Python libraries.

[Implementation Rules]
1. Begin with imports: `from spatial_tools import *` and `import json`.
2. Initialize the scene by calling `load_spatial_mirror(r\"{mirror_json_path}\")` — this loads all entities from the mirror.json file into the global state. DO NOT try to manually populate entities.
3. After initialization, call tool functions directly: `get_entity("entity_name")`, `set_egocentric_view(origin, forward)`, `get_relative_direction("entity_name")`, etc. These functions operate on the global state.
4. `get_entity(identifier)` returns a dict with keys: name, instance_id, position (list), orientation (list or None), size (list), area (float).
5. `set_egocentric_view(origin, forward)` takes origin as [x,y,z] list and forward as [x,y,z] list. It modifies the global state in-place.
6. `turn(direction)` takes a string like "left", "right", "back". It modifies the global state in-place.
7. `get_relative_direction(entity)` takes an entity identifier string and returns a dict with keys: left_or_right, front_or_back, up_or_down, four_directions, angle.
8. For multiple-choice, ensure the calculated result is mapped to A/B/C/D and the final printed prediction is a single clean label token.
9. Output ONLY the Python code block (starting with ```python). Print the result EXACTLY as: print(f"Pred: {{{{ans}}}}").

--- Inputs ---
PSEUDOCODE TO REWRITE:
{pseudocode}
"""

    os.makedirs(script_output_dir, exist_ok=True)
    script_path = os.path.join(script_output_dir, "exec_script.py")

    prev_code = None
    for attempt_idx in range(1, max_attempts + 1):
        # Build prompt: base + retry context on failure
        if attempt_idx > 1 and prev_code is not None:
            retry_context = (
                f"\n\n--- Retry Context ---\n"
                f"Attempt {attempt_idx - 1} failed (exit code: {last_rc}).\n\n"
                f"Captured stderr / traceback (raw):\n```text\n{last_stderr}\n```\n\n"
                f"Previously generated script (failed version):\n```python\n{prev_code}\n```\n\n"
                "Action required:\n"
                "- Inspect the traceback and identify the root cause(s) and failing line(s).\n"
                "- Fix the underlying issue(s); do not merely patch around the symptom.\n"
                "- Preserve required imports (`from spatial_tools import *`) and strictly use the provided `spatial_tools` API.\n"
                "- Do NOT try to manually initialize entities — always use `load_spatial_mirror(path)`.\n"
                "- Add defensive checks when accessing dictionary fields or optional values.\n"
                "- Ensure the script deterministically computes the final answer and prints exactly\n"
                f'  `print(f"Pred: {{{{ans}}}}")`.\n\n'
                "Now rewrite the full script from scratch addressing the errors above. "
                "Output ONLY one complete python code block (```python ... ```). Do not include any analysis."
            )
            user_prompt = base_prompt + retry_context
        else:
            user_prompt = base_prompt

        # Call LLM to generate code
        print(f"  [Code] Attempt {attempt_idx}/{max_attempts}: generating executable script ...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
                extra_body={"enable_thinking": False},
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Code] API Error: {e}")
            continue

        # Extract ```python ... ``` block
        match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            code = content.replace("```python", "").replace("```", "").strip()

        # Save script
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Execute in isolated subprocess
        rc, stdout, stderr = _run_isolated_script(script_path)
        last_rc, last_stderr = rc, (stderr or "").strip()
        prev_code = code

        if rc == 0:
            pred = _extract_pred_from_stdout(stdout)
            if pred is not None:
                print(f"  [Code] Execution successful on attempt {attempt_idx}.")
                print(f"  [Code] Pred: {pred}")
                return pred
            else:
                print(f"  [Code] Script ran (rc=0) but no 'Pred:' line found in stdout.")
                print(f"  [Code] stdout: {stdout[:500]}")
        else:
            print(f"  [Code] Attempt {attempt_idx} failed (rc={rc}).")
            print(f"  [Code] stderr: {(stderr or '').strip()[:300]}")

        time.sleep(1)  # rate-limit

    print(f"  [Code] All {max_attempts} attempts exhausted.")
    return None


# ============================================================
# Stage 4: Instance Deduplication — VLM judging
# ============================================================

def _extract_json_block(text: str) -> Optional[str]:
    """Extract the first JSON object from raw VLM text."""
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("{") and t.endswith("}"):
        return t
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        return t[i: j + 1]
    return None


def judge_best_instance(
    client: OpenAI,
    model_name: str,
    label: str,
    candidate_image_paths: list,
    candidate_summary: list,
) -> Optional[dict]:
    """Ask VLM to pick the single best instance among duplicates.

    Args:
        client: OpenAI-compatible client.
        model_name: VLM model name.
        label: Object label being deduplicated.
        candidate_image_paths: List of rendered candidate image file paths.
        candidate_summary: List of dicts with instance_id, point_count, frame_count, etc.

    Returns:
        {"keep_instance_id": int, "reason": str} or None on failure.
    """
    import base64

    system_prompt = (
        "You are an instance quality judge. "
        "Given candidates of the same object label, choose exactly one best instance. "
        "Return JSON only."
    )

    cand_text = json.dumps(candidate_summary, ensure_ascii=False, indent=2)
    user_text = (
        f"[Object Label]\n{label}\n\n"
        f"[Candidates]\n{cand_text}\n\n"
        f"[Task Guidelines]\n"
        f"- [Task]: Choose exactly ONE most reliable instance for the object label "
        f"based on the numerical ID annotations. Only judge segmentation instance quality.\n"
        "[Rules]"
        f"- Candidates belong to the same label. You must pick exactly one instance. Since the target might be ambiguous or the candidates may contain segmentation errors of the wrong category, you must compare the candidates and select the one that is the most correct and salient.\n"
        f"- Pick exactly one appropriate instance ID.\n"
        f"- Remove obvious duplicate, noisy, or fragmented instances. "
        f"Prefer candidates with relatively better bounding box alignment.\n"
        f"- Return JSON only.\n\n"
        f'[Output JSON Schema Example]\n'
        f'{{\n'
        f'  "reason": "<brief reason>",\n'
        f'  "tool_calls": [\n'
        f'    {{\n'
        f'      "type": "function",\n'
        f'      "function": {{\n'
        f'        "name": "OP_perceive",\n'
        f'        "arguments": {{\n'
        f'          "instance_id": 1,\n'
        f'          "entity": {{\n'
        f'            "name": "{label}",\n'
        f'            "kind": "object"\n'
        f'          }},\n'
        f'          "hint2d": {{\n'
        f'            "type": "tag",\n'
        f'            "value": "4",\n'
        f'            "frame_index": "<candidate_image_1>"\n'
        f'          }}\n'
        f'        }}\n'
        f'      }}\n'
        f'    }}\n'
        f'  ]\n'
        f'}}'
    )

    content: list = []
    for i, p in enumerate(candidate_image_paths, 1):
        content.append({"type": "text", "text": f"<candidate_image_{i}>"})
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": user_text})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()

        block = _extract_json_block(text)
        if block is not None:
            try:
                obj = json.loads(block)
                reason = obj.get("reason", "")
                for call in obj.get("tool_calls", []):
                    args = call["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    hint2d = args.get("hint2d", {})
                    if hint2d.get("type") == "tag":
                        return {
                            "keep_instance_id": int(args.get("instance_id", hint2d.get("value", 0))),
                            "reason": reason,
                        }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        return None
    except Exception as e:
        print(f"  [Dedup] VLM error for '{label}': {e}")
        return None
