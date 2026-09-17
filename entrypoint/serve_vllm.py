"""Deploy a local LLM via vLLM on Ray Serve.

This script starts an OpenAI-compatible vLLM inference server as a Ray Serve
deployment, allowing the rest of the CogSpace pipeline to call it exactly like
a cloud API (same openai.OpenAI client, different base_url).

Usage:
    # 1. Start Ray cluster first
    ray start --head

    # 2. Deploy vLLM model (example with Qwen2.5-7B-Instruct)
    python entrypoints/serve_vllm.py \
        --model-path /path/to/Qwen2.5-7B-Instruct \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --num-gpus 1 \
        --port 8100

    # 3. Set environment variable to use local model
    export LLM_MODE=local
    export VLLM_BASE_URL=http://127.0.0.1:8100/v1

    # 4. Run pipeline normally - it will auto-use local vLLM
    python entrypoints/run_graph.py

Environment Variables:
    VLLM_MODEL_PATH       : Path to local model weights (HuggingFace format)
    VLLM_MODEL_NAME       : Model identifier name for API calls
    VLLM_NUM_GPUS         : Number of GPUs per replica (default: 1)
    VLLM_SERVE_PORT       : HTTP port for vLLM endpoint (default: 8100)
    VLLM_DTYPE            : Model dtype, e.g. "bfloat16", "float16", "auto"
    VLLM_MAX_MODEL_LEN    : Max context length (default: 16384)
    VLLM_GPU_MEMORY_UTILIZATION: GPU memory fraction (default: 0.9)
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray import serve
from starlette.responses import JSONResponse as Response


# ============================================================
# vLLM Deployment Definition
# ============================================================


def create_vllm_deployment(
    model_path: str,
    model_name: str,
    num_gpus: int = 1,
    dtype: str = "auto",
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.9,
    enable_auto_tool_call: bool = False,
):
    """Create a Ray Serve deployment that wraps vLLM's AsyncLLMEngine.

    NOTE: This Ray-actor mode requires vLLM to be installed and uses the
    low-level AsyncLLMEngine API.  For most users the subprocess mode
    (``deploy_vllm_subprocess``) is simpler and more stable — it launches
    vLLM's built-in OpenAI-compatible server directly.
    """

    try:
        import vllm  # noqa: F401 — verify vLLM is installed
    except ImportError:
        raise ImportError(
            "vLLM is not installed. Install it with:\n"
            "  pip install vllm\n"
            "For GPU support, also ensure CUDA is available."
        )

    # Build vLLM engine arguments
    engine_args = dict(
        model=model_path,
        tensor_parallel_size=num_gpus,  # TP across GPUs if >1 GPU per replica
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_auto_tool_call=enable_auto_tool_call,
        # Disable unnecessary features for serving
        disable_log_stats=False,
        # Trust remote code (for custom models like Qwen)
        trust_remote_code=True,
    )

    @serve.deployment(
        name="vllm_llm",
        ray_actor_options={
            "num_gpus": num_gpus,
            # Give vLLM enough resources
            "memory": 20 * 1024 * 1024 * 1024,  # 20GB
        },
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 1,
            "target_num_ongoing_requests_per_replica": 10,
        },
    )
    class VLLMDeployment:
        """Ray Serve deployment wrapping vLLM's OpenAI-compatible server."""

        def __init__(self):
            """Initialize the vLLM async engine and OpenAI server wrapper."""
            from vllm import SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            print(f"[VLLM] Initializing model from: {model_path}")
            print(f"[VLLM] Tensor parallel size: {num_gpus}")
            print(f"[VLLM] Dtype: {dtype}, Max model len: {max_model_len}")

            self.engine_args = AsyncEngineArgs(**engine_args)
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

            # Create the OpenAI-compatible server wrapper
            # This handles /v1/chat/completions, /v1/models, etc.
            self.model_config = None
            self.served_model_names = [model_name]

            # We'll use a simple request handler approach
            import asyncio
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Pre-warm the engine by running a dummy request
            print("[VLLM] Engine initialized successfully.")

        async def __call__(self, request):
            """Handle HTTP requests (used when deployed with ingress)."""
            # For direct Python API calls we use the methods below;
            # this is for HTTP access through Ray Serve proxy.
            import json

            # Parse the request
            body = await request.body()
            data = json.loads(body)

            if data.get("model") is None:
                data["model"] = model_name

            # Route to appropriate handler
            path = request.url.path

            if path in ("/v1/chat/completions", "/chat/completions"):
                return await self._handle_chat(data)
            elif path in ("/v1/models", "/models"):
                return await self._handle_models()
            elif path in ("/v1/completions", "/completions"):
                return await self._handle_completion(data)
            else:
                return Response(content=json.dumps({"error": f"Unknown path: {path}"}), status_code=404, media_type="application/json")

        async def _handle_chat(self, data):
            """Handle /v1/chat/completions requests."""
            from vllm import SamplingParams

            messages = data.get("messages", [])
            model = data.get("model", model_name)
            temperature = data.get("temperature", 1.0)
            max_tokens = data.get("max_tokens", 256)
            top_p = data.get("top_p", 1.0)

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            # Convert messages to prompt string
            prompt = self._messages_to_prompt(messages)

            results_generator = self.engine.generate(prompt, sampling_params, request_id="serve-req")
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if final_output is None or len(final_output.outputs) == 0:
                return Response(content=json.dumps({"error": "No output generated"}), status_code=500, media_type="application/json")

            output = final_output.outputs[0]
            prompt_tokens = len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0
            completion_tokens = len(output.token_ids) if output.token_ids else 0
            payload = {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output.text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            }
            return Response(content=json.dumps(payload), media_type="application/json")

        async def _handle_completion(self, data):
            """Handle /v1/completions requests."""
            from vllm import SamplingParams

            prompt = data.get("prompt", "")
            model = data.get("model", model_name)
            temperature = data.get("temperature", 1.0)
            max_tokens = data.get("max_tokens", 256)
            top_p = data.get("top_p", 1.0)

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            results_generator = self.engine.generate(prompt, sampling_params, request_id="serve-req")
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if final_output is None or len(final_output.outputs) == 0:
                return Response(content=json.dumps({"error": "No output generated"}), status_code=500, media_type="application/json")

            output = final_output.outputs[0]
            payload = {
                "id": "cmpl-local",
                "object": "text.completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": output.text,
                    "finish_reason": "stop",
                }],
            }
            return Response(content=json.dumps(payload), media_type="application/json")

        async def _handle_models(self):
            """Handle /v1/models requests."""
            payload = {
                "object": "list",
                "data": [
                    {"id": model_name, "object": "model", "owned_by": "local-vllm"}
                ],
            }
            return Response(content=json.dumps(payload), media_type="application/json")

        def _messages_to_prompt(self, messages):
            """Convert OpenAI chat messages format to a single prompt string.

            For models with chat template support (e.g., Qwen), use tokenizer.
            Otherwise fall back to simple concatenation.
            """
            try:
                tokenizer = self.engine.get_tokenizer()

                # Try apply_chat_template (preferred for chat models)
                if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

            # Fallback: concatenate messages
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle multimodal content (text + images)
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = "\n".join(text_parts)
                parts.append(f"{role}: {content}")

            return "\n".join(parts) + "\nassistant:"

    return VLLMDeployment


# ============================================================
# Alternative: Use vLLM's built-in OpenAI server subprocess
# ============================================================


def deploy_vllm_subprocess(
    model_path: str,
    port: int = 8100,
    num_gpus: int = 1,
    dtype: str = "auto",
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.9,
):
    """Launch vLLM's built-in OpenAI server as a background subprocess.

    This is simpler and more feature-complete than the programmatic approach above.
    The downside is that it runs outside of Ray's actor management.

    Returns the process handle so the caller can manage its lifecycle.
    """
    import subprocess

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(num_gpus),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
    ]

    print(f"[vLLM Subprocess] Launching: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"[vLLM Subprocess] PID={proc.pid}, waiting for server to be ready...")

    # Wait until server responds
    import time
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/models"
    max_wait = 300  # 5 minutes for large models
    start = time.time()

    while time.time() - start < max_wait:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    elapsed = time.time() - start
                    print(f"[vLLM Subprocess] Server ready at {url} ({elapsed:.1f}s)")
                    return proc
        except Exception:
            pass

        # Check if process died
        poll = proc.poll()
        if poll is not None:
            out, _ = proc.communicate(timeout=5)
            print(f"[vLLM Subprocess] Process exited with code {poll}")
            print(f"[vLLM Subprocess] Output:\n{out[-2000:] if out else '(empty)'}")
            raise RuntimeError(f"vLLM server process exited unexpectedly with code {poll}")

        time.sleep(2)

    raise TimeoutError(f"vLLM server did not become ready within {max_wait}s")


# ============================================================
# Main deployment script
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Deploy local LLM via vLLM on Ray Serve")
    parser.add_argument("--model-path", type=str, default=os.environ.get("VLLM_MODEL_PATH", ""),
                        help="Path to local model weights (HuggingFace format)")
    parser.add_argument("--model-name", type=str, default=os.environ.get("VLLM_MODEL_NAME", "local-model"),
                        help="Model identifier name for API calls")
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("VLLM_NUM_GPUS", "1")),
                        help="Number of GPUs per replica")
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_SERVE_PORT", "8100")),
                        help="HTTP port for the vLLM endpoint")
    parser.add_argument("--dtype", type=str, default=os.environ.get("VLLM_DTYPE", "auto"),
                        help="Model dtype (auto, bfloat16, float16, half)")
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384")),
                        help="Maximum context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
                        help="GPU memory utilization fraction (0.0-1.0)")
    parser.add_argument("--mode", type=str, choices=["ray", "subprocess"], default="subprocess",
                        help="Deployment mode: 'ray' (Ray Serve deployment) or 'subprocess' (standalone vLLM server)")

    args = parser.parse_args()

    if not args.model_path:
        parser.error(
            "Model path is required. Set VLLM_MODEL_PATH env var or pass --model-path.\n\n"
            "Example:\n"
            "  python entrypoints/serve_vllm.py --model-path /data/models/Qwen2.5-7B-Instruct "
            "--model-name Qwen/Qwen2.5-7B-Instruct"
        )

    if not os.path.isdir(args.model_path):
        print(f"[ERROR] Model path does not exist: {args.model_path}")
        sys.exit(1)

    print("=" * 60)
    print("Local LLM Deployment (vLLM)")
    print("=" * 60)
    print(f"  Model path:   {args.model_path}")
    print(f"  Model name:   {args.model_name}")
    print(f"  Mode:         {args.mode}")
    print(f"  Port:         {args.port}")
    print(f"  GPUs:         {args.num_gpus}")
    print(f"  Dtype:        {args.dtype}")
    print("=" * 60)

    if args.mode == "ray":
        # Deploy inside Ray Serve
        ray.init(ignore_reinit_error=True)
        deployment = create_vllm_deployment(
            model_path=args.model_path,
            model_name=args.model_name,
            num_gpus=args.num_gpus,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        serve.run(deployment.bind(), name="vllm_app", route_prefix="/vllm")
        print(f"\nvLLM deployed via Ray Serve at http://127.0.0.1:8000/vllm")
        print(f"Set environment: VLLM_BASE_URL=http://127.0.0.1:8000/vllm/v1")
        print(f"Set environment: LLM_MODE=local")
    else:
        # Deploy as standalone subprocess (recommended for stability)
        proc = deploy_vllm_subprocess(
            model_path=args.model_path,
            port=args.port,
            num_gpus=args.num_gpus,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        base_url = f"http://127.0.0.1:{args.port}/v1"
        print(f"\n{'='*60}")
        print(f"vLLM server running at {base_url}")
        print(f"\nTo use this model in your pipeline, set:")
        print(f"  export LLM_MODE=local")
        print(f"  export VLLM_BASE_URL={base_url}")
        print(f"\nTo stop the server, press Ctrl+C or kill PID {proc.pid}")
        print(f"{'='*60}")

        # Keep alive
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("\n[vLLM] Stopping server...")
            proc.terminate()
            proc.wait(timeout=30)
            print("[vLLM] Server stopped.")


if __name__ == "__main__":
    main()
