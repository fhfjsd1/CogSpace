#!/usr/bin/env python3
"""CogSpace Pipeline - Unified Launcher

One script to rule them all:
  1. Start Ray cluster (if not running)
  2. Deploy GPU vision models via Ray Serve (Pi3X / SAM3 / OrientV2)
  3. Optionally deploy local LLM via vLLM
  4. Run the full inference pipeline (LangGraph)

Usage:
    # Full pipeline with cloud API (default)
    python START.py run

    # Deploy only models (no inference)
    python START.py deploy-models

    # Deploy local LLM + models + run pipeline
    python START.py --llm-mode local --vllm-model-path /data/models/Qwen2.5-7B-Instruct run

    # Run inference only (assume Ray + models already deployed)
    python START.py run-only

    # Stop everything
    python START.py stop

Environment Variables:
    LLM_MODE            : "api" (default) or "local"
    DASHSCOPE_API_KEY   : API key for cloud mode
    VLLM_BASE_URL       : URL for local vLLM server (default: http://127.0.0.1:8100/v1)
    VLLM_MODEL_PATH     : Path to local model weights
    VLLM_MODEL_NAME     : Model identifier for vLLM
    VLLM_NUM_GPUS       : GPUs per vLLM replica (default: 1)
"""

import os
import sys
import argparse
import subprocess
import time
import signal
import shutil

try:
    import requests
except ImportError:
    requests = None  # fallback: skip health checks


# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_cmd(cmd: str, check=True, capture=False) -> subprocess.CompletedProcess:
    """Shell out a command."""
    print(f"  $ {cmd}")
    kw = dict(shell=True, text=True)
    if capture:
        kw.update(capture_output=True)
    result = subprocess.run(cmd, **kw)
    if check and result.returncode != 0:
        print(f"  [FAIL] exit code {result.returncode}")
        if result.stdout:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.stderr:
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        sys.exit(result.returncode)
    return result


# ---------------------------------------------------------------------------
# Step 1: Ray Cluster
# ---------------------------------------------------------------------------


def ensure_ray_cluster():
    """Start Ray head node if not already running."""
    import ray

    banner("Step 1/4: Ray Cluster")

    # 1) Try connecting to an existing cluster (with short timeout)
    try:
        info = ray.init(address="auto", ignore_reinit_error=True, _timeout=10)
        print(f"  [OK] Connected to existing cluster at {info.address}")
        return True
    except Exception as e:
        err_msg = str(e)
        print(f"  No usable cluster found ({err_msg}). Cleaning up stale state ...")
        # 2) Stale session detected → wipe it and start fresh
        import subprocess as sp
        sp.run("ray stop --force", shell=True, capture_output=True, timeout=15)

    # 3) Start a fresh head node
    try:
        result = run_cmd("ray start --head --disable-usage-stats", capture=True)
        time.sleep(2)

        ray.init(ignore_reinit_error=True)
        print("  [OK] Ray head node started")
        return True
    except Exception as e:
        print(f"  [WARN] Could not start ray: {e}")
        print("  Falling back to single-node init...")
        ray.init(ignore_reinit_error=True)
        return False


# ---------------------------------------------------------------------------
# Step 2: Deploy Vision Models (Ray Serve)
# ---------------------------------------------------------------------------


def deploy_vision_models():
    """Deploy Pi3X / SAM3 / OrientV2 on Ray Serve."""
    banner("Step 2/4: Vision Models (Ray Serve)")

    from entrypoint.serve_models import deploy_all
    deploy_all()

    # Quick health check
    if requests:
        endpoints = [
            ("Pi3XServer", "http://127.0.0.1:8000/pi3x"),
            ("SAM3Server", "http://127.0.0.1:8000/sam3"),
            ("OrientV2Server", "http://127.0.0.1:8000/orient"),
        ]
        all_ok = True
        for name, url in endpoints:
            try:
                resp = requests.get(url, timeout=5)
                status = f"[OK] {resp.status_code}"
            except Exception as e:
                status = f"[WARN] {e}"
                all_ok = False
            print(f"  {name:20s} → {status}")

        if all_ok:
            print("\n  All vision models ready.")
        else:
            print("\n  Some endpoints not responding — this is normal during cold-start.")
            print("  Models will initialize on first request.")
    else:
        print("  [SKIP] Health check (requests not installed)")


# ---------------------------------------------------------------------------
# Step 3: Local LLM (optional)
# ---------------------------------------------------------------------------

_vllm_process = None


def deploy_local_llm(model_path: str, model_name: str, num_gpus: int, port: int):
    """Launch vLLM server as a background process."""
    global _vllm_process

    banner(f"Step 3/4: Local LLM (vLLM) @ port {port}")

    # Check if vLLM is installed
    try:
        import vllm
        print(f"  vLLM version: {vllm.__version__}")
    except ImportError:
        print("  [ERROR] vLLM not installed. Run: pip install vllm")
        sys.exit(1)

    if not os.path.isdir(model_path):
        print(f"  [ERROR] Model path not found: {model_path}")
        sys.exit(1)

    cmd = (
        f"{sys.executable} -m vllm.entrypoints.openai.api_server "
        f"--model {model_path} "
        f"--port {port} "
        f"--tensor-parallel-size {num_gpus} "
        f"--dtype auto "
        f"--max-model-len 16384 "
        f"--gpu-memory-utilization 0.9 "
        f"--trust-remote-code"
    )
    print(f"  Launching: {cmd}")

    _vllm_process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to be ready
    url = f"http://127.0.0.1:{port}/v1/models"
    max_wait = 300
    start = time.time()
    print(f"  Waiting for vLLM server at {url} ...")
    while time.time() - start < max_wait:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                elapsed = time.time() - start
                print(f"  [OK] vLLM ready in {elapsed:.1f}s")
                return
        except Exception:
            pass

        poll = _vllm_process.poll()
        if poll is not None:
            out = _vllm_process.stdout.read()[-1000:]
            print(f"  [ERROR] vLLM exited with code {poll}\n{out}")
            sys.exit(1)

        elapsed = int(time.time() - start)
        dots = "." * ((elapsed // 2) % 20)
        print(f"\r  Still waiting... {elapsed}s {dots}  ", end="", flush=True)
        time.sleep(2)

    print(f"\n  [TIMEOUT] vLLM did not start in {max_wait}s")
    stop_vllm()
    sys.exit(1)


def stop_vllm():
    """Stop the vLLM subprocess."""
    global _vllm_process
    if _vllm_process and _vllm_process.poll() is None:
        print("  Stopping vLLM...")
        _vllm_process.terminate()
        try:
            _vllm_process.wait(timeout=30)
            print("  [OK] vLLM stopped")
        except subprocess.TimeoutExpired:
            _vllm_process.kill()
            _vllm_process.wait()


# ---------------------------------------------------------------------------
# Step 4: Run Inference
# ---------------------------------------------------------------------------


def run_pipeline():
    """Execute the LangGraph inference pipeline."""
    banner("Step 4/4: Inference Pipeline")

    from entrypoint.run_graph import main
    main()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def action_deploy(args):
    """Deploy models only."""
    ensure_ray_cluster()
    deploy_vision_models()
    if args.llm_mode == "local":
        deploy_local_llm(
            model_path=args.vllm_model_path or os.environ.get("VLLM_MODEL_PATH", ""),
            model_name=args.vllm_model_name or os.environ.get("VLLM_MODEL_NAME", "local-model"),
            num_gpus=int(os.environ.get("VLLM_NUM_GPUS", "1")),
            port=int(os.environ.get("VLLM_SERVE_PORT", "8100")),
        )

    banner("Deployment Complete — keeping services alive...")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        stop_all()


def action_run(args):
    """Full pipeline: Ray → Vision Models → optional LLM → Inference."""
    ensure_ray_cluster()
    deploy_vision_models()

    if args.llm_mode == "local":
        deploy_local_llm(
            model_path=args.vllm_model_path or os.environ.get("VLLM_MODEL_PATH", ""),
            model_name=args.vllm_model_name or os.environ.get("VLLM_MODEL_NAME", "local-model"),
            num_gpus=int(os.environ.get("VLLM_NUM_GPUS", "1")),
            port=int(os.environ.get("VLLM_SERVE_PORT", "8100")),
        )

    run_pipeline()


def action_run_only(args):
    """Run inference assuming Ray + models are already up."""
    run_pipeline()


def action_stop(args):
    """Stop Ray and vLLM."""
    banner("Stopping Services")
    stop_vllm()
    run_cmd("ray stop --force", check=False)
    print("[OK] All services stopped.")


def stop_all():
    """Graceful shutdown."""
    print("\n\nShutting down...")
    stop_vllm()
    run_cmd("ray stop --force", check=False)
    print("Done. Goodbye!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="START.py",
        description="CogSpace unified launcher: Ray + Serve + vLLM + Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline using cloud API
  python START.py run

  # Use local LLM model
  python START.py --llm-mode local --vllm-model-path /models/Qwen2.5-7B-Instruct run

  # Deploy models separately, then run later
  python START.py deploy-models
  python START.py run-only

  # Clean shutdown
  python START.py stop

Environment Variables:
  DASHSCOPE_API_KEY     API key for cloud mode (required when llm-mode=api)
  LLM_MODE             "api" | "local" (default: api)
  VLLM_BASE_URL         vLLM endpoint (default: http://127.0.0.1:8100/v1)
  VLLM_MODEL_PATH       Path to local model weights
  VLLM_MODEL_NAME       Model name served by vLLM
  VLLM_NUM_GPUS         GPUs for vLLM (default: 1)
  VLLM_SERVE_PORT       vLLM HTTP port (default: 8100)
""",
    )

    # LLM config
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument(
        "--llm-mode",
        choices=["api", "local"],
        default=os.environ.get("LLM_MODE", "api"),
        help="LLM backend: 'api' (cloud DashScope) or 'local' (vLLM). Env: LLM_MODE",
    )
    llm_group.add_argument(
        "--vllm-model-path",
        default="",
        help="Path to local model weights (HF format). Env: VLLM_MODEL_PATH",
    )
    llm_group.add_argument(
        "--vllm-model-name",
        default=os.environ.get("VLLM_MODEL_NAME", "local-model"),
        help="Model identifier for vLLM API calls. Env: VLLM_MODEL_NAME",
    )

    # Action
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("run", help="Full pipeline: Ray → Vision Models → LLM → Inference")
    subparsers.add_parser("deploy-models", help="Deploy vision models (+ optional LLM), keep alive")
    subparsers.add_parser("run-only", help="Run inference only (assumes Ray + models up)")
    subparsers.add_parser("stop", help="Stop Ray + vLLM services")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Set env vars from CLI args so downstream code picks them up
    if args.llm_mode:
        os.environ["LLM_MODE"] = args.llm_mode
    if args.vllm_model_path:
        os.environ["VLLM_MODEL_PATH"] = args.vllm_model_path
    if args.vllm_model_name:
        os.environ["VLLM_MODEL_NAME"] = args.vllm_model_name

    # Print summary
    print("=" * 60)
    print("  CogSpace Pipeline - Unified Launcher")
    print("=" * 60)
    print(f"  Action:      {args.action}")
    print(f"  LLM Mode:    {args.llm_mode}")
    if args.llm_mode == "local":
        mp = args.vllm_model_path or os.environ.get("VLLM_MODEL_PATH", "(not set)")
        mn = args.vllm_model_name
        print(f"  vLLM Model:  {mn} @ {mp}")
    else:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        key_display = f"...{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
        print(f"  API Key:     {key_display}")
    print("=" * 60 + "\n")

    # Register signal handler for clean Ctrl+C
    signal.signal(signal.SIGINT, lambda s, f: (_ := stop_all()) or sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: (_ := stop_all()) or sys.exit(0))

    # Dispatch
    dispatch = {
        "deploy-models": action_deploy,
        "run": action_run,
        "run-only": action_run_only,
        "stop": action_stop,
    }
    dispatch[args.action](args)


if __name__ == "__main__":
    main()
