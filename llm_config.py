"""Unified LLM client factory supporting both cloud API and local vLLM deployment.

Usage:
    from llm_config import get_llm_client, get_llm_config

    # Get a configured OpenAI-compatible client
    client = get_llm_client()  # auto-detects mode from env

    # Or explicitly specify mode
    client = get_llm_client(mode="local")   # local vLLM
    client = get_llm_client(mode="api")     # cloud API (DashScope / OpenAI)

    # Check current config
    config = get_llm_config()
    print(config.model_name, config.base_url)
"""

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


# ============================================================
# Configuration
# ============================================================


@dataclass
class LLMConfig:
    """Holds all LLM backend configuration."""

    # Mode: "api" (cloud) or "local" (vLLM)
    mode: str = "api"

    # --- Cloud API settings ---
    api_key: str = ""
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # --- Local vLLM settings ---
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_api_key: str = "dummy"  # vLLM doesn't need a real key

    # --- Local vLLM model name ---
    # The single model name your vLLM server is serving (set via VLLM_MODEL_NAME env var).
    # All pipeline calls in local mode will use this name regardless of task type.
    # Example: "Qwen/Qwen2.5-7B-Instruct"
    vllm_model_name: str = "local-model"

    # Default model names per task category (used in API mode; these are real API model IDs)
    default_chat_model: str = "qwen3.5-plus"
    default_code_model: str = "qwen3.5-plus"
    default_vlm_model: str = "qwen-vl-plus"


# Global singleton config instance
_config: Optional[LLMConfig] = None


def _load_env_config() -> LLMConfig:
    """Build LLMConfig from environment variables."""
    cfg = LLMConfig()

    # Mode selection
    cfg.mode = os.environ.get("LLM_MODE", "api").lower().strip()

    # API credentials
    cfg.api_key = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    cfg.api_base_url = os.environ.get(
        "LLM_API_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Local vLLM endpoint
    cfg.vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    cfg.vllm_model_name = os.environ.get("VLLM_MODEL_NAME", "local-model")

    # Override defaults via env
    if os.environ.get("LLM_CHAT_MODEL"):
        cfg.default_chat_model = os.environ["LLM_CHAT_MODEL"]
    if os.environ.get("LLM_CODE_MODEL"):
        cfg.default_code_model = os.environ["LLM_CODE_MODEL"]
    if os.environ.get("LLM_VLM_MODEL"):
        cfg.default_vlm_model = os.environ["LLM_VLM_MODEL"]

    return cfg


def get_llm_config() -> LLMConfig:
    """Get (or lazily initialize) the global LLM config."""
    global _config
    if _config is None:
        _config = _load_env_config()
    return _config


def reset_llm_config(cfg: Optional[LLMConfig] = None):
    """Reset config (useful for testing or re-init)."""
    global _config
    _config = cfg


# ============================================================
# Client Factory
# ============================================================


def resolve_model_name(logical_name: str) -> str:
    """Resolve a logical model name to the actual model name for the current mode.

    In 'api' mode, returns the logical name as-is (it IS the real API model ID).
    In 'local' mode, returns the single vLLM model name configured via VLLM_MODEL_NAME,
    ignoring the logical name entirely — configure VLLM_MODEL_NAME to match whatever
    model your vLLM server is serving.
    """
    cfg = get_llm_config()
    if cfg.mode == "local":
        return cfg.vllm_model_name
    return logical_name


def get_llm_client(
    mode: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """Create an OpenAI-compatible client configured for the desired backend.

    Args:
        mode: Override mode ("api" or "local"). If None, uses env config.
        api_key: Override API key.
        base_url: Override base URL.

    Returns:
        An openai.OpenAI instance pointing to either cloud API or local vLLM.
    """
    cfg = get_llm_config()
    effective_mode = (mode or cfg.mode).lower().strip()

    if effective_mode == "local":
        effective_key = api_key or cfg.vllm_api_key
        effective_url = base_url or cfg.vllm_base_url
    else:
        # api mode (default)
        effective_key = api_key or cfg.api_key
        effective_url = base_url or cfg.api_base_url

    if not effective_key and effective_mode == "api":
        raise ValueError(
            f"No API key provided for mode='api'. "
            f"Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable."
        )

    print(f"[LLM Config] mode={effective_mode}, base_url={effective_url}")

    return OpenAI(api_key=effective_key, base_url=effective_url)


# ============================================================
# Convenience helpers for pipeline nodes
# ============================================================


def make_chat_client(api_key: str = "", override_mode: Optional[str] = None) -> OpenAI:
    """Create a client optimized for chat/completion tasks (perception proposal, pseudocode, etc.)."""
    cfg = get_llm_config()
    mode = override_mode or cfg.mode
    return get_llm_client(mode=mode, api_key=api_key or None)


def make_vlm_client(api_key: str = "", override_mode: Optional[str] = None) -> OpenAI:
    """Create a client optimized for vision-language tasks (point inference, dedup judging).

    Note: For VLM tasks, ensure the served model supports vision input.
    """
    cfg = get_llm_config()
    mode = override_mode or cfg.mode
    return get_llm_client(mode=mode, api_key=api_key or None)


def make_code_client(api_key: str = "", override_mode: Optional[str] = None) -> OpenAI:
    """Create a client optimized for code generation tasks."""
    cfg = get_llm_config()
    mode = override_mode or cfg.mode
    return get_llm_client(mode=mode, api_key=api_key or None)


# ============================================================
# CLI helper
# ============================================================

def print_llm_status():
    """Print current LLM configuration status."""
    cfg = get_llm_config()
    print("=" * 50)
    print("LLM Backend Configuration")
    print("=" * 50)
    print(f"  Mode:       {cfg.mode}")
    print(f"  Chat model: {cfg.default_chat_model} -> {resolve_model_name(cfg.default_chat_model)}")
    print(f"  Code model: {cfg.default_code_model} -> {resolve_model_name(cfg.default_code_model)}")
    print(f"  VLM model:  {cfg.default_vlm_model} -> {resolve_model_name(cfg.default_vlm_model)}")
    if cfg.mode == "api":
        print(f"  Base URL:   {cfg.api_base_url}")
        print(f"  API Key:    {'***' + cfg.api_key[-4:] if len(cfg.api_key) > 4 else '(not set)' if cfg.api_key else '(empty)'}")
    else:
        print(f"  vLLM URL:   {cfg.vllm_base_url}")
    print("=" * 50)


if __name__ == "__main__":
    print_llm_status()
