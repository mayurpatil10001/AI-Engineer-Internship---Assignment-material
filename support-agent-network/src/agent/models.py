# Defensive backend guard — must come before ANY transformers/torch import.
# Prevents TensorFlow collision if a reviewer has TF installed for an unrelated project.
# (This is exactly the tf_keras/Keras-3 traceback that appeared in test-machine logs.)
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

"""
Centralised local model loading with caching and load-time timing.

Design principles:
  1. OFFLINE FIRST: sets HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE env vars at import time.
     On first run these must be 0 to allow download; after download, flip to 1.
  2. LOAD ONCE: both the embedding model and the generator are loaded exactly once
     per process and cached in module-level singletons. Subsequent calls return
     the cached instance without re-loading.
  3. CPU SAFE: no unconditional CUDA assumptions. Uses float32 on CPU; int8 on CUDA.
  4. TIMING: load_time_seconds is recorded and surfaced in CLI output + logs.

Models used (exact names and revisions):
  - Embeddings : sentence-transformers/all-MiniLM-L6-v2  (revision: main)
  - Generation : Qwen/Qwen2.5-0.5B-Instruct              (revision: main)

See docs/design.md §7 for hardware-aware justification of these choices.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("orbitdesk.agent")

# ── Offline enforcement ───────────────────────────────────────────────────────
# Set to "1" in .env (via TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE) after first download.
# We read those env vars but do NOT override them here — let the user control the flag.
# The offline_check.py script sets them programmatically before running tests.
_HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1"
_TRANSFORMERS_OFFLINE = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"

if _HF_OFFLINE or _TRANSFORMERS_OFFLINE:
    # Propagate to child processes / any lazy imports that check at runtime
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.debug("Offline mode: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")


# ── Model identifiers (single source of truth) ────────────────────────────────

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "main"

GENERATOR_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATOR_MODEL_REVISION = "main"

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))


# ── Load-time records ─────────────────────────────────────────────────────────

@dataclass
class ModelTiming:
    """Records how long each model took to load and per-call inference times."""
    embedding_load_s: float = 0.0
    generator_load_s: float = 0.0
    generation_calls: list[float] = field(default_factory=list)

    @property
    def avg_generation_s(self) -> float:
        if not self.generation_calls:
            return 0.0
        return sum(self.generation_calls) / len(self.generation_calls)

    def report(self) -> dict:
        return {
            "embedding_model": EMBEDDING_MODEL_ID,
            "embedding_load_s": round(self.embedding_load_s, 2),
            "generator_model": GENERATOR_MODEL_ID,
            "generator_load_s": round(self.generator_load_s, 2),
            "generation_call_count": len(self.generation_calls),
            "avg_generation_s": round(self.avg_generation_s, 2),
            "total_generation_s": round(sum(self.generation_calls), 2),
        }


# Module-level singletons
_embedding_model = None
_tokenizer = None
_generator_model = None
_timing = ModelTiming()


# ── Embedding model ───────────────────────────────────────────────────────────

def load_embedding_model():
    """
    Load and cache the sentence-transformers embedding model.

    Returns:
        sentence_transformers.SentenceTransformer instance.

    Load time is recorded in _timing.embedding_load_s and logged.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer  # type: ignore

    logger.info(
        "Loading embedding model",
        extra={"model_id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_MODEL_REVISION},
    )
    t0 = time.perf_counter()
    _embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_ID,
        revision=EMBEDDING_MODEL_REVISION,
    )
    elapsed = time.perf_counter() - t0
    _timing.embedding_load_s = elapsed
    logger.info(
        "Embedding model loaded",
        extra={"model_id": EMBEDDING_MODEL_ID, "load_s": round(elapsed, 2)},
    )
    return _embedding_model


def embed(texts: list[str]) -> "list[list[float]]":
    """
    Embed a list of texts using the cached embedding model.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (numpy arrays converted to lists for JSON-serialisability).
    """
    model = load_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings  # numpy ndarray; callers handle as ndarray


# ── Generator model ───────────────────────────────────────────────────────────

def load_generator():
    """
    Load and cache the local causal LM for response generation.

    Device strategy:
      - CUDA available + bitsandbytes installed → load_in_8bit=True on GPU
      - Otherwise → float32 on CPU (safe for all hardware)

    Returns:
        Tuple of (tokenizer, model).
    """
    global _tokenizer, _generator_model

    if _tokenizer is not None and _generator_model is not None:
        return _tokenizer, _generator_model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    use_cuda = torch.cuda.is_available()
    load_kwargs: dict = {
        "pretrained_model_name_or_path": GENERATOR_MODEL_ID,
        "revision": GENERATOR_MODEL_REVISION,
    }

    if use_cuda:
        try:
            import bitsandbytes  # noqa: F401  # type: ignore
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
            logger.info("Generator: using 8-bit quantisation on CUDA")
        except ImportError:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
            logger.info("Generator: bitsandbytes not available; using float16 on CUDA")
    else:
        load_kwargs["torch_dtype"] = torch.float32
        logger.info("Generator: CUDA not available; using float32 on CPU")

    logger.info(
        "Loading generator model",
        extra={"model_id": GENERATOR_MODEL_ID, "revision": GENERATOR_MODEL_REVISION},
    )
    t0 = time.perf_counter()

    _tokenizer = AutoTokenizer.from_pretrained(
        GENERATOR_MODEL_ID,
        revision=GENERATOR_MODEL_REVISION,
    )
    _generator_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    if not use_cuda:
        # Explicitly move to CPU for clarity
        _generator_model = _generator_model.to("cpu")

    elapsed = time.perf_counter() - t0
    _timing.generator_load_s = elapsed
    logger.info(
        "Generator model loaded",
        extra={
            "model_id": GENERATOR_MODEL_ID,
            "load_s": round(elapsed, 2),
            "device": "cuda" if use_cuda else "cpu",
        },
    )
    return _tokenizer, _generator_model


def generate(prompt: str) -> tuple[str, float]:
    """
    Generate a response from the local causal LM.

    Args:
        prompt: The complete prompt string (already formatted by the generation node).

    Returns:
        Tuple of (generated_text: str, elapsed_seconds: float).
        The generated_text has the prompt prefix stripped.

    Per-call timing is appended to _timing.generation_calls.
    """
    import torch

    tokenizer, model = load_generator()

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # deterministic greedy decoding
            temperature=1.0,          # override Qwen generation_config default (0.7)
            top_p=1.0,                # override Qwen generation_config default (0.8)
            top_k=0,                  # override Qwen generation_config default (20)
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    _timing.generation_calls.append(elapsed)

    # Strip the prompt tokens from the output
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    logger.info(
        "Generation complete",
        extra={
            "elapsed_s": round(elapsed, 2),
            "output_tokens": len(new_ids),
        },
    )
    return generated_text, elapsed


# ── Timing accessor ───────────────────────────────────────────────────────────

def get_timing() -> ModelTiming:
    """Return the module-level timing record (mutable singleton)."""
    return _timing


def get_timing_report() -> dict:
    """Return a JSON-serialisable timing report for CLI output and logs."""
    return _timing.report()
