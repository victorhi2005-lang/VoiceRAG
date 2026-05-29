import gc
import os
import threading
from typing import Any

from sentence_transformers import CrossEncoder, SentenceTransformer
from faster_whisper import BatchedInferencePipeline, WhisperModel
import torch


EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
WHISPER_MODEL_NAME = "large-v3-turbo"

EMBEDDING_DEVICE = os.getenv("VOICERAG_EMBEDDING_DEVICE", "cuda")
RERANKER_DEVICE = os.getenv("VOICERAG_RERANKER_DEVICE", "cuda")
WHISPER_DEVICE = os.getenv("VOICERAG_WHISPER_DEVICE", "cuda")


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


EMBEDDING_BATCH_SIZE = _int_env("VOICERAG_EMBEDDING_BATCH_SIZE", 8)
RERANKER_BATCH_SIZE = _int_env("VOICERAG_RERANKER_BATCH_SIZE", 8)
WHISPER_BATCH_SIZE = _int_env("VOICERAG_WHISPER_BATCH_SIZE", 16)
CLOUD_EMBEDDING_BATCH_SIZE = _int_env("VOICERAG_CLOUD_EMBEDDING_BATCH_SIZE", 32)
CLOUD_RERANKER_BATCH_SIZE = _int_env("VOICERAG_CLOUD_RERANKER_BATCH_SIZE", 16)
CLOUD_WHISPER_BATCH_SIZE = _int_env("VOICERAG_CLOUD_WHISPER_BATCH_SIZE", 32)

WHISPER_COMPUTE_TYPE = os.getenv(
    "VOICERAG_WHISPER_COMPUTE_TYPE",
    "float16" if WHISPER_DEVICE.startswith("cuda") else "int8",
)

_model_lock = threading.RLock()
_embeddings_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_base_whisper_model: WhisperModel | None = None
_whisper_model: BatchedInferencePipeline | None = None


def _torch_model_kwargs(device: str) -> dict[str, Any]:
    if device.startswith("cuda"):
        return {"torch_dtype": torch.float16}
    return {}


def clear_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def get_embeddings_model() -> SentenceTransformer:
    global _embeddings_model

    with _model_lock:
        if _embeddings_model is None:
            print(
                f"Loading embedding model ({EMBEDDING_MODEL_NAME}) on {EMBEDDING_DEVICE}...",
                flush=True,
            )
            _embeddings_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device=EMBEDDING_DEVICE,
                model_kwargs=_torch_model_kwargs(EMBEDDING_DEVICE),
            )
        return _embeddings_model


def get_reranker() -> CrossEncoder:
    global _reranker

    with _model_lock:
        if _reranker is None:
            print(
                f"Loading reranker model ({RERANKER_MODEL_NAME}) on {RERANKER_DEVICE}...",
                flush=True,
            )
            _reranker = CrossEncoder(
                RERANKER_MODEL_NAME,
                device=RERANKER_DEVICE,
                model_kwargs=_torch_model_kwargs(RERANKER_DEVICE),
            )
        return _reranker


def _is_cloud_llm_provider(llm_provider: str | None = None) -> bool:
    return (llm_provider or "").strip().lower() in {"deepseek"}


def get_embedding_batch_size(llm_provider: str | None = None) -> int:
    if _is_cloud_llm_provider(llm_provider):
        return CLOUD_EMBEDDING_BATCH_SIZE
    return EMBEDDING_BATCH_SIZE


def get_reranker_batch_size(llm_provider: str | None = None) -> int:
    if _is_cloud_llm_provider(llm_provider):
        return CLOUD_RERANKER_BATCH_SIZE
    return RERANKER_BATCH_SIZE


def get_whisper_batch_size(llm_provider: str | None = None) -> int:
    if _is_cloud_llm_provider(llm_provider):
        return CLOUD_WHISPER_BATCH_SIZE
    return WHISPER_BATCH_SIZE


def preload_retrieval_models() -> None:
    get_embeddings_model()
    get_reranker()
    print("[OK] Embedding and reranker models are ready.", flush=True)


def preload_models_for_provider(llm_provider: str | None = None) -> None:
    preload_retrieval_models()
    if _is_cloud_llm_provider(llm_provider):
        get_whisper_model()
        print("[OK] Cloud LLM mode preload completed, including Whisper.", flush=True)


def get_whisper_model() -> BatchedInferencePipeline:
    global _base_whisper_model, _whisper_model

    with _model_lock:
        if _whisper_model is None:
            print(
                f"Loading Whisper model ({WHISPER_MODEL_NAME}) on {WHISPER_DEVICE}...",
                flush=True,
            )
            _base_whisper_model = WhisperModel(
                WHISPER_MODEL_NAME,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            _whisper_model = BatchedInferencePipeline(model=_base_whisper_model)
        return _whisper_model


def unload_whisper_model() -> None:
    global _base_whisper_model, _whisper_model

    with _model_lock:
        if _whisper_model is None and _base_whisper_model is None:
            return

        _whisper_model = None
        _base_whisper_model = None
    clear_cuda_memory()
