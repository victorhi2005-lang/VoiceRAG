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
                f"正在載入 Embedding 向量模型 ({EMBEDDING_MODEL_NAME}) 到 {EMBEDDING_DEVICE}，請稍候...",
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
                f"正在載入 Reranker 精排模型 ({RERANKER_MODEL_NAME}) 到 {RERANKER_DEVICE}，請稍候...",
                flush=True,
            )
            _reranker = CrossEncoder(
                RERANKER_MODEL_NAME,
                device=RERANKER_DEVICE,
                model_kwargs=_torch_model_kwargs(RERANKER_DEVICE),
            )
        return _reranker


def get_whisper_model() -> BatchedInferencePipeline:
    global _base_whisper_model, _whisper_model

    with _model_lock:
        if _whisper_model is None:
            print(
                f"正在載入本地端 Whisper 模型 ({WHISPER_MODEL_NAME}) 到 {WHISPER_DEVICE}，請稍候...",
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
