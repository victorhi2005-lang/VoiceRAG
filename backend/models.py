import gc
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=False)


def _is_enabled_env(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


FORCE_LOCAL_MODEL_FILES = _is_enabled_env("VOICERAG_FORCE_LOCAL_MODELS")

if FORCE_LOCAL_MODEL_FILES:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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


def _model_loading_mode() -> str:
    if FORCE_LOCAL_MODEL_FILES:
        return "local cache only"
    return "local cache or online download"


def _local_model_error(model_name: str, error: Exception) -> RuntimeError:
    return RuntimeError(
        f"Unable to load model {model_name} from local files. "
        "VOICERAG_FORCE_LOCAL_MODELS=1 forces local cache loading and blocks "
        "Hugging Face downloads. Download the model once with network access, "
        f"or check that the Hugging Face cache still exists. Original error: {error}"
    )


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
                f"Loading embedding model ({EMBEDDING_MODEL_NAME}) on {EMBEDDING_DEVICE} "
                f"({_model_loading_mode()})...",
                flush=True,
            )
            try:
                _embeddings_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME,
                    device=EMBEDDING_DEVICE,
                    model_kwargs=_torch_model_kwargs(EMBEDDING_DEVICE),
                    local_files_only=FORCE_LOCAL_MODEL_FILES,
                )
            except Exception as error:
                raise _local_model_error(EMBEDDING_MODEL_NAME, error) from error
        return _embeddings_model


def get_reranker() -> CrossEncoder:
    global _reranker

    with _model_lock:
        if _reranker is None:
            print(
                f"Loading reranker model ({RERANKER_MODEL_NAME}) on {RERANKER_DEVICE} "
                f"({_model_loading_mode()})...",
                flush=True,
            )
            try:
                _reranker = CrossEncoder(
                    RERANKER_MODEL_NAME,
                    device=RERANKER_DEVICE,
                    model_kwargs=_torch_model_kwargs(RERANKER_DEVICE),
                    local_files_only=FORCE_LOCAL_MODEL_FILES,
                )
            except Exception as error:
                raise _local_model_error(RERANKER_MODEL_NAME, error) from error
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
                f"Loading Whisper model ({WHISPER_MODEL_NAME}) on {WHISPER_DEVICE} "
                f"({_model_loading_mode()})...",
                flush=True,
            )
            try:
                _base_whisper_model = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                    local_files_only=FORCE_LOCAL_MODEL_FILES,
                )
            except Exception as error:
                raise _local_model_error(WHISPER_MODEL_NAME, error) from error
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
