import os
import time
from pathlib import Path
from typing import Any

import ollama
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

LLM_PROVIDER_OLLAMA = "ollama"
LLM_PROVIDER_GEMINI = "gemini"
SUPPORTED_LLM_PROVIDERS = {LLM_PROVIDER_OLLAMA, LLM_PROVIDER_GEMINI}
GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_DELAYS_SECONDS = (2, 5)


def reload_llm_env() -> None:
    load_dotenv(ENV_PATH, override=True)


def get_ollama_model() -> str:
    reload_llm_env()
    return os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b-q4_K_M")


def get_gemini_model() -> str:
    reload_llm_env()
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_default_llm_provider() -> str:
    reload_llm_env()
    return os.getenv("VOICERAG_DEFAULT_LLM_PROVIDER", LLM_PROVIDER_OLLAMA)


class LLMProviderError(RuntimeError):
    pass


def normalize_llm_provider(provider: str | None = None) -> str:
    selected = (provider or get_default_llm_provider() or LLM_PROVIDER_OLLAMA).strip().lower()
    provider_aliases = {
        "local": LLM_PROVIDER_OLLAMA,
        "local_ai": LLM_PROVIDER_OLLAMA,
        "google": LLM_PROVIDER_GEMINI,
        "google_gemini": LLM_PROVIDER_GEMINI,
    }
    selected = provider_aliases.get(selected, selected)
    if selected in SUPPORTED_LLM_PROVIDERS:
        return selected
    return LLM_PROVIDER_OLLAMA


def is_ollama_provider(provider: str | None = None) -> bool:
    return normalize_llm_provider(provider) == LLM_PROVIDER_OLLAMA


def is_gemini_provider(provider: str | None = None) -> bool:
    return normalize_llm_provider(provider) == LLM_PROVIDER_GEMINI


def is_gemini_configured() -> bool:
    reload_llm_env()
    return bool(os.getenv("GEMINI_API_KEY"))


def get_llm_provider_config() -> dict[str, Any]:
    default_provider = normalize_llm_provider(get_default_llm_provider())
    gemini_ready = is_gemini_configured()
    ollama_model = get_ollama_model()
    gemini_model = get_gemini_model()
    return {
        "status": "success",
        "default_provider": default_provider,
        "providers": [
            {
                "id": LLM_PROVIDER_OLLAMA,
                "label": "本地 AI",
                "available": True,
                "model": ollama_model,
                "message": "",
            },
            {
                "id": LLM_PROVIDER_GEMINI,
                "label": "Gemini",
                "available": gemini_ready,
                "model": gemini_model,
                "message": "" if gemini_ready else "Gemini API Key 尚未設定",
            },
        ],
    }


def generate_text(prompt: str, llm_provider: str | None = None) -> str:
    provider = normalize_llm_provider(llm_provider)
    if provider == LLM_PROVIDER_GEMINI:
        return _generate_gemini_text(prompt)
    return _generate_ollama_text(prompt)


def stop_llm_generation(llm_provider: str | None = None) -> dict[str, Any]:
    provider = normalize_llm_provider(llm_provider)
    if provider == LLM_PROVIDER_GEMINI:
        return {
            "status": "success",
            "message": "Gemini API 由遠端處理，前端已停止等待回覆。",
            "provider": provider,
            "model": get_gemini_model(),
        }

    try:
        ollama_model = get_ollama_model()
        ollama.generate(model=ollama_model, prompt="", keep_alive=0)
        return {
            "status": "success",
            "message": "已送出停止 Ollama 模型的請求。",
            "provider": provider,
            "model": ollama_model,
        }
    except Exception as error:
        return {
            "status": "error",
            "message": f"停止 Ollama 模型失敗：{error}",
            "provider": provider,
            "model": get_ollama_model(),
        }


def _generate_ollama_text(prompt: str) -> str:
    response = ollama.chat(
        model=get_ollama_model(),
        messages=[{"role": "user", "content": prompt + "\n/no_think"}],
    )
    return str(response["message"]["content"])


def _generate_gemini_text(prompt: str) -> str:
    reload_llm_env()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMProviderError("Gemini API Key 尚未設定，請在 backend/.env 設定 GEMINI_API_KEY。")

    try:
        from google import genai
    except ImportError as error:
        raise LLMProviderError("尚未安裝 google-genai，請執行 pip install -r requirements.txt。") from error

    client = genai.Client(api_key=api_key)
    response = None
    last_error: Exception | None = None
    for attempt in range(GEMINI_MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(model=get_gemini_model(), contents=prompt)
            break
        except Exception as error:
            last_error = error
            should_retry = _is_retryable_gemini_error(error) and attempt < GEMINI_MAX_ATTEMPTS - 1
            if not should_retry:
                raise LLMProviderError(_format_gemini_error(error)) from error
            time.sleep(GEMINI_RETRY_DELAYS_SECONDS[attempt])

    if response is None:
        raise LLMProviderError(_format_gemini_error(last_error))

    text = str(getattr(response, "text", "") or "").strip()
    if not text:
        raise LLMProviderError("Gemini 沒有回傳文字內容，請稍後再試。")
    return text


def _get_error_text(error: Exception | None) -> str:
    if error is None:
        return "未知錯誤"
    return str(error)


def _is_retryable_gemini_error(error: Exception | None) -> bool:
    error_text = _get_error_text(error).upper()
    retryable_markers = (
        "503",
        "UNAVAILABLE",
        "HIGH DEMAND",
        "RESOURCE_EXHAUSTED",
        "429",
        "RATE_LIMIT",
    )
    return any(marker in error_text for marker in retryable_markers)


def _format_gemini_error(error: Exception | None) -> str:
    error_text = _get_error_text(error)
    normalized = error_text.upper()

    if "503" in normalized or "UNAVAILABLE" in normalized or "HIGH DEMAND" in normalized:
        return (
            "Gemini 目前服務忙碌，Google 暫時無法處理這次請求。"
            "請稍後再試，或先切回「本地 AI」。"
            f"原始錯誤：{error_text}"
        )

    if "429" in normalized or "RESOURCE_EXHAUSTED" in normalized or "RATE_LIMIT" in normalized:
        return (
            "Gemini 免費額度或請求頻率可能已達上限。"
            "請稍後再試、降低使用頻率，或先切回「本地 AI」。"
            f"原始錯誤：{error_text}"
        )

    return f"Gemini 呼叫失敗：{error_text}"
