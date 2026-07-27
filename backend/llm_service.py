import os
import time
from pathlib import Path
from typing import Any

import httpx
import ollama
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

LLM_PROVIDER_OLLAMA = "ollama"
LLM_PROVIDER_DEEPSEEK = "deepseek"
SUPPORTED_LLM_PROVIDERS = {LLM_PROVIDER_OLLAMA, LLM_PROVIDER_DEEPSEEK}

DEEPSEEK_MAX_ATTEMPTS = 3
DEEPSEEK_RETRY_DELAYS_SECONDS = (2, 5)
DEEPSEEK_TIMEOUT_SECONDS = 120


def reload_llm_env() -> None:
    load_dotenv(ENV_PATH, override=True)


def get_ollama_model() -> str:
    reload_llm_env()
    return os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b-q4_K_M")


def get_deepseek_model() -> str:
    reload_llm_env()
    return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def get_deepseek_base_url() -> str:
    reload_llm_env()
    return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")


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
        "paid": LLM_PROVIDER_DEEPSEEK,
        "cloud": LLM_PROVIDER_DEEPSEEK,
        "deepseek_v4_flash": LLM_PROVIDER_DEEPSEEK,
    }
    selected = provider_aliases.get(selected, selected)
    if selected in SUPPORTED_LLM_PROVIDERS:
        return selected
    return LLM_PROVIDER_OLLAMA


def is_ollama_provider(provider: str | None = None) -> bool:
    return normalize_llm_provider(provider) == LLM_PROVIDER_OLLAMA


def is_deepseek_provider(provider: str | None = None) -> bool:
    return normalize_llm_provider(provider) == LLM_PROVIDER_DEEPSEEK


def is_deepseek_configured() -> bool:
    reload_llm_env()
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def get_llm_provider_config() -> dict[str, Any]:
    default_provider = normalize_llm_provider(get_default_llm_provider())
    deepseek_ready = is_deepseek_configured()
    ollama_model = get_ollama_model()
    deepseek_model = get_deepseek_model()
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
                "id": LLM_PROVIDER_DEEPSEEK,
                "label": "線上 AI",
                "available": deepseek_ready,
                "model": deepseek_model,
                "message": "" if deepseek_ready else "線上 AI 尚未設定（缺少 DEEPSEEK_API_KEY）",
            },
        ],
    }


def generate_text(prompt: str, llm_provider: str | None = None) -> str:
    provider = normalize_llm_provider(llm_provider)
    if provider == LLM_PROVIDER_DEEPSEEK:
        return _generate_deepseek_text(prompt)
    return _generate_ollama_text(prompt)


def stop_llm_generation(llm_provider: str | None = None) -> dict[str, Any]:
    provider = normalize_llm_provider(llm_provider)
    if provider == LLM_PROVIDER_DEEPSEEK:
        return {
            "status": "success",
            "message": "DeepSeek API 為雲端服務，沒有需要卸載的本地模型。",
            "provider": provider,
            "model": get_deepseek_model(),
        }

    try:
        ollama_model = get_ollama_model()
        ollama.generate(model=ollama_model, prompt="", keep_alive=0)
        return {
            "status": "success",
            "message": "已要求 Ollama 卸載本地 LLM 模型。",
            "provider": provider,
            "model": ollama_model,
        }
    except Exception as error:
        return {
            "status": "error",
            "message": f"卸載 Ollama 模型失敗：{error}",
            "provider": provider,
            "model": get_ollama_model(),
        }


def _generate_ollama_text(prompt: str) -> str:
    response = ollama.chat(
        model=get_ollama_model(),
        messages=[{"role": "user", "content": prompt + "\n/no_think"}],
    )
    return str(response["message"]["content"])


def _generate_deepseek_text(prompt: str) -> str:
    reload_llm_env()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise LLMProviderError("DeepSeek API Key 尚未設定，請在 backend/.env 設定 DEEPSEEK_API_KEY。")

    request_body = {
        "model": get_deepseek_model(),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response_json: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(DEEPSEEK_MAX_ATTEMPTS):
        try:
            with httpx.Client(timeout=DEEPSEEK_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{get_deepseek_base_url()}/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                response.raise_for_status()
                response_json = response.json()
            break
        except Exception as error:
            last_error = error
            should_retry = _is_retryable_deepseek_error(error) and attempt < DEEPSEEK_MAX_ATTEMPTS - 1
            if not should_retry:
                raise LLMProviderError(_format_deepseek_error(error)) from error
            time.sleep(DEEPSEEK_RETRY_DELAYS_SECONDS[attempt])

    if response_json is None:
        raise LLMProviderError(_format_deepseek_error(last_error))

    choices = response_json.get("choices") or []
    if not choices:
        raise LLMProviderError("DeepSeek 沒有回傳可用的回答內容，請稍後再試。")

    message = choices[0].get("message") or {}
    text = str(message.get("content") or "").strip()
    if not text:
        raise LLMProviderError("DeepSeek 回傳空白內容，請稍後再試。")
    return text


def _get_error_text(error: Exception | None) -> str:
    if error is None:
        return "未知錯誤"
    if isinstance(error, httpx.HTTPStatusError):
        try:
            error_body = error.response.json()
        except Exception:
            error_body = error.response.text
        return f"HTTP {error.response.status_code}: {error_body}"
    return str(error)


def _is_retryable_deepseek_error(error: Exception | None) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {429, 500, 502, 503, 504}
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True

    error_text = _get_error_text(error).upper()
    retryable_markers = ("429", "RATE_LIMIT", "RESOURCE_EXHAUSTED", "TIMEOUT", "503", "UNAVAILABLE")
    return any(marker in error_text for marker in retryable_markers)


def _format_deepseek_error(error: Exception | None) -> str:
    error_text = _get_error_text(error)
    normalized = error_text.upper()

    if "401" in normalized or "UNAUTHORIZED" in normalized:
        return f"DeepSeek API Key 驗證失敗，請檢查 backend/.env 的 DEEPSEEK_API_KEY。錯誤：{error_text}"

    if "429" in normalized or "RATE_LIMIT" in normalized or "RESOURCE_EXHAUSTED" in normalized:
        return f"DeepSeek API 配額或流量限制，請稍後再試或檢查帳戶額度。錯誤：{error_text}"

    if "TIMEOUT" in normalized or "503" in normalized or "UNAVAILABLE" in normalized:
        return f"DeepSeek API 暫時無法回應，請稍後再試。錯誤：{error_text}"

    return f"DeepSeek API 呼叫失敗：{error_text}"
