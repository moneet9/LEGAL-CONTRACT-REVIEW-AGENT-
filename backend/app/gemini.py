import json, time
import httpx
from .config import settings
class ModelGenerationError(RuntimeError):
    pass

class GeminiClient:
    def __init__(self):
        self.available=bool(settings.gemini_api_key); self.client=None
        if self.available:
            from google import genai
            self.client=genai.Client(api_key=settings.gemini_api_key)
    def generate_structured(self,prompt,schema,model=None):
        if not self.client: return None
        for attempt in range(settings.max_retries+1):
            try:
                response=self.client.models.generate_content(model=model or settings.gemini_model,contents=prompt,config={"response_mime_type":"application/json","response_schema":schema})
                return json.loads(response.text)
            except Exception:
                if attempt>=settings.max_retries: raise
                time.sleep(2**attempt)

def available_models() -> list[dict[str, str]]:
    models = []
    if settings.gemini_api_key:
        try:
            from google import genai
            client = genai.Client(api_key=settings.gemini_api_key)
            for model in client.models.list():
                name = str(getattr(model, "name", "")).removeprefix("models/")
                actions = list(getattr(model, "supported_actions", []) or [])
                unsupported = ("tts", "image", "transcribe", "lyria", "robotics", "deep-research", "nano-banana")
                if name and not any(marker in name for marker in unsupported) and (not actions or "generateContent" in actions):
                    models.append({"id": name, "name": name, "provider": "Google"})
        except Exception:
            pass
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=1.5)
        response.raise_for_status()
        models.extend({"id": item["name"], "name": item["name"], "provider": "Ollama"} for item in response.json().get("models", []))
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    unique = {item["id"]: item for item in models}
    return list(unique.values())


def default_model(models: list[dict[str, str]]) -> str | None:
    ids = [item["id"] for item in models]
    if settings.gemini_model in ids:
        return settings.gemini_model
    google = next((item["id"] for item in models if item["provider"] == "Google"), None)
    return google or (ids[0] if ids else None)


def generate(prompt: str, schema: dict, model: str | None = None):
    if not model:
        return None
    selected = model
    if selected.startswith(("gemini", "gemma")):
        candidates = [selected]
        try:
            candidates.extend(item["id"] for item in available_models() if item["provider"] == "Google" and item["id"] != selected)
        except Exception:
            pass
        errors=[]
        for candidate in candidates[:4]:
            try:
                return GeminiClient().generate_structured(prompt, schema, candidate)
            except Exception as error:
                errors.append(f"{candidate}: {error}")
                continue
        raise ModelGenerationError("Google model API failed after trying available models. " + " | ".join(errors))
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": selected, "prompt": prompt, "format": "json", "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return json.loads(response.json()["response"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelGenerationError(f"Ollama model API failed for '{selected}': {error}") from error
