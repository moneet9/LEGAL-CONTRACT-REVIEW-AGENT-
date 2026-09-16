import json, time
from .config import settings
class GeminiClient:
    def __init__(self):
        self.available=bool(settings.gemini_api_key); self.client=None
        if self.available:
            from google import genai
            self.client=genai.Client(api_key=settings.gemini_api_key)
    def generate_structured(self,prompt,schema):
        if not self.client: return None
        for attempt in range(settings.max_retries+1):
            try:
                response=self.client.models.generate_content(model=settings.gemini_model,contents=prompt,config={"response_mime_type":"application/json","response_schema":schema})
                return json.loads(response.text)
            except Exception:
                if attempt>=settings.max_retries: raise
                time.sleep(2**attempt)
    def embed(self,texts):
        if not self.client: return None
        response=self.client.models.embed_content(model=settings.gemini_embedding_model,contents=texts)
        return [list(item.values) for item in response.embeddings]
