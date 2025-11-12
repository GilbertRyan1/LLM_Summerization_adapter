# adapter.py
import time
from typing import List, Dict, Any
from google import genai
from google.genai import types

class GeminiAdapter:
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.4) -> Dict[str, Any]:
        if not messages:
            return {"text": "no messages", "in_tokens": 0, "out_tokens": 0, "latency": 0.0}

        user_text = messages[-1].get("content", "") or ""
        t0 = time.time()
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[user_text],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=500,
                ),
            )
        except Exception as e:
            return {"text": f"error: {e}", "in_tokens": 0, "out_tokens": 0, "latency": time.time() - t0}

        if not getattr(resp, "text", None):
            return {"text": "blocked or empty", "in_tokens": 0, "out_tokens": 0, "latency": time.time() - t0}

        usage = getattr(resp, "usage_metadata", None)
        in_t = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_t = getattr(usage, "candidates_token_count", 0) if usage else 0

        return {
            "text": resp.text.strip(),
            "in_tokens": in_t,
            "out_tokens": out_t,
            "latency": time.time() - t0,
        }
