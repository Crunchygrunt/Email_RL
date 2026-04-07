"""Quick test to check if the LLM API connection works."""
from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

print(f"API_BASE_URL: {base}")
print(f"MODEL_NAME:   {model}")
print(f"API_KEY set:  {bool(key)}")
print(f"API_KEY preview: {key[:10] if key else 'NONE'}...")

print("\nCalling LLM...")
try:
    client = OpenAI(base_url=base, api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one word"}],
        max_tokens=10,
    )
    print(f"SUCCESS: {resp.choices[0].message.content}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")