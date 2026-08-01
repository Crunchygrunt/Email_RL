import os

try:
    from dotenv import load_dotenv
    found = load_dotenv()
    print(f"load_dotenv() found a .env file: {found}")
except ImportError:
    print("python-dotenv not installed -- .env file (if any) will NOT be loaded, "
          "same as it wouldn't be for inference.py either in that case.")

print()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
# MODEL_NAME   = os.getenv("MODEL_NAME",   "openai/gpt-oss-120b")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
# MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")

print(f"API_BASE_URL = {API_BASE_URL}")
print(f"MODEL_NAME   = {MODEL_NAME}")
if HF_TOKEN:
    print(f"HF_TOKEN     = {HF_TOKEN[:4]}...{HF_TOKEN[-4:] if len(HF_TOKEN) > 8 else ''} (len={len(HF_TOKEN)})")
else:
    print("HF_TOKEN     = NOT SET (this would have crashed inference.py's import with "
          "'ValueError: HF_TOKEN environment variable is required')")
print()

if not HF_TOKEN:
    print("Stopping here -- no token to test with.")
else:
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=10,
        )
        print("SUCCESS. Raw response:")
        print(completion.choices[0].message.content)
    except Exception as exc:
        print("FAILED. This is the real exception _call_llm was swallowing:")
        print(f"{type(exc).__name__}: {exc}")
