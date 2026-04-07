"""Print the RAW JSON response from the server."""
import httpx
import json

SERVER = "https://crunchygrunt-algorithm-architects-email-rl.hf.space"

print("=== RAW /reset response ===")
resp = httpx.post(f"{SERVER}/reset")
data = resp.json()
print(json.dumps(data, indent=2)[:3000])

print("\n\n=== ALL TOP-LEVEL KEYS ===")
print(list(data.keys()))

print("\n=== Checking nested structures ===")
if "observation" in data:
    print("Found 'observation' key")
    obs = data["observation"]
    print(f"  observation keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
else:
    print("NO 'observation' key - fields may be at top level")
    # Check if metadata is at top level
    if "metadata" in data:
        print(f"  metadata at top level: {data['metadata']}")
    if "true_priority" in data:
        print(f"  true_priority at top level: {data['true_priority']}")