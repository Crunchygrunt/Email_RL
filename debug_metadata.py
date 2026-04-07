"""Check what metadata the server actually returns."""
import httpx
import json

SERVER = "https://crunchygrunt-algorithm-architects-email-rl.hf.space"

print("=== Calling /reset ===")
resp = httpx.post(f"{SERVER}/reset")
data = resp.json()

obs = data.get("observation", {})
metadata = obs.get("metadata", {})

print(f"\nEmail Subject: {obs.get('email_subject', 'MISSING')[:60]}")
print(f"Email Sender:  {obs.get('email_sender', 'MISSING')}")
print(f"\nFull metadata keys: {list(metadata.keys())}")
print(f"\nMetadata contents:")
for k, v in metadata.items():
    print(f"  {k}: {v}")

# Check for ground truth
print(f"\n=== Ground Truth Check ===")
print(f"true_priority:        {metadata.get('true_priority', 'MISSING')}")
print(f"true_category:        {metadata.get('true_category', 'MISSING')}")
print(f"true_route:           {metadata.get('true_route', 'MISSING')}")
print(f"is_business_critical: {metadata.get('is_business_critical', 'MISSING')}")

if not metadata.get("true_priority"):
    print("\n*** PROBLEM: true_priority is MISSING from metadata ***")
    print("*** The server has NOT been updated with the fix ***")
    print("*** Make sure you pushed the updated Email_RL_environment.py ***")
else:
    print("\n*** OK: Ground truth is present in metadata ***")