from .client import VantaBlockClient

client = VantaBlockClient()

# Replace with real event data
result = client.predict(events=[])

if result.get("bot_prob", 0) > 0.8:
    print("Blocked")
else:
    print("Allowed")