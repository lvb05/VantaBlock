import requests

class VantaBlockClient:
    def __init__(self, endpoint="http://localhost:8001/predict", timeout=10):
        self.endpoint = endpoint
        self.timeout = timeout

    def predict(self, events):
        response = requests.post(
            self.endpoint,
            json={"events": events},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()