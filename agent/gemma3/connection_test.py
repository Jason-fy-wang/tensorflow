import requests
from dotenv import load_dotenv
import json
import os

load_dotenv()

HOST = os.getenv("OLLAMA_HOST")
MODEL = os.getenv("OLLAMA_MODEL")
url = f"{HOST}/api/generate"
payload = {
    "model": MODEL,
    "prompt": "why is the sky blue?",
    "stream": False,
}
try:
    print(f"requesting {url}, model: {MODEL}")
    response = requests.post(url=url, data=json.dumps(payload), timeout=300)
    response.raise_for_status()
    resp_text = json.loads(response.text)
    print(f"response text: {resp_text}")
except requests.Timeout as e:
    print(f"time out exception: {e}")
except Exception as e:
    print(f"other exception: {e}")






