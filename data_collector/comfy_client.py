import json
import urllib.error
import urllib.parse
import urllib.request
import uuid

import requests
import websocket


class ComfyClient:
    def __init__(self, server_address):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def connect(self):
        """Connect to the WebSocket server."""
        self.ws = websocket.WebSocket()
        try:
            self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
            return True
        except Exception as e:
            print(f"Failed to connect to ComfyUI at {self.server_address}: {e}")
            return False

    def close(self):
        """Close the WebSocket connection."""
        if self.ws:
            self.ws.close()

    def upload_image(self, image_path):
        """Upload an image to ComfyUI input directory."""
        try:
            with open(image_path, "rb") as img_file:
                files = {"image": img_file}
                response = requests.post(f"http://{self.server_address}/upload/image", files=files)
                response.raise_for_status()
                return response.json()["name"]
        except Exception as e:
            print(f"Error uploading image {image_path}: {e}")
            return None

    def queue_prompt(self, workflow):
        """Queue a workflow for execution."""
        p = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        try:
            req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else str(e)
            print(f"Error queuing prompt (HTTP {e.code}): {body[:500]}")
            return None
        except Exception as e:
            print(f"Error queuing prompt: {e}")
            return None

    def wait_for_completion(self, prompt_id):
        """Wait for the specific prompt_id to complete via WebSocket."""
        if not self.ws:
            return False

        while True:
            try:
                out = self.ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message["type"] == "executing":
                        data = message["data"]
                        if data["node"] is None and data["prompt_id"] == prompt_id:
                            return True
            except Exception as e:
                print(f"WebSocket error: {e}")
                return False

    def get_history(self, prompt_id):
        """Retrieve execution history for a prompt."""
        try:
            with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
                return json.loads(response.read())
        except Exception as e:
            print(f"Error getting history: {e}")
            return None

    def get_image(self, filename, subfolder, folder_type):
        """Download an image from the server."""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        try:
            with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
                return response.read()
        except Exception as e:
            print(f"Error getting image {filename}: {e}")
            return None
