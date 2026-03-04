import json
import os
import requests
import time
from custom_logger import logger_config


class HFTTTClient:
    def __init__(self):
        self.base_url = os.environ.get("TTT_API_URL", "https://jebin2-ttt.hf.space")
        self.system_prompt = os.environ.get("TTT_SYSTEM_PROMPT", "You are a helpful assistant.")

    def generate(self, text: str, system_prompt: str = None) -> str:
        """
        Submit text to the HF TTT API and poll until complete.
        Returns the generated text string, or raises on failure.
        """
        logger_config.info("Using remote HF TTT API")

        payload = {
            "text": text,
            "system_prompt": system_prompt or self.system_prompt,
            "hide_from_ui": 1,
        }
        response = requests.post(f"{self.base_url}/api/submit", json=payload)
        response.raise_for_status()
        task_id = response.json().get("id")

        iteration = 0
        while True:
            status_response = requests.get(f"{self.base_url}/api/tasks/{task_id}")
            status_response.raise_for_status()
            task_info = status_response.json()

            status = task_info.get("status")
            queue_pos = task_info.get("queue_position")
            progress = task_info.get("progress", 0)
            progress_text = task_info.get("progress_text") or "Processing..."

            if status == "not_started":
                logger_config.debug(f"TTT API Status: Queued (Position {queue_pos}) - {iteration}", overwrite=True)
            elif status == "processing":
                logger_config.debug(f"TTT API Status: {progress_text} ({progress}%) - {iteration}", overwrite=True)

            if status == "completed":
                result = json.loads(task_info.get("result", "{}"))
                return result.get("text", "")

            if status == "failed":
                raise RuntimeError(f"HF TTT API failed: {task_info.get('result')}")

            time.sleep(2)
            iteration += 1
