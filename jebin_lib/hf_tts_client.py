import os
import requests
import time
from custom_logger import logger_config

class HFTTSClient:
    def __init__(self):
        self.voice = os.environ.get("VOICE_NAME", "4")
        self.speed = os.environ.get("SPEED", "1.0")

    def generate_audio_segment(self, content: str, output_path: str) -> str:
        logger_config.info("Using remote HF TTS API")
        try:
            url = "https://jebin2-tts.hf.space/api/generate"
            payload = {
                "text": content,
                "voice": str(self.voice),
                "speed": float(self.speed),
                "hide_from_ui": 1
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            task_id = response.json().get("id")
            
            iteration = 0
            # Poll for completion
            while True:
                status_url = f"https://jebin2-tts.hf.space/api/files/{task_id}"
                files_response = requests.get(status_url)
                files_response.raise_for_status()
                task_info = files_response.json()
                
                status = task_info.get("status")
                queue_pos = task_info.get("queue_position")
                progress = task_info.get("progress")
                
                if status == 'not_started':
                    logger_config.debug(f"TTS API Status: Queued (Position {queue_pos}) - {iteration}", overwrite=True)
                elif status == 'processing':
                    logger_config.debug(f"TTS API Status: {progress} ({progress}%) - {iteration}", overwrite=True)

                if status == "completed":
                    download_url = f"https://jebin2-tts.hf.space/api/download/{task_id}"
                    audio_response = requests.get(download_url)
                    audio_response.raise_for_status()
                    
                    with open(output_path, "wb") as f:
                        f.write(audio_response.content)
                    break
                elif task_info.get("status") == "failed":
                    logger_config.error(f"HF API TTS Failed: {task_info.get('error')}")
                    return None
                    
                time.sleep(1)
                iteration += 1
            
            return output_path
        except Exception as e:
            logger_config.error(f"HF API TTS failed: {e}. Falling back to local process...")
            return None
