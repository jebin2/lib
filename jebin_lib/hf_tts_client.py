import os
import requests
import time
from custom_logger import logger_config

class HFTTSClient:
    def __init__(self):
        self.base_url = os.environ.get("TTS_API_URL", "https://jebin2-tts.hf.space")
        self.voice = os.environ.get("VOICE_NAME", "4")
        self.speed = os.environ.get("SPEED", "1.0")

    def generate_audio_segment(self, content: str, output_path: str) -> str:
        logger_config.info("Using remote HF TTS API")
        try:
            files = {'file': ('input.txt', content.encode('utf-8'), 'text/plain')}
            data = {
                'voice': str(self.voice),
                'speed': float(self.speed),
                'hide_from_ui': '1'
            }
            logger_config.info(f"voice={data['voice']}, speed={data['speed']}")
            response = requests.post(f"{self.base_url}/api/tasks/upload", files=files, data=data)
            response.raise_for_status()
            task_id = response.json().get("id")
            
            iteration = 0
            while True:
                status_response = requests.get(f"{self.base_url}/api/tasks/{task_id}")
                status_response.raise_for_status()
                task_info = status_response.json()
                
                status = task_info.get("status")
                queue_pos = task_info.get("queue_position")
                progress = task_info.get("progress")
                
                if status == 'not_started':
                    logger_config.debug(f"TTS API Status: Queued (Position {queue_pos}) - {iteration}", overwrite=True)
                elif status == 'processing':
                    logger_config.debug(f"TTS API Status: {progress} ({progress}%) - {iteration}", overwrite=True)

                if status == "completed":
                    download_url = f"{self.base_url}/api/download/{task_id}"
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
