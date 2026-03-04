import json
import requests
import time
import os
import subprocess
from custom_logger import logger_config

class HFSTTClient:
    def transcribe(self, source_path: str) -> bool:
        """
        Transcribes the given audio/video file. 
        If it's a video, FFmpeg is used to extract the audio first.
        Saves the result as <source_path>.json and returns True on success, False otherwise.
        """
        logger_config.info("Using remote HF STT API")
        try:
            url = "https://jebin2-stt.hf.space/api/upload"

            target_file = source_path
            temp_audio_path = None

            ext = os.path.splitext(source_path)[1].lower()
            if ext in ['.mp4', '.mkv', '.avi', '.mov']:
                logger_config.info(f"Extracting audio from video {source_path}...")
                temp_audio_path = f"{source_path}.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", source_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                target_file = temp_audio_path

            logger_config.info(f"Uploading file {target_file} to HF STT API (this may take a while)...")

            with open(target_file, 'rb') as f:
                files = {'audio': f}
                data = {'hide_from_ui': 1}
                response = requests.post(url, files=files, data=data)

            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            response.raise_for_status()
            logger_config.success("Upload complete!")

            file_id = response.json().get("id")

            iteration = 0
            while True:
                status_url = f"https://jebin2-stt.hf.space/api/files/{file_id}"
                status_response = requests.get(status_url)
                status_response.raise_for_status()
                file_info = status_response.json()

                status = file_info.get("status")
                queue_pos = file_info.get("queue_position")
                progress = file_info.get("progress", 0)
                progress_text = file_info.get("progress_text") or "Processing..."

                if status == 'not_started':
                    logger_config.debug(f"STT API Status: Queued (Position {queue_pos}) - {iteration}", overwrite=True)
                elif status == 'processing':
                    logger_config.debug(f"STT API Status: {progress_text} ({progress}%) - {iteration}", overwrite=True)

                if status == "completed":
                    result = json.loads(file_info.get("caption", "{}"))
                    out_path = os.path.splitext(source_path)[0] + ".json"
                    with open(out_path, 'w') as out_f:
                        json.dump(result, out_f)
                    return out_path

                elif file_info.get("status") == "failed":
                    logger_config.error(f"HF API STT Failed: {file_info.get('caption')}")
                    return None

                time.sleep(2)
                iteration += 1
        except Exception as e:
            logger_config.error(f"HF API STT failed: {e}. Falling back to local STT...")
            return None
