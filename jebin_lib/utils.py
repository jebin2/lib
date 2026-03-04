import subprocess
import os
import json

def is_valid_audio(file_path):
    if not os.path.exists(file_path):
        return False

    if os.path.getsize(file_path) < 100:
        return False

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-select_streams", "a", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0 and result.stdout != b""
    except Exception:
        return False

def is_valid_video(file_path):
    if not os.path.exists(file_path):
        return False

    if os.path.getsize(file_path) < 100:
        return False

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-select_streams", "v", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0 and result.stdout != b""
    except Exception:
        return False

def is_valid_json(file_path):
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except Exception:
        return False