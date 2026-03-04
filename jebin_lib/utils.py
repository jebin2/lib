import subprocess
import os
import json
from pathlib import Path
import shutil
from custom_logger import logger_config
import time

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
    except:
        return False

def is_valid_json(file_path):
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except:
        return False


def path_exists(path):
    return file_exists(path) or dir_exists(path)

def file_exists(file_path):
    try:
        return Path(file_path).is_file()
    except:
        pass
    return False

def dir_exists(file_path):
    try:
        return Path(file_path).is_dir()
    except:
        pass
    return False

def list_files_recursive(directory):
    remove_zone_identifier(directory)
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def list_directories_recursive(directory):
    remove_zone_identifier(directory)
    directory_list = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            directory_list.append(os.path.join(root, dir_name))
    
    return directory_list

def remove_zone_identifier(directory):
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(":Zone.Identifier"):
                    full_path = os.path.join(root, file)
                    remove_file(full_path)
    except: pass

def list_files(directory):
    remove_zone_identifier(directory)
    file_list = []
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if os.path.isfile(full_path):
            file_list.append(full_path)
    
    return file_list

def remove_path(path):
    remove_file(path, True)
    remove_all_files_and_dirs(path)

def remove_file(file_path, retry=True):
    try:
        if file_exists(file_path):
            Path(file_path).unlink()
    except Exception as e:
        if retry:
            time.sleep(10)
            remove_file(file_path, False)
        else:
            logger_config.warning(f"Error occurred while trying to remove the file: {e}")

def remove_all_files_and_dirs(directory):
    try:
        shutil.rmtree(directory)
    except Exception as e:
        logger_config.warning(f"Failed to delete {directory}. Reason: {e}")

def remove_directory(directory_path):
    try:
        if dir_exists(directory_path):
            shutil.rmtree(directory_path)
    except Exception as e:
        logger_config.warning(f'An error occurred: {e}')

def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        logger_config.error(f'An error occurred: {e}')

def get_docker_volume_mounts(base_path, config):
    additional_flags = []
    additional_flags.append(f'-v {base_path}:{config.neko_attach_folder}')
    additional_flags.append(config.policy_volume_mount())
    return additional_flags