import subprocess
import os
import json
from pathlib import Path
import shutil
from custom_logger import logger_config
import time

def is_valid_audio(file_path):
    if not file_exists(file_path):
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
    if not file_exists(file_path):
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
    if not file_exists(file_path):
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

def get_docker_volume_mounts(config, base_path=None):
    additional_flags = []
    if base_path:
        additional_flags.append(f'-v {base_path}:{config.neko_attach_folder}')
    additional_flags.append(config.policy_volume_mount())
    return additional_flags

def setup_git_repo_get_install_pip(repo_url, target_path, pip_name=None, requirements_file=None, force_install=False):
    is_new = False
    if not dir_exists(target_path):
        logger_config.info(f"Cloning {repo_url} to {target_path}")
        subprocess.run(
            ["git", "clone", repo_url, target_path],
            check=True
        )
        logger_config.info(f"{repo_url} cloned.")
        is_new = True
    else:
        logger_config.info(f"Pulling {target_path}")
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=target_path,
                check=True
            )
            logger_config.info(f"{target_path} pulled.")
        except subprocess.CalledProcessError:
            logger_config.info(f"Git pull failed for {target_path}, deleting and re-cloning...")
            shutil.rmtree(target_path)
            subprocess.run(
                ["git", "clone", repo_url, target_path],
                check=True
            )
            logger_config.info(f"{repo_url} re-cloned to {target_path}.")
            is_new = True

    if is_new or force_install:
        if requirements_file:
            req_path = os.path.join(target_path, requirements_file)
            if file_exists(req_path):
                logger_config.info(f"Installing dependencies from {requirements_file}")
                subprocess.run(
                    [
                        "bash",
                        "-ic",
                        f"penv {pip_name} && pip install -r {requirements_file}"
                    ],
                    check=True,
                    cwd=target_path
                )
                logger_config.info(f"Dependencies from {requirements_file} installed.")
        elif pip_name:
            logger_config.info(f"Installing {pip_name} via pip")
            subprocess.run(
                [
                    "bash",
                    "-ic",
                    f"penv {pip_name} && pip install -e .[{pip_name}]"
                ],
                check=True,
                cwd=target_path
            )
            logger_config.info(f"{pip_name} installed.")
    return target_path

def get_threads():
    import psutil
    return len(psutil.Process().cpu_affinity())

def get_taskset_cores(reserve=2):
    total = os.cpu_count()
    usable = max(1, total - reserve)  # ensure at least 1 core
    cores = list(range(reserve, reserve + usable))
    return ",".join(map(str, cores))

def run_ffmpeg(cmd):
    threads = get_threads()
    cpu_list = get_taskset_cores()

    cmd = [
        "taskset", "-c", cpu_list,
        "nice", "-n", "15",
        "ffmpeg",
        "-nostdin",
        "-threads", str(threads)
    ] + cmd[1:]

    logger_config.debug(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=True)

def rename_file(current_name, new_name):
    try:
        # Rename the file
        os.rename(current_name, new_name)
        logger_config.success(f"File renamed from '{current_name}' to '{new_name}'")
    except Exception as e:
        logger_config.error(f"An error occurred: {e}")

def copy(source, dest):
    try:
        shutil.copy2(source, dest)
        logger_config.success(f"Copied file from '{source}' to '{dest}'")
    except Exception as e:
        logger_config.error(f"An error occurred: {e}")