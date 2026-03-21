import subprocess
import os
import json
from pathlib import Path
import shutil
from custom_logger import logger_config
import time
import string
import secrets
import hashlib
import random

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
        additional_flags.append(f'-v "{base_path}":{config.neko_attach_folder}')
    additional_flags.append(config.policy_volume_mount())
    return additional_flags

def _is_repo_complete(target_path, sparse_dirs=None):
    """Return True if target_path is a valid, fully checked-out git repo."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=target_path, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    if sparse_dirs:
        for d in sparse_dirs:
            dir_path = os.path.join(target_path, d)
            if not os.path.isdir(dir_path) or not os.listdir(dir_path):
                return False
    return True

def _clone_repo(repo_url, target_path, sparse_dirs=None):
    """Clone repo, cleaning up on any failure (including KeyboardInterrupt)."""
    try:
        if sparse_dirs:
            subprocess.run(
                ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", repo_url, target_path],
                check=True
            )
            subprocess.run(
                ["git", "sparse-checkout", "set"] + sparse_dirs,
                check=True, cwd=target_path
            )
        else:
            subprocess.run(
                ["git", "clone", repo_url, target_path],
                check=True
            )
    except BaseException:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        raise

def setup_git_repo_get_install_pip(repo_url, target_path, pip_name=None, requirements_file=None, force_install=False, sparse_dirs=None):
    is_new = False
    if dir_exists(target_path) and not _is_repo_complete(target_path, sparse_dirs):
        logger_config.info(f"Incomplete repo detected at {target_path}, deleting and re-cloning...")
        shutil.rmtree(target_path)

    if not dir_exists(target_path):
        logger_config.info(f"Cloning {repo_url} to {target_path}")
        _clone_repo(repo_url, target_path, sparse_dirs)
        logger_config.info(f"{repo_url} cloned.")
        is_new = True
    else:
        logger_config.info(f"Pulling {target_path}")
        try:
            subprocess.run(["git", "reset", "--hard"], cwd=target_path, check=True)
            subprocess.run(["git", "clean", "-fd"], cwd=target_path, check=True)
            subprocess.run(["git", "pull"], cwd=target_path, check=True)
            logger_config.info(f"{target_path} pulled.")
        except BaseException:
            logger_config.info(f"Git pull failed for {target_path}, deleting and re-cloning...")
            shutil.rmtree(target_path)
            _clone_repo(repo_url, target_path, sparse_dirs)
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

def run_ffmpeg(cmd, **kwargs):
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
    return subprocess.run(cmd, text=True, check=True, **kwargs)

def encode_video_consistent(input_path, output_path, start_time=None, duration=None, fps=24, crf=18, preset="fast", include_audio=False):
    """
    Re-encode video with stable timing/PTS so downstream clips have deterministic duration.
    Optional trim is applied with re-encoding to avoid stream-copy timestamp drift.
    """
    cmd = ["ffmpeg", "-y"]

    if start_time is not None:
        safe_start = max(0.0, float(start_time))
        cmd += ["-ss", f"{safe_start:.6f}"]

    cmd += ["-i", input_path]

    if duration is not None:
        safe_duration = max(0.0, float(duration))
        cmd += ["-t", f"{safe_duration:.6f}"]

    cmd += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-vf", f"fps={fps}"
    ]

    if include_audio:
        cmd += ["-c:a", "aac"]
    else:
        cmd += ["-an"]

    cmd += [output_path]
    return run_ffmpeg(cmd)

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

def generate_random_string(length=10):
    characters = string.ascii_letters
    random_string = ''.join(secrets.choice(characters) for _ in range(length))
    return random_string


def generate_random_string_from_input(input_string, length=16):
    # Hash the input string to get a consistent value
    hash_object = hashlib.sha256(input_string.encode())
    hashed_string = hash_object.hexdigest()

    # Use the hash to seed the random number generator
    random.seed(hashed_string)

    # Generate a random string based on the seed
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string

def write_videofile(video_clip, output_path, fps=24):
    audio_file = f'/tmp/{generate_random_string()}.mp3'
    video_clip.write_videofile(
        output_path,
        fps=fps,
        codec='libx264',
        # audio_codec='aac',
        # preset='faster',  # Faster encoding, slightly larger file
        threads=get_threads(),
        bitrate='8000k',  # Adjust based on your quality needs
        remove_temp=True,
        temp_audiofile=audio_file
    )
    remove_file(audio_file)

def write_audiofile(audio_clip, output_path, fps=44100, codec="libmp3lame", bitrate="192k"):
    audio_clip.write_audiofile(
        output_path,
        fps=fps,
        codec=codec,
        bitrate=bitrate
    )

# ------------------------------------------------------------------ text utils

def clean_text(text):
    import re
    text = re.sub(r"\\+", "", text)
    return re.sub(r'\s+', ' ', text).strip()

def only_alpha(text: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z]', '', text).lower()

def is_same_sentence(sentence_1, sentence_2, threshold=0.9):
    import difflib
    s1 = only_alpha(sentence_1)
    s2 = only_alpha(sentence_2)
    similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
    logger_config.info(f"is_same_sentence :: similarity-{similarity}")
    return similarity > threshold

# ------------------------------------------------------------------ gpu utils

def is_gpu_available(verbose=True):
    import torch
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available.")
        return False
    try:
        torch.empty(1, device="cuda")
        if verbose:
            print(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")
        return True
    except RuntimeError as e:
        if "CUDA-capable device(s) is/are busy or unavailable" in str(e) or "CUDA error" in str(e):
            if verbose:
                print("CUDA detected but busy/unavailable. Please CPU.")
            return False
        raise

def get_device(is_vision=False):
    import torch
    if not is_vision and os.getenv("USE_CPU_IF_POSSIBLE", None):
        device = "cpu"
    else:
        device = "cuda" if is_gpu_available() else "cpu"
    if device == "cpu":
        torch.cuda.is_available = lambda: False
    return device

def manage_gpu(size_gb: float = 0, gpu_index: int = 0, action: str = "check"):
    try:
        import pynvml, signal, gc, time
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gb = info.free / 1024**3
        total_gb = info.total / 1024**3
        print(f"\nGPU {gpu_index}: Free {free_gb:.2f} GB / Total {total_gb:.2f} GB")
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        print("\nActive GPU Processes:")
        print(f"{'PID':<8} {'Process Name':<40} {'Used (GB)':<10}")
        print("-" * 60)
        for p in processes:
            used_gb = p.usedGpuMemory / 1024**3
            proc_name = pynvml.nvmlSystemGetProcessName(p.pid).decode(errors="ignore")
            print(f"{p.pid:<8} {proc_name:<40} {used_gb:.2f}")
        if action == "clear_cache":
            try:
                import torch
                gc.collect()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                time.sleep(1)
                print("\n🧹 Cleared PyTorch CUDA cache")
            except ImportError:
                print("\n⚠️ PyTorch not installed, cannot clear cache.")
        elif action == "kill":
            for p in processes:
                proc_name = pynvml.nvmlSystemGetProcessName(p.pid).decode(errors="ignore")
                try:
                    os.kill(p.pid, signal.SIGKILL)
                    print(f"❌ Killed {p.pid} ({proc_name})")
                except Exception as e:
                    print(f"⚠️ Could not kill {p.pid}: {e}")
            manage_gpu(action="clear_cache")
        gc.collect()
        gc.collect()
        return free_gb > size_gb
    except Exception:
        return False

# ------------------------------------------------------------------ json utils

def extract_json(text):
    """Strip non-JSON garbage prefix/suffix and return the JSON portion.

    Handles the AIStudio UI pattern where a prefix such as
    'codeJson[downloadcontent_copyexpand_less' is prepended to the response,
    absorbing the opening '[' of the original JSON array.
    Works for arrays of objects, arrays of strings, and plain objects.
    """
    import re
    first_bracket = text.find('[')
    first_brace = text.find('{')

    if first_bracket == -1 and first_brace == -1:
        return text

    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        after = text[first_bracket + 1:]
        stripped = after.lstrip()
        first_val_char = stripped[0] if stripped else ''
        if first_val_char not in ('"', '[', '{') and not (first_val_char.isdigit() or first_val_char in '-tfn'):
            val_match = re.search(r'["\[{]|[-\d]|true\b|false\b|null\b', after)
            if val_match:
                content_start = first_bracket + 1 + val_match.start()
                end = text.rfind(']')
                return '[' + text[content_start: end + 1] if end > content_start else text[content_start:]
        return text[first_bracket:]
    else:
        return text[first_brace:]

def parse_json(text, schema=None):
    """Extract, repair, and optionally validate JSON from a string.

    Args:
        text: Raw string that may contain garbage prefix/suffix.
        schema: Optional dict to validate structure after parsing. Supported keys:
            - "type": expected Python type, e.g. list or dict
            - "items": dict with "required" (list of required keys in each item),
                       only checked when type is list
            - "required": list of required top-level keys (for dicts)

    Returns:
        Parsed object (list or dict).

    Raises:
        ValueError: if parsing fails or schema validation fails.
    """
    import json_repair

    parsed = json_repair.loads(extract_json(text))

    if schema is None:
        return parsed

    expected_type = schema.get("type")
    if expected_type is not None and not isinstance(parsed, expected_type):
        raise ValueError(f"Expected {expected_type.__name__}, got {type(parsed).__name__}")

    if isinstance(parsed, list):
        item_schema = schema.get("items")
        if item_schema is not None:
            required_keys = item_schema.get("required", [])
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    raise ValueError(f"Item {i} is not a dict: {type(item).__name__}")
                missing = [k for k in required_keys if k not in item]
                if missing:
                    raise ValueError(f"Item {i} missing keys: {missing}")

    elif isinstance(parsed, dict):
        required_keys = schema.get("required", [])
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

    return parsed

def to_rel(path, base_path):
    """Convert an path to relative path."""
    if path and os.path.isabs(path):
        return os.path.relpath(path, base_path)
    return path

def to_abs(path, base_path):
    """Convert an path to absolute path."""
    if path and not os.path.isabs(path):
        return os.path.join(base_path, path)
    return path

def trim_silence(file_path: str, threshold_db: float = -40.0) -> None:
    """Trim silence from both ends of an audio file in-place using ffmpeg."""
    tmp_path = file_path + ".tmp_trim.wav"
    silence_filter = (
        f"silenceremove=start_periods=1:start_silence=0.05:start_threshold={threshold_db}dB,"
        f"areverse,"
        f"silenceremove=start_periods=1:start_silence=0.05:start_threshold={threshold_db}dB,"
        f"areverse"
    )
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-filter:a", silence_filter, tmp_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.replace(tmp_path, file_path)
        logger_config.info(f"Silence trimmed from both ends")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def speed_up_audio(file_path: str, speed: float = 1.25) -> None:
    """Speed up audio in-place using ffmpeg atempo filter (pitch-preserving)."""
    tmp_path = file_path + ".tmp_speed.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-filter:a", f"atempo={speed}", tmp_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.replace(tmp_path, file_path)
        logger_config.info(f"Audio sped up by {speed}x")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
