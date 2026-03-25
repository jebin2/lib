import os
import subprocess

from custom_logger import logger_config


def ensure_hf_mount_installed():
    if subprocess.run(["which", "hf-mount"], capture_output=True).returncode == 0:
        return
    logger_config.info("hf-mount not found, installing...")
    subprocess.run(
        "curl -fsSL https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh",
        shell=True, check=True
    )
    local_bin = os.path.expanduser("~/.local/bin")
    os.environ["PATH"] = local_bin + os.pathsep + os.environ.get("PATH", "")


def ensure_hf_mounted(hf_bucket_id, hf_token, mount_path):
    if not hf_bucket_id or not hf_token:
        return
    ensure_hf_mount_installed()
    result = subprocess.run(["hf-mount", "status"], capture_output=True, text=True)
    if mount_path in result.stdout:
        return
    if not os.path.exists(mount_path):
        subprocess.run(["sudo", "mkdir", "-p", mount_path], check=True)
        subprocess.run(["sudo", "chown", os.getlogin(), mount_path], check=True)
    logger_config.info(f"Mounting HF bucket {hf_bucket_id} at {mount_path}")
    subprocess.run(
        ["hf-mount", "start", "--hf-token", hf_token,
         "bucket", hf_bucket_id, mount_path],
        check=True
    )
