import os
import subprocess

from custom_logger import logger_config


def ensure_nfs_sudo():
    sudoers_file = "/etc/sudoers.d/hf-mount"
    mount_nfs = subprocess.run(["which", "mount.nfs"], capture_output=True, text=True).stdout.strip()
    if not mount_nfs:
        logger_config.info("Installing nfs-common...")
        subprocess.run(["sudo", "apt-get", "install", "-y", "nfs-common"], check=True)
        mount_nfs = subprocess.run(["which", "mount.nfs"], capture_output=True, text=True).stdout.strip()
    if os.path.exists(sudoers_file):
        return
    user = os.getlogin()
    rules = (
        f"{user} ALL=(ALL) NOPASSWD: {mount_nfs}\n"
        f"{user} ALL=(ALL) NOPASSWD: /bin/umount\n"
        f"{user} ALL=(ALL) NOPASSWD: /usr/bin/umount\n"
    )
    logger_config.info("Configuring passwordless sudo for NFS mount...")
    subprocess.run(
        ["sudo", "tee", sudoers_file],
        input=rules, text=True, check=True, capture_output=True
    )
    subprocess.run(["sudo", "chmod", "440", sudoers_file], check=True)


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


def cleanup_stale_files(path, _is_child=False):
    """Recursively delete any files/dirs under path that have stale NFS handles."""
    import shutil
    logger_config.info(f"Scanning for stale entries in: {path}")
    try:
        entries = list(os.scandir(path))
    except OSError as e:
        if e.errno == 116:
            if _is_child:
                logger_config.warning(f"Stale directory detected (scandir failed), removing: {path}")
                shutil.rmtree(path, ignore_errors=True)
                logger_config.info(f"Removed stale directory: {path}")
            else:
                logger_config.warning(f"Stale directory detected (scandir failed) on root path: {path} — skipping delete")
        else:
            logger_config.error(f"Failed to scan {path}: {e}")
        return

    # Test write access — existing files may stat fine but creating new files can still raise ESTALE
    if _is_child:
        test_path = os.path.join(path, '.stale_write_check')
        try:
            with open(test_path, 'w') as f:
                f.write('')
            os.remove(test_path)
        except OSError as e:
            if e.errno == 116:
                logger_config.warning(f"Stale directory detected (write test failed), removing: {path}")
                shutil.rmtree(path, ignore_errors=True)
                logger_config.info(f"Removed stale directory: {path}")
                return
            else:
                logger_config.error(f"Write test failed on {path}: {e}")

    stale_count = 0
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                cleanup_stale_files(entry.path, _is_child=True)
            else:
                # utime forces SETATTR on NFS server — non-destructive write test
                try:
                    os.utime(entry.path, None)
                except OSError as write_err:
                    if write_err.errno == 116:
                        raise  # stale — outer except will delete the file
                    # EACCES (read-only) — fall back to read test
                    fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        os.read(fd, 1)
                    finally:
                        os.close(fd)
        except OSError as e:
            if e.errno == 116:
                logger_config.warning(f"Stale entry detected: {entry.path}")
                shutil.rmtree(entry.path, ignore_errors=True)
                logger_config.info(f"Removed stale entry: {entry.path}")
                stale_count += 1
            else:
                logger_config.error(f"Unexpected OSError on {entry.path}: {e}")
    if stale_count:
        logger_config.info(f"Cleanup complete: removed {stale_count} stale entries under {path}")
    else:
        logger_config.info(f"No stale entries found in: {path}")


def is_mount_stale(mount_path):
    try:
        os.listdir(mount_path)
        return False
    except OSError as e:
        return e.errno == 116


def remount_hf(hf_bucket_id, hf_token, mount_path):
    logger_config.info(f"Stale file handle detected, remounting {mount_path}...")
    subprocess.run(["hf-mount", "stop", mount_path], capture_output=True)
    result = subprocess.run(
        ["hf-mount", "start", "--hf-token", hf_token,
         "bucket", hf_bucket_id, mount_path],
        capture_output=True, text=True
    )
    if result.returncode != 0 and "daemon already running" not in result.stderr:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    logger_config.info(f"Remounted {mount_path} successfully.")


def ensure_hf_mounted(hf_bucket_id, hf_token, mount_path):
    if not hf_bucket_id or not hf_token:
        return
    ensure_hf_mount_installed()
    ensure_nfs_sudo()
    if mount_path and is_mount_stale(mount_path):
        remount_hf(hf_bucket_id, hf_token, mount_path)
        return
    result = subprocess.run(["hf-mount", "status"], capture_output=True, text=True)
    if mount_path in result.stdout or mount_path in result.stderr:
        return
    if not os.path.exists(mount_path):
        subprocess.run(["sudo", "mkdir", "-p", mount_path], check=True)
        subprocess.run(["sudo", "chown", os.getlogin(), mount_path], check=True)
    logger_config.info(f"Mounting HF bucket {hf_bucket_id} at {mount_path}")
    result = subprocess.run(
        ["hf-mount", "start", "--hf-token", hf_token,
         "bucket", hf_bucket_id, mount_path],
        capture_output=True, text=True
    )
    if result.returncode != 0 and "daemon already running" not in result.stderr:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
