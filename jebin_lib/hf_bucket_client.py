import os
import threading
from huggingface_hub import HfFileSystem, sync_bucket
from custom_logger import logger_config

FILE_UPLOAD_TIMEOUT = 120  # seconds per file


class HFBucketClient:
    def __init__(self, token=None, bucket_id=None):
        self.token = os.getenv("HF_TOKEN") if not token else token
        self.bucket_id = os.getenv("HF_BUCKET_ID") if not bucket_id else bucket_id

        if not self.token:
            raise ValueError("Environment variable HF_TOKEN is not set.")
        if not self.bucket_id:
            raise ValueError("Environment variable HF_BUCKET_ID is not set.")

        os.environ["HF_TOKEN"] = self.token
        self._fs = HfFileSystem(token=self.token)
        logger_config.info(f"HFBucketClient initialized using bucket: {self.bucket_id}")

    def _bucket_url(self, remote_path=""):
        base = f"hf://buckets/{self.bucket_id}"
        if remote_path:
            return f"{base}/{remote_path.strip('/')}"
        return base

    def _list_remote(self, remote_url):
        """Return {rel_path: size} for all files under remote_url."""
        remote = {}
        try:
            entries = self._fs.find(remote_url, detail=True)
            for path, info in entries.items():
                if info.get("type") == "file":
                    rel = os.path.relpath(path, remote_url.replace("hf://", "", 1).lstrip("/"))
                    # HfFileSystem paths don't have hf:// prefix in find() keys
                    # Compute rel relative to the bucket url path portion
                    bucket_prefix = remote_url.replace("hf://", "")
                    rel = path[len(bucket_prefix):].lstrip("/")
                    remote[rel] = info.get("size", 0)
        except Exception as e:
            logger_config.warning(f"Could not list remote {remote_url}: {e}")
        return remote

    def _upload_file(self, local_path, remote_url):
        """Upload a single file with a per-file timeout."""
        result = {"error": None}

        def _do():
            try:
                self._fs.put_file(local_path, remote_url)
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        t.join(timeout=FILE_UPLOAD_TIMEOUT)
        if t.is_alive():
            logger_config.warning(f"Upload timed out ({FILE_UPLOAD_TIMEOUT}s), skipping: {os.path.basename(local_path)}")
            return False
        if result["error"]:
            logger_config.error(f"Upload failed: {os.path.basename(local_path)}: {result['error']}")
            return False
        return True

    def upload_folder(self, local_folder: str, remote_path: str = "", delete: bool = False) -> bool:
        if not os.path.isdir(local_folder):
            logger_config.error(f"Folder not found: {local_folder}")
            return False

        dst = self._bucket_url(remote_path)
        logger_config.info(f"Uploading folder: {local_folder} → {dst}")

        # Build local file list
        local_files = {}
        for root, _, files in os.walk(local_folder):
            for fname in files:
                abs_path = os.path.join(root, fname)
                rel = os.path.relpath(abs_path, local_folder)
                local_files[rel] = abs_path

        remote_files = self._list_remote(dst)

        uploads = []
        for rel, abs_path in local_files.items():
            local_size = os.path.getsize(abs_path)
            remote_size = remote_files.get(rel)
            if remote_size is None or remote_size != local_size:
                uploads.append((rel, abs_path))

        deletes = []
        if delete:
            for rel in remote_files:
                if rel not in local_files:
                    deletes.append(rel)

        logger_config.info(f"Sync plan: uploads={len(uploads)}, deletes={len(deletes)}, skips={len(local_files) - len(uploads)}")

        for rel, abs_path in uploads:
            remote_file_url = f"{dst}/{rel}"
            logger_config.info(f"Uploading: {rel}")
            self._upload_file(abs_path, remote_file_url)

        for rel in deletes:
            remote_file_url = f"{dst}/{rel}"
            try:
                self._fs.rm(remote_file_url)
                logger_config.info(f"Deleted remote: {rel}")
            except Exception as e:
                logger_config.warning(f"Delete failed: {rel}: {e}")

        logger_config.success("Folder upload completed!")
        return True

    def download_folder(self, remote_path: str, local_folder: str) -> bool:
        os.makedirs(local_folder, exist_ok=True)
        src = self._bucket_url(remote_path)
        logger_config.info(f"Downloading folder: {src} → {local_folder}")
        try:
            sync_bucket(src, local_folder)
            logger_config.success(f"Downloaded folder to: {local_folder}")
            return True
        except Exception as e:
            logger_config.error(f"Download folder failed: {e}")
            return False
