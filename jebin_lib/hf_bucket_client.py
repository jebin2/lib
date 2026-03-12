import os
from huggingface_hub import sync_bucket
from custom_logger import logger_config


class HFBucketClient:
    def __init__(self, token=None, bucket_id=None):
        self.token = os.getenv("HF_TOKEN") if not token else token
        self.bucket_id = os.getenv("HF_BUCKET_ID") if not bucket_id else bucket_id

        if not self.token:
            raise ValueError("Environment variable HF_TOKEN is not set.")
        if not self.bucket_id:
            raise ValueError("Environment variable HF_BUCKET_ID is not set.")

        os.environ["HF_TOKEN"] = self.token
        logger_config.info(f"HFBucketClient initialized using bucket: {self.bucket_id}")

    def _bucket_url(self, remote_path=""):
        base = f"hf://buckets/{self.bucket_id}"
        if remote_path:
            return f"{base}/{remote_path.strip('/')}"
        return base

    def upload_folder(self, local_folder: str, remote_path: str = "", delete: bool = False) -> bool:
        """
        Sync local_folder → bucket remote_path.
        delete=True mirrors exactly (removes remote files not present locally).
        """
        if not os.path.isdir(local_folder):
            logger_config.error(f"Folder not found: {local_folder}")
            return False

        dst = self._bucket_url(remote_path)
        logger_config.info(f"Uploading folder: {local_folder} → {dst}")
        try:
            sync_bucket(local_folder, dst, delete=delete)
            logger_config.success("Folder upload completed!")
            return True
        except Exception as e:
            logger_config.error(f"Upload folder failed: {e}")
            return False

    def download_folder(self, remote_path: str, local_folder: str) -> bool:
        """
        Sync bucket remote_path → local_folder.
        """
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
