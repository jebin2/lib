import os
import shutil
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from custom_logger import logger_config

class HFDatasetClient:
	def __init__(self, token=None, repo_id=None):
		# --- Strict env checks ---
		self.token = os.getenv("HF_TOKEN") if not token else token
		self.repo_id = os.getenv("HF_REPO_ID") if not repo_id else repo_id

		if not self.token:
			raise ValueError("Environment variable HF_TOKEN is not set.")

		if not self.repo_id:
			raise ValueError("Environment variable HF_REPO_ID is not set.")

		# constant settings
		self.repo_type = "dataset"
		self.branch = "main"

		# init api
		self.api = HfApi(token=self.token)

		logger_config.info(f"HFMediaClient initialized using repo: {self.repo_id}")

	# --------------------------
	#		UPLOAD
	# --------------------------
	def upload(self, local_path: str, repo_path: str):
		logger_config.info(f"Uploading {local_path} → {repo_path}")

		try:
			self.api.upload_file(
				path_or_fileobj=local_path,
				path_in_repo=repo_path,
				repo_id=self.repo_id,
				repo_type=self.repo_type,
				revision=self.branch,
				commit_message=f"Upload media"
			)
			logger_config.success(f"Uploaded: {repo_path}")
			return True
		except Exception as e:
			logger_config.error(f"Upload failed: {e}")
		return False

	# --------------------------
	#	   UPLOAD FOLDER
	# --------------------------
	def upload_folder(self, local_folder: str, repo_base_path: str = "", ignore_patterns=None, delete_patterns=None) -> bool:
		"""
		Upload all files inside a folder (recursive).
		local_folder: The local folder path to upload.
		repo_base_path: Base path inside the repo (e.g., "videos")
		ignore_patterns: List of patterns to ignore (e.g., ["*.txt", "temp/*"])
		delete_patterns: List of patterns to delete from remote before upload (e.g., ["*"] to overwrite completely)
		"""
		if not os.path.isdir(local_folder):
			logger_config.error(f"Folder not found: {local_folder}")
			return False

		logger_config.info(f"Uploading folder: {local_folder}")

		try:
			# Default patterns
			default_ignore = [".git/*", ".DS_Store", ".*"]

			# Merge with user-provided patterns
			if ignore_patterns:
				all_ignore_patterns = default_ignore + ignore_patterns
			else:
				all_ignore_patterns = default_ignore

			# Use HfApi's upload_folder method with multi_commit for better performance on large folders
			self.api.upload_folder(
				folder_path=local_folder,
				path_in_repo=repo_base_path,
				repo_id=self.repo_id,
				repo_type=self.repo_type,
				revision=self.branch,
				commit_message=f"Upload folder: {local_folder}",
				ignore_patterns=all_ignore_patterns,
				delete_patterns=delete_patterns
			)

			logger_config.success("Folder upload completed!")
			return True
		except Exception as e:
			logger_config.error(f"Upload folder failed: {e}")
			return False

	# --------------------------
	#		LIST
	# --------------------------
	def list_files(self):
		"""
		List all files in the Hugging Face dataset repo.
		"""
		logger_config.info("Fetching file list...")

		try:
			info = self.api.dataset_info(
				repo_id=self.repo_id,
				revision=self.branch,
				token=self.token
			)

			files = [item.rfilename for item in info.siblings]

			logger_config.success(f"Found {len(files)} files:")

			return files

		except Exception as e:
			logger_config.error(f"Failed to list files: {e}")
			return []


	# --------------------------
	#		DOWNLOAD
	# --------------------------
	def download(self, repo_path: str, local_path: str):
		logger_config.info(f"Downloading {repo_path} → {local_path}")
		try:
			# Download with local_dir parameter (no cache)
			tmp_path = hf_hub_download(
				repo_id=self.repo_id,
				filename=repo_path,
				repo_type=self.repo_type,
				revision=self.branch,
				token=self.token,
				local_dir=os.path.dirname(local_path)
			)
			
			# If filename differs, rename it
			if tmp_path != local_path:
				shutil.move(tmp_path, local_path)
			
			logger_config.success(f"Downloaded to: {local_path}")
			return True
		except Exception as e:
			logger_config.error(f"Download failed: {e}")
		return False

	# --------------------------
	#	   DOWNLOAD FOLDER
	# --------------------------
	def download_folder(self, repo_path: str, local_folder: str, allow_patterns=None, ignore_patterns=None):
		"""
		Download a folder from the Hugging Face dataset repo.
		repo_path: The folder path in the repo to download (e.g., "videos"). Use "" for the root.
		local_folder: The local directory to save the files to.
		"""
		if not os.path.exists(local_folder):
			os.makedirs(local_folder, exist_ok=True)

		logger_config.info(f"Downloading folder '{repo_path}' → {local_folder}")
		try:
			if repo_path and not allow_patterns:
				# allow_patterns uses glob, so to download a specific folder:
				clean_repo_path = repo_path.rstrip('/')
				allow_patterns = [f"{clean_repo_path}/**"]

			snapshot_download(
				repo_id=self.repo_id,
				repo_type=self.repo_type,
				revision=self.branch,
				token=self.token,
				local_dir=local_folder,
				allow_patterns=allow_patterns,
				ignore_patterns=ignore_patterns
			)
			
			logger_config.success(f"Downloaded folder to: {local_folder}")
			return True
		except Exception as e:
			logger_config.error(f"Download folder failed: {e}")
		return False

	# --------------------------
	#		DELETE
	# --------------------------
	def delete(self, repo_path: str):
		logger_config.info(f"Deleting {repo_path}")

		try:
			self.api.delete_file(
				path_in_repo=repo_path,
				repo_id=self.repo_id,
				repo_type=self.repo_type,
				revision=self.branch,
			)
			logger_config.success(f"Deleted: {repo_path}")
			return True
		except Exception as e:
			logger_config.error(f"Delete failed: {e}")
		return False
