import os
import shutil
from huggingface_hub import HfApi, hf_hub_download


class PrintLogger:
	@staticmethod
	def info(msg):
		print(f"[INFO] {msg}")

	@staticmethod
	def success(msg):
		print(f"[SUCCESS] {msg}")

	@staticmethod
	def error(msg):
		print(f"[ERROR] {msg}")


class HFDatasetClient:
	def __init__(self):
		# --- Strict env checks ---
		self.token = os.getenv("HF_TOKEN")
		self.repo_id = os.getenv("HF_REPO_ID")

		if not self.token:
			raise ValueError("Environment variable HF_TOKEN is not set.")

		if not self.repo_id:
			raise ValueError("Environment variable HF_REPO_ID is not set.")

		# constant settings
		self.repo_type = "dataset"
		self.branch = "main"

		# init api
		self.api = HfApi(token=self.token)

		PrintLogger.info(f"HFMediaClient initialized using repo: {self.repo_id}")

	# --------------------------
	#		UPLOAD
	# --------------------------
	def upload(self, local_path: str, repo_path: str):
		PrintLogger.info(f"Uploading {local_path} → {repo_path}")

		try:
			self.api.upload_file(
				path_or_fileobj=local_path,
				path_in_repo=repo_path,
				repo_id=self.repo_id,
				repo_type=self.repo_type,
				revision=self.branch,
				commit_message=f"Upload media"
			)
			PrintLogger.success(f"Uploaded: {repo_path}")
			return True
		except Exception as e:
			PrintLogger.error(f"Upload failed: {e}")
		return False

	# --------------------------
	#	   UPLOAD FOLDER
	# --------------------------
	def upload_folder(self, local_folder: str, repo_base_path: str = "") -> bool:
		"""
		Upload all files inside a folder (recursive).
		local_folder: The local folder path to upload.
		repo_base_path: Base path inside the repo (e.g., "videos")
		"""
		if not os.path.isdir(local_folder):
			PrintLogger.error(f"Folder not found: {local_folder}")
			return False

		PrintLogger.info(f"Uploading folder: {local_folder}")

		try:
			for root, dirs, files in os.walk(local_folder):
				for file in files:
					# Skip hidden files like .DS_Store, .gitkeep, etc.
					if file.startswith("."):
						continue

					local_path = os.path.join(root, file)

					# Compute relative path inside repo
					rel_path = os.path.relpath(local_path, local_folder)

					# Target path inside repo
					repo_path = os.path.join(repo_base_path, rel_path).replace("\\", "/")

					PrintLogger.info(f"Uploading file: {local_path} → {repo_path}")

					ok = self.upload(local_path, repo_path)

					if not ok:
						PrintLogger.error(f"Failed uploading: {local_path}")
						return False

			PrintLogger.success("Folder upload completed!")
			return True

		except Exception as e:
			PrintLogger.error(f"Upload folder failed: {e}")
			return False

	# --------------------------
	#		LIST
	# --------------------------
	def list_files(self):
		"""
		List all files in the Hugging Face dataset repo.
		"""
		PrintLogger.info("Fetching file list...")

		try:
			info = self.api.dataset_info(
				repo_id=self.repo_id,
				revision=self.branch,
				token=self.token
			)

			files = [item.rfilename for item in info.siblings]

			PrintLogger.success(f"Found {len(files)} files:")

			return files

		except Exception as e:
			PrintLogger.error(f"Failed to list files: {e}")
			return []


	# --------------------------
	#		DOWNLOAD
	# --------------------------
	def download(self, repo_path: str, local_path: str):
		PrintLogger.info(f"Downloading {repo_path} → {local_path}")

		try:
			tmp_path = hf_hub_download(
				repo_id=self.repo_id,
				filename=repo_path,
				repo_type=self.repo_type,
				revision=self.branch,
				token=self.token,
			)

			os.makedirs(os.path.dirname(local_path), exist_ok=True)
			shutil.move(tmp_path, local_path)

			PrintLogger.success(f"Downloaded to: {local_path}")
			return True
		except Exception as e:
			PrintLogger.error(f"Download failed: {e}")
		return False

	# --------------------------
	#		DELETE
	# --------------------------
	def delete(self, repo_path: str):
		PrintLogger.info(f"Deleting {repo_path}")

		try:
			self.api.delete_file(
				path_in_repo=repo_path,
				repo_id=self.repo_id,
				repo_type=self.repo_type,
				revision=self.branch,
			)
			PrintLogger.success(f"Deleted: {repo_path}")
			return True
		except Exception as e:
			PrintLogger.error(f"Delete failed: {e}")
		return False


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------
if __name__ == "__main__":
	try:
		client = HFDatasetClient()
	except ValueError as err:
		PrintLogger.error(err)
		exit(1)

	# client.upload("local.mp4", "videos/local.mp4")
	# client.download("videos/local.mp4", "downloads/video.mp4")
	# client.delete("videos/local.mp4")
