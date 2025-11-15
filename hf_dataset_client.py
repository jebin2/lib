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
		except Exception as e:
			PrintLogger.error(f"Upload failed: {e}")

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
			for f in files:
				print(" -", f)

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
		except Exception as e:
			PrintLogger.error(f"Download failed: {e}")

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
		except Exception as e:
			PrintLogger.error(f"Delete failed: {e}")


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
