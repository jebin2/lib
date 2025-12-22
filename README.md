# Jebin's Personal Library

A collection of utility modules for my projects.

## Installation

### From Git (recommended)
```bash
# Basic install (HuggingFace dataset client only)
pip install git+https://github.com/jebin2/lib.git

# With scene detection support
pip install "git+https://github.com/jebin2/lib.git#egg=jebin-lib[scene_detect]"

# All optional dependencies
pip install "git+https://github.com/jebin2/lib.git#egg=jebin-lib[all]"
```

### For development
```bash
git clone https://github.com/jebin2/lib.git
cd lib
pip install -e .
```

## Usage

### HuggingFace Dataset Client
```python
from jebin_lib import HFDatasetClient

# Requires environment variables: HF_TOKEN, HF_REPO_ID
client = HFDatasetClient()
client.upload("local_file.mp4", "videos/file.mp4")
client.download("videos/file.mp4", "downloads/file.mp4")
client.list_files()
client.delete("videos/file.mp4")
```

### NSFW Detector (requires scene_detect extras)
```python
from jebin_lib.scene_detect import SimpleNSFWDetector

detector = SimpleNSFWDetector(threshold=0.6)
detector.process_video("video.mp4", output_folder="nsfw_clips")
```

## Modules

- **jebin_lib** - Main package
  - `HFDatasetClient` - Upload/download files to HuggingFace datasets
  - `PrintLogger` - Simple logging utility
  
- **jebin_lib.scene_detect** - Scene detection tools
  - `SimpleNSFWDetector` - NSFW content detection in videos
