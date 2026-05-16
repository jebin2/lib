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

## Scripts

### Cloudflare Tunnel Setup

Sets up a Cloudflare Tunnel to expose local services on a public domain. Supports single or multiple services under one tunnel.

**Prerequisites:** `cloudflared` installed on the server and `voidall.com` (or your domain) managed on Cloudflare.

```bash
# Install cloudflared (Debian/Ubuntu)
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

**Single service:**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jebin2/lib/main/scripts/setup_cloudflare_tunnel.sh)" bash opencode.voidall.com:7860
```

**Multiple services (one tunnel, one config):**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jebin2/lib/main/scripts/setup_cloudflare_tunnel.sh)" bash opencode.voidall.com:7860 nvr.voidall.com:2126
```

**Custom tunnel name:**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jebin2/lib/main/scripts/setup_cloudflare_tunnel.sh)" bash --name myserver opencode.voidall.com:7860
```

The tunnel name defaults to the first subdomain label (e.g. `opencode` from `opencode.voidall.com`).

After setup, check status with:
```bash
sudo systemctl status cloudflared
journalctl -u cloudflared -f
```

---

## Modules

- **jebin_lib** - Main package
  - `HFDatasetClient` - Upload/download files to HuggingFace datasets
  - `load_env` - `.env` file loader
  
- **jebin_lib.scene_detect** - Scene detection tools
  - `SimpleNSFWDetector` - NSFW content detection in videos
