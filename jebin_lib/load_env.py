import os
from pathlib import Path


def load_env(project_folder: str = None):
    """
    Load environment variables from .env files in order of priority.
    Later files override earlier ones.
    
    Order:
    1. ~/.env
    2. ~/.envs/.env
    3. ~/.envs/.<project_folder_name>_env
    4. <project_folder>/.env
    
    Args:
        project_folder: Path to the project folder. If None, uses current working directory.
    """
    if project_folder is None:
        project_folder = os.getcwd()
    
    project_name = os.path.basename(os.path.abspath(project_folder))
    home = Path.home()
    
    # Define paths in loading order (later overrides earlier)
    env_paths = [
        home / ".env",
        home / ".envs" / ".env",
        home / ".envs" / f".{project_name}_env",
        Path(project_folder) / ".env",
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            _load_env_file(env_path)
            print(f"[ENV] Loaded: {env_path}")


def _load_env_file(path: Path):
    """Parse and load a .env file into os.environ"""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key] = value
