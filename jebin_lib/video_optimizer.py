import subprocess
import os
import json
from jebin_lib import utils

def format_size(size_bytes):
    """
    Helper to format bytes into readable strings (MB, GB).
    """
    if size_bytes == 0:
        return "0B"
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    size_name = ("B", "KB", "MB", "GB", "TB")
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def get_file_tags(input_file):
    """
    Helper to get all global metadata tags from the file using ffprobe.
    Returns a dictionary of tags.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            input_file
        ]
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(result)
        return data.get('format', {}).get('tags', {})
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        # Return empty dict if fails, assuming no metadata or file read error
        return {}

def check_if_already_processed(tags):
    """
    Checks if the 'hevc_compressed' signature exists within the tags.
    Prioritizes checking 'description' as JSON, falls back to 'comment'.
    """
    # 1. Check description field (JSON format)
    description = tags.get('description', '')
    if description:
        try:
            data = json.loads(description)
            # Check if our key is present and True
            if isinstance(data, dict) and data.get('hevc_compressed'):
                return True
        except json.JSONDecodeError:
            pass

    # 2. Fallback: Check comment tag (legacy support)
    comment = tags.get('comment', '')
    if 'hevc_compressed' in comment:
        return True

    return False

def convert_and_compare(input_file, output_file, overwrite_original=False):
    """
    Runs FFmpeg with specific HEVC settings, preserves/appends metadata, and prints a comparison.
    """
    # 1. Validate Input
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # 2. Get Tags & Check Metadata
    tags = get_file_tags(input_file)

    if check_if_already_processed(tags):
        print(f"SKIP: '{input_file}' already has 'hevc_compressed' metadata. Skipping.")
        return

    # 3. Get Original File Stats
    try:
        original_size = os.path.getsize(input_file)
    except OSError as e:
        print(f"Error reading input file: {e}")
        return

    print(f"Processing '{input_file}'...")

    # 4. Prepare Metadata (JSON logic)
    existing_description = tags.get('description', "")
    meta_data = {}

    # Try to parse existing description to preserve it
    if existing_description:
        try:
            meta_data = json.loads(existing_description)
            # If it parses but isn't a dict (e.g. list), wrap it
            if not isinstance(meta_data, dict):
                meta_data = {"post": existing_description}
        except json.JSONDecodeError:
            # If it's plain text, wrap it in "post" to match your structure
            meta_data = {"post": existing_description}
            print(f"Converting plain text description to JSON structure.")

    # Add our flag
    meta_data['hevc_compressed'] = True

    # Dump to JSON string
    description_json = json.dumps(meta_data)
    print(f"Setting Metadata: description={description_json}")

    print("Running FFmpeg (this might take a while due to -preset slow)...")

    # 5. Construct FFmpeg Command
    cmd = [
        "ffmpeg",
        "-y",                 # Overwrite output file if it exists
        "-hide_banner",       # Hide build info
        "-loglevel", "info",  # Show progress
        "-i", input_file,
        "-c:v", "libx265",
        "-preset", "slow",
        "-crf", "20",
        "-c:a", "copy",       # Copy audio stream (lossless)

        # Metadata Handling
        "-map_metadata", "0",              # explicitly copy global metadata from input 0
        "-metadata", f"description={description_json}", # Write the JSON to description

        output_file
    ]

    # 6. Execute Command
    try:
        utils.run_ffmpeg(cmd)
    except subprocess.CalledProcessError:
        print("\nError: FFmpeg command failed.")
        return
    except FileNotFoundError:
        print("\nError: FFmpeg not found. Please install FFmpeg.")
        return

    # 7. Get New File Stats
    if not os.path.isfile(output_file):
        print("Error: Output file was not created.")
        return

    new_size = os.path.getsize(output_file)

    # 8. Calculate Differences
    size_diff = original_size - new_size
    percent_change = (size_diff / original_size) * 100

    if size_diff > 0:
        status = "REDUCTION"
        color_start = "\033[92m" # Green
    else:
        status = "INCREASE"
        color_start = "\033[91m" # Red
    color_end = "\033[0m"

    # 9. Print Comparison Table
    print("\n" + "="*50)
    print(f"{'METRIC':<15} | {'ORIGINAL':<12} | {'HEVC (CRF 20)':<12}")
    print("-" * 50)
    print(f"{'File Size':<15} | {format_size(original_size):<12} | {format_size(new_size):<12}")
    print("-" * 50)
    print(f"Result: {color_start}{abs(percent_change):.2f}% File Size {status}{color_end}")
    print("="*50 + "\n")

    # 10. Overwrite Logic
    if overwrite_original:
        print(f"Overwriting original file '{input_file}' with new version...")
        try:
            os.replace(output_file, input_file)
            print("Overwrite complete.")
        except OSError as e:
            print(f"Error overwriting file: {e}")
