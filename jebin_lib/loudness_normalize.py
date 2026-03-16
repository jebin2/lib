import json
import re
from custom_logger import logger_config
from jebin_lib.utils import run_ffmpeg


def normalize_loudness(
    input_path: str,
    output_path: str = None,
    target_lufs: float = -14.0,
    target_tp: float = -1.0,
    target_lra: float = 7.0,
) -> str:
    """
    Two-pass EBU R128 loudness normalization using FFmpeg's loudnorm filter.

    Pass 1: Analyze the input to measure actual loudness.
    Pass 2: Apply a precision correction using the measured values.

    Args:
        input_path:  Path to the input audio or video file.
        output_path: Destination path. Defaults to overwriting input_path.
        target_lufs: Target integrated loudness in LUFS. Default -14 (YouTube Shorts standard).
        target_tp:   Target true peak in dBTP. Default -1.0.
        target_lra:  Target loudness range. Default 7.
    Returns:
        Path to the normalized file.
    """
    if output_path is None:
        output_path = input_path

    # --- Pass 1: analyze ---
    logger_config.info("Loudness normalization: analyzing...")
    cmd_analyze = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:print_format=json",
        "-f", "null", "-"
    ]
    try:
        result = run_ffmpeg(cmd_analyze, capture_output=True)
    except Exception as e:
        logger_config.warning(f"Loudness normalization: analysis failed: {e}")
        return input_path
    match = re.search(r"\{[\s\S]*\}", result.stderr)
    if not match:
        logger_config.warning("Loudness normalization: could not parse analysis output, skipping.")
        return input_path

    ln = json.loads(match.group(0))
    logger_config.debug(f"Loudness analysis: I={ln['input_i']} LUFS, TP={ln['input_tp']} dBTP, LRA={ln['input_lra']}")

    # --- Pass 2: apply ---
    logger_config.info("Loudness normalization: applying...")
    loudnorm_filter = (
        f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}:"
        f"measured_I={ln['input_i']}:measured_TP={ln['input_tp']}:"
        f"measured_LRA={ln['input_lra']}:measured_thresh={ln['input_thresh']}:"
        f"offset={ln['target_offset']}:linear=true:print_format=summary"
    )

    # Use a temp file when overwriting in-place to avoid FFmpeg read/write conflict
    in_place = output_path == input_path
    write_target = output_path + ".tmp.mp4" if in_place else output_path

    cmd_apply = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-af", loudnorm_filter,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        write_target
    ]
    try:
        run_ffmpeg(cmd_apply)
    except Exception as e:
        logger_config.warning(f"Loudness normalization failed: {e}")
        return input_path

    if in_place:
        import os
        os.replace(write_target, output_path)

    logger_config.info(f"Loudness normalization complete: {output_path}")
    return output_path
