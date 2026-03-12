from transformers import pipeline
import scipy.io.wavfile
import numpy as np
from browser_manager.browser_config import BrowserConfig
from chat_bot_ui_handler import GeminiUIChat
import json_repair
import os
import torch
from contextlib import contextmanager
import time
import gc
import subprocess, sys

from .. import config, common
from jebin_lib import utils

# ---------------------------------
# 1. SYSTEM PROMPT
# ---------------------------------
with open(config.CREATE_MUSIC_SYSTEM_PROMPT, 'r') as file:
    SYSTEM_PROMPT = file.read()

# ---------------------------------
# 2. CONTEXT MANAGER FOR MODEL
# ---------------------------------
@contextmanager
def musicgen_model(model_name="facebook/musicgen-small"):
    print(f"[MusicGen] Loading model: {model_name} ...")
    synthesiser = pipeline("text-to-audio", model=model_name, device=-1 if common.get_device() == "cpu" else 0)
    try:
        yield synthesiser
    finally:
        print("[MusicGen] Unloading model and clearing GPU memory...")
        try:
            del synthesiser.model
            del synthesiser.tokenizer
        except Exception:
            pass
        del synthesiser
        gc.collect()
        if common.is_gpu_available():
            common.manage_gpu("clear_cache")

# ---------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------
def get_music_prompt(system_prompt, text, max_attempts=5):
    """
    Converts narrative text to a MusicGen prompt using an LLM.
    Tries multiple times to ensure valid structured data.
    """
    for _ in range(max_attempts):
        try:
            cfg = BrowserConfig()
            baseUIChat = GeminiUIChat(cfg)
            music_prompt_json = json_repair.loads(baseUIChat.quick_chat(
                user_prompt=text,
                system_prompt=system_prompt
            ))
            return music_prompt_json["prompt"]
        except Exception as e:
            print(f"[PromptGen] Failed to parse: {e}")

        time.sleep(10)

def process_audio(audio_data, background_volume=0.3):
    """
    Scales raw audio for background use and converts to int16.
    Assumes audio_data is in float32 [-1, 1].
    """
    scaled_data = audio_data
    return (scaled_data * 32767).astype(np.int16)

def save_audio(filename, audio_data_int16, sampling_rate):
    """
    Saves processed audio to .wav file.
    """
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_data_int16)

# ---------------------------------
# 4. MAIN PIPELINE
# ---------------------------------
def run_musicgen_pipeline(text, outfile, background_volume=0.3):
    """
    Main function to run the MusicGen pipeline end-to-end.
    Uses context manager to load/unload model automatically.
    """
    if utils.file_exists(outfile):
        return outfile

    prompt_outfile = outfile.replace(".wav", ".txt")

    # Step 1: Create Music Prompt
    if utils.file_exists(prompt_outfile):
        with open(prompt_outfile, 'r') as file:
            music_prompt = file.read()
    else:
        music_prompt = get_music_prompt(SYSTEM_PROMPT, text)
        with open(prompt_outfile, 'w') as file:
            file.write(music_prompt)

    print(f"[Prompt] Generated: {music_prompt}")

    # Step 2: Use context manager to load model, generate music, and unload
    with musicgen_model("facebook/musicgen-small") as synthesiser:
        music = synthesiser(music_prompt, forward_params={"do_sample": True})
    common.manage_gpu(action="clear_cache")
    # Step 3: Process audio
    audio_data = music["audio"]  # float32 in [-1, 1]
    sampling_rate = music["sampling_rate"]
    audio_data_int16 = process_audio(audio_data, background_volume)

    # Step 4: Save audio
    save_audio(outfile, audio_data_int16, sampling_rate)
    print(f"[Output] Saved to '{outfile}' ({len(audio_data_int16) / sampling_rate:.2f} sec)")
    return outfile

# ---------------------------------
# 5. ENTRY POINT
# ---------------------------------
if __name__ == "__main__":
    print(sys.argv)
    prompt = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) == 3 else None
    run_musicgen_pipeline(text=prompt, outfile=outfile)
