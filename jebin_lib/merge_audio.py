from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut, MultiplyVolume
import numpy as np
from custom_logger import logger_config


def get_audio_rms(audio_clip, sample_duration=1.0):
    duration = min(sample_duration, audio_clip.duration)
    audio_segment = audio_clip.subclipped(0, duration)
    audio_array = audio_segment.to_soundarray(fps=44100)
    return np.sqrt(np.mean(audio_array**2))


def calculate_bg_volume(main_rms, bg_rms):
    if main_rms > 0.03:
        base_volume = 0.3
    elif main_rms > 0.01:
        base_volume = 0.4
    else:
        base_volume = 0.5

    if bg_rms > 0.15:
        bg_volume = base_volume * 0.2
    elif bg_rms > 0.08:
        bg_volume = base_volume * 0.4
    elif bg_rms > 0.03:
        bg_volume = base_volume * 0.6
    elif bg_rms > 0.01:
        bg_volume = base_volume * 0.8
    else:
        bg_volume = base_volume * 1.2

    return max(0.1, min(0.8, bg_volume))


def process(audio_path_1, audio_path_2, output_path):
    original_audio = AudioFileClip(audio_path_1)
    bg_music = AudioFileClip(audio_path_2)

    if bg_music.duration > original_audio.duration:
        bg_music = bg_music.subclipped(0, original_audio.duration)
    elif bg_music.duration < original_audio.duration:
        fade_dur = 1.0
        clips = []
        t = 0
        while t < original_audio.duration:
            part = bg_music.subclipped(0, min(bg_music.duration, original_audio.duration - t))
            part = part.with_effects([AudioFadeIn(fade_dur), AudioFadeOut(fade_dur)])
            clips.append(part)
            t += part.duration
        bg_music = concatenate_audioclips(clips).with_duration(original_audio.duration)

    main_rms = get_audio_rms(original_audio)
    bg_rms = get_audio_rms(bg_music)
    bg_volume = calculate_bg_volume(main_rms, bg_rms)

    logger_config.debug(f"Main RMS: {main_rms:.4f}, BG RMS: {bg_rms:.4f}, BG Volume: {bg_volume:.2f}")

    bg_music = bg_music.with_effects([MultiplyVolume(bg_volume)])
    original_audio = original_audio.with_effects([MultiplyVolume(0.8)])

    combined = CompositeAudioClip([original_audio, bg_music])
    combined.write_audiofile(output_path)

    return output_path
