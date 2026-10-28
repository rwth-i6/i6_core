__all__ = ["compute_rec_duration", "distinct_channels"]


import logging
import subprocess as sp
import tempfile

import numpy as np
import soundfile as sf


def _compute_rec_duration_convert_to_wav(audio_file: str) -> float:
    """
    Tries to compute the recording duration by converting the file provided as parameter to wav.
    Needs ffmpeg to be installed in the system. Raises `FileNotFoundError` if not present.

    :return: Duration of :param:`audio_file` converted to wav.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:  # Reserve a temporary file.
        # Convert to wav.
        sp.run(
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                "-i",
                audio_file,
                "-y",
                tmp_audio.name,
            ]  # Overwrite the temporary file.
        )
        return compute_rec_duration(tmp_audio.name)  # Now in wav format.


def compute_rec_duration(audio_file: str) -> float:
    """
    Computes the duration of a given recording in seconds.

    :param str audio_file: Path of the recording.
        The accepted formats are mp3, wav, aac or any audio format parseable by soundfile.
    """
    if audio_file.endswith("mp3"):
        try:
            from mutagen.mp3 import MP3

            return MP3(audio_file).info.length
        except ImportError:
            logging.warning(
                "The 'mutagen' module doesn't exist. We recommend installing it. "
                "Falling back to wav conversion (much slower)..."
            )
            try:
                return _compute_rec_duration_convert_to_wav(audio_file)
            except FileNotFoundError:
                # ffmpeg doesn't exist either.
                raise FileNotFoundError(
                    f"mutagen python module not found, required to calculate duration of mp3 file.\n"
                    "ffmpeg binary not found either, so mp3 -> wav conversion can't succeed.\n"
                    "Please either install the mutagen python module or the ffmpeg binary.\n"
                    f"Can't read duration of mp3 file: {audio_file}."
                )
    elif audio_file.endswith("aac"):
        # The aac format has unreliable timestamps. Convert to wav through ffmpeg for an accurate measurement.
        try:
            return _compute_rec_duration_convert_to_wav(audio_file)
        except FileNotFoundError:
            # ffmpeg doesn't exist.
            raise FileNotFoundError(
                "ffmpeg binary not found, so aac -> wav conversion can't succeed.\n"
                f"Refusing to give unreliable timestamp of aac file {audio_file}."
            )
    else:
        # Wav or any format parseable by soundfile.
        with sf.SoundFile(audio_file) as f:
            return f.frames / f.samplerate  # In seconds.


def distinct_channels(audio_file: str, *, max_seconds: float, max_correlation: float, min_rms_ratio: float) -> int:
    """
    :param max_seconds: audio seconds read to decide whether the channels differ
    :return: the number of channels carrying different signals: 1 for mono files and files whose
        channels are copies of each other (correlation above `max_correlation` or one channel
        quieter than `min_rms_ratio` times the other), else the number of channels
    """
    info = sf.info(audio_file)
    if info.channels == 1:
        return 1
    wav, _ = sf.read(audio_file, frames=int(max_seconds * info.samplerate), dtype="float64", always_2d=True)
    wav = wav - wav.mean(0)  # (T, C)
    rms = np.sqrt((wav**2).mean(0))  # (C,)
    for i in range(info.channels):
        for j in range(i + 1, info.channels):
            rms_ratio = min(rms[i], rms[j]) / max(rms[i], rms[j], 1e-12)
            correlation = (wav[:, i] * wav[:, j]).mean() / max(rms[i] * rms[j], 1e-12)
            if correlation > max_correlation or rms_ratio < min_rms_ratio:
                return 1
    return info.channels
