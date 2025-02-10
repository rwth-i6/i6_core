__all__ = ["compute_rec_duration"]


import logging
import subprocess as sp
import tempfile

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
