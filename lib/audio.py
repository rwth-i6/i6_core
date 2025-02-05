__all__ = ["compute_rec_duration"]


import subprocess as sp
import tempfile

try:
    from mutagen.mp3 import MP3
except ImportError:
    # No mutagen module. mp3 conversion will not be supported.
    pass
import soundfile as sf


def compute_rec_duration(rec_audio: str) -> float:
    """
    Computes the duration of a given recording in seconds.

    :param str rec_audio: Path of the recording.
        The accepted formats are mp3, wav, aac or any audio format parseable by soundfile.
    """
    if rec_audio.endswith("mp3"):
        try:
            audio_length = MP3(rec_audio).info.length
        except ImportError as ie:
            ie.msg = (
                "The 'mutagen' module is required to calculate the duration of the mp3 file "
                f"{rec_audio}, but it's not installed in your system."
            )
            raise ie
        return audio_length
    elif rec_audio.endswith("aac"):
        # The aac format has unreliable timestamps. Convert to wav on the fly for an accurate measurement.
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:  # Reserve a temporary file.
            # Convert to wav.
            sp.run(
                ["ffmpeg", "-loglevel", "quiet", "-i", rec_audio, "-y", tmp_audio.name]  # Overwrite the temporary file.
            )
            return compute_rec_duration(tmp_audio.name)  # Now in wav format.
    else:
        # Wav or any format parseable by soundfile.
        with sf.SoundFile(rec_audio) as f:
            return f.frames / f.samplerate  # In seconds.
