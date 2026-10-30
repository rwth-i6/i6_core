"""
Speaker diarization of whole recordings with pyannote pipelines.
"""

__all__ = ["PyannoteDiarizeRecordingsJob"]

import json
import os

from sisyphus import Job, Task, tk


def _load_mix(audio_file: str):
    """
    :return: the downmix of all channels (1, T) and the sample rate
    """
    import soundfile
    import torch

    wav, sample_rate = soundfile.read(audio_file, dtype="float32", always_2d=True)
    mix = torch.from_numpy(wav.T.copy()).mean(0, keepdim=True)  # (T, C) -> (1, T)
    return mix, sample_rate


def _synchronize():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _diarize(pipeline, mix, sample_rate: int):
    """
    :param mix: (1, T)
    :return: the diarized `(speaker, start, end)` sorted by start, and the seconds the diarization took
    """
    import time

    import torchaudio

    _synchronize()
    start_time = time.monotonic()
    if sample_rate != pipeline._audio.sample_rate:
        # the pipeline would resample every overlapping chunk separately
        mix = torchaudio.functional.resample(mix.to(pipeline.device), sample_rate, pipeline._audio.sample_rate).cpu()
        sample_rate = pipeline._audio.sample_rate
    output = pipeline({"waveform": mix, "sample_rate": sample_rate})
    _synchronize()
    seconds = time.monotonic() - start_time
    segments = sorted(
        ((str(speaker), float(turn.start), float(turn.end)) for turn, speaker in output.speaker_diarization),
        key=lambda s: (s[1], s[2]),
    )
    return segments, seconds


class PyannoteDiarizeRecordingsJob(Job):
    """
    pyannote diarization of whole audio files, multi-channel files from their downmix. Writes per file
    `rttm/<name>.rttm` (`SPEAKER <name> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>`, empty for
    audio shorter than a second) and `stats/<name>.json`, where `name` is the recording name for a
    bliss corpus and the file name without extension otherwise. The files are split into `num_shards`
    tasks of about equal audio duration; an interrupted shard resumes. The pipeline is loaded from the
    HuggingFace hub with the token of `huggingface_hub.get_token`.
    """

    def __init__(
        self,
        *,
        audio_files: list[tk.Path] | tk.Path,
        python_site_pkg: tk.Path,
        diarization_model: str = "pyannote/speaker-diarization-community-1",
        num_shards: int = 1,
    ):
        """
        :param audio_files: audio files, a bliss corpus (`.xml`, `.xml.gz`), or a text file with one audio
            file path per line
        :param python_site_pkg: site-packages with pyannote.audio
        :param diarization_model: pyannote pipeline on the HuggingFace hub
        :param num_shards: number of GPU tasks
        """
        self.audio_files = audio_files
        self.python_site_pkg = python_site_pkg
        self.diarization_model = diarization_model
        self.num_shards = num_shards

        self.out_rttm_dir = self.output_path("rttm", directory=True)
        self.out_stats_dir = self.output_path("stats", directory=True)
        self.out_summary = self.output_path("summary.json")

        self.rqmt = {"cpu": 4, "mem": 16, "gpu": 1, "gpu_mem": 16, "time": 12}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, self.num_shards + 1))
        yield Task("summarize", mini_task=True)

    def _recordings(self) -> list[tuple[str, str]]:
        """
        :return: `(name, audio file)` per recording
        """
        from i6_core.lib.corpus import Corpus
        from i6_core.util import uopen

        if not isinstance(self.audio_files, tk.Path):
            audio_files = [f.get_path() for f in self.audio_files]
        elif self.audio_files.get_path().endswith((".xml", ".xml.gz")):
            bliss = Corpus()
            bliss.load(self.audio_files.get_path())
            recordings = [(r.name, r.audio) for r in bliss.all_recordings()]
            assert len({name for name, _ in recordings}) == len(recordings), "recording names are not unique"
            return recordings
        else:
            with uopen(self.audio_files, "rt") as f:
                audio_files = [line.strip() for line in f if line.strip()]
        names = [os.path.splitext(os.path.basename(f))[0] for f in audio_files]
        assert len(set(names)) == len(names), "audio file names are not unique"
        return list(zip(names, audio_files))

    def _load_pipeline(self):
        import sys

        sys.path.insert(0, self.python_site_pkg.get_path())
        import torch
        from huggingface_hub import get_token
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(self.diarization_model, token=get_token())
        return pipeline.to(torch.device("cuda"))

    def run(self, task_id: int):
        import soundfile

        from i6_core.util import balanced_chunks

        recordings = self._recordings()
        durations = {name: soundfile.info(audio_file).duration for name, audio_file in recordings}
        shard = balanced_chunks(recordings, self.num_shards, lambda r: durations[r[0]])[task_id - 1]
        pipeline = self._load_pipeline()
        for n, (name, audio_file) in enumerate(shard):
            self._process(name, audio_file, pipeline=pipeline)
            print(f"[{n + 1}/{len(shard)}] {audio_file}", flush=True)

    def _process(self, name: str, audio_file: str, *, pipeline):
        rttm_path = os.path.join(self.out_rttm_dir.get_path(), f"{name}.rttm")
        stats_path = os.path.join(self.out_stats_dir.get_path(), f"{name}.json")
        if os.path.exists(rttm_path) and os.path.exists(stats_path):
            return

        mix, sample_rate = _load_mix(audio_file)
        stats = {"name": name, "duration": mix.shape[-1] / sample_rate}
        segments = []
        if mix.shape[-1] >= sample_rate:
            segments, seconds = _diarize(pipeline, mix, sample_rate)
            stats.update(
                {
                    "speech": sum(end - start for _, start, end in segments),
                    "num_speakers": len({speaker for speaker, _, _ in segments}),
                    "t_diarization": seconds,
                }
            )
        else:
            stats["empty"] = True

        with open(rttm_path + ".tmp", "wt") as f:
            for speaker, start, end in segments:
                f.write(f"SPEAKER {name} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>\n")
        with open(stats_path + ".tmp", "wt") as f:
            json.dump(stats, f)
        os.replace(stats_path + ".tmp", stats_path)
        os.replace(rttm_path + ".tmp", rttm_path)

    def summarize(self):
        from collections import Counter

        stats = []
        for name, _ in self._recordings():
            with open(os.path.join(self.out_stats_dir.get_path(), f"{name}.json")) as f:
                stats.append(json.load(f))
        processed = [s for s in stats if not s.get("empty")]
        audio = sum(s["duration"] for s in processed)
        t_diarization = sum(s["t_diarization"] for s in processed)
        summary = {
            "model": self.diarization_model,
            "recordings": len(stats),
            "empty_recordings": len(stats) - len(processed),
            "hours": {"audio": audio / 3600, "diarized_speech": sum(s["speech"] for s in processed) / 3600},
            "num_speakers_per_recording": dict(sorted(Counter(s["num_speakers"] for s in processed).items())),
            "rtf_diarization": t_diarization / audio if audio else None,
            "gpu_hours": t_diarization / 3600,
        }
        with open(self.out_summary.get_path(), "wt") as f:
            json.dump(summary, f, indent=2)
