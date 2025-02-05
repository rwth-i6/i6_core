__all__ = ["DumpRecordingAudiosJob"]

from typing import Dict, List, Optional, Union

from sisyphus import Job, Task, tk

from i6_core.lib import corpus
from i6_core.lib.audio import compute_rec_duration
from i6_core.util import uopen


class DumpRecordingAudiosJob(Job):
    """
    Dump all recordings of a given corpus file, one audio per line.
    """

    def __init__(
        self, corpus_files: Union[tk.Path, List[tk.Path]], dump_durations: bool = False, zip_output: bool = False
    ):
        r"""
        :param corpus_file: Corpus file from which to obtain the audio list.
        :param dump_durations: Whether to dump the durations of the audios along with the audio list.
            The `out_audio_durations` output file will contain an audio/duration pair, separated by `\t`.
        :param zip_output: Whether the output should be zipped.
        """
        self.corpus_files = [corpus_files] if isinstance(corpus_files, tk.Path) else corpus_files
        self.dump_durations = dump_durations

        suffix = ".gz" if zip_output else ""
        self.out_audio_list = self.output_path(f"out.txt{suffix}")
        if dump_durations:
            self.out_audio_durations = self.output_path(f"out_durations.txt{suffix}")

        self.rqmt = {"cpu": 1, "mem": 1.0, "time": 1.0}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        audio_to_duration: Dict[str, Optional[float]] = {}
        for corpus_file in self.corpus_files:
            c = corpus.Corpus()
            c.load(corpus_file.get_path())

            for r in c.all_recordings():
                duration = compute_rec_duration(r.audio) if self.dump_durations else None
                audio_to_duration[r.audio] = duration

        with uopen(self.out_audio_list.get_path(), "wt") as f:
            for audio in audio_to_duration.keys():
                f.write(f"{audio}\n")

        if self.dump_durations:
            with uopen(self.out_audio_durations.get_path(), "wt") as f:
                for audio, duration in audio_to_duration.items():
                    f.write(f"{audio}\t{duration}\n")
