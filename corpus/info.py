__all__ = ["DumpRecordingAudiosJob"]

from sisyphus import Job, Task, tk

from i6_core.lib import corpus
from i6_core.lib.audio import compute_rec_duration


class DumpRecordingAudiosJob(Job):
    """
    Dump all recordings of a given corpus file, one audio per line.
    """

    def __init__(self, corpus_file: tk.Path, dump_durations: bool = False, duration_file_delimiter: str = "\t"):
        r"""
        :param corpus_file: Corpus file from which to obtain the audio list.
        :param dump_durations: Whether to dump the durations of the audios along with the audio list.
            The `out_audio_durations` output file will contain an audio/duration pair,
            separated by :param:`duration_file_delimiter`.
        :param duration_file_delimiter: Delimiter of the duration file. Defaults to `\t`.
            It only makes sense to override this whenever :param:`dump_durations` is `True`.
        """
        self.corpus_file = corpus_file
        self.dump_durations = dump_durations
        self.duration_file_delimiter = duration_file_delimiter

        self.out_audio_list = self.output_path("out.txt")
        if dump_durations:
            self.out_audio_durations = self.output_path("out_durations.txt")

        self.rqmt = {"cpu": 1, "mem": 1.0, "time": 1.0}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        c = corpus.Corpus()
        c.load(self.corpus_file.get_path())

        with open(self.out_audio_list.get_path(), "w") as f:
            for r in c.all_recordings():
                f.write(f"{r.audio}\n")

        if self.dump_durations:
            with open(self.out_audio_durations.get_path(), "w") as f:
                for r in c.all_recordings():
                    duration = compute_rec_duration(r.audio)
                    f.write(f"{r.audio}{self.duration_file_delimiter}{duration}")
