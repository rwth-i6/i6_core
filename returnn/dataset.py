__all__ = ["ExtractDatasetMeanStddevJob"]

from sisyphus import *

import os
import pickle
import shutil
import subprocess

import numpy

from i6_core.returnn.config import ReturnnConfig
from i6_core.lib import corpus
from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.util import create_executable, uopen


class ExtractDatasetMeanStddevJob(Job):
    """
    Runs the RETURNN tool dump-dataset with statistic extraction.
    Collects mean and std-var for each feature as file and in total as sisyphus var.

    Outputs:

    Variable out_mean: a global mean over all sequences and features
    Variable out_std_dev: a global std-dev over all sequences and features

    Path out_mean_file: a text file with #feature entries for the mean
    Path out_std_dev_file: a text file with #features entries for the standard deviation
    """

    def __init__(self, returnn_config, returnn_python_exe=None, returnn_root=None):
        """

        :param ReturnnConfig returnn_config:
        :param Path|str|None returnn_python_exe:
        :param Path|str|None returnn_root:
        """

        self.returnn_config = returnn_config
        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.out_mean = self.output_var("mean_var")
        self.out_std_dev = self.output_var("std_dev_var")
        self.out_mean_file = self.output_path("mean")
        self.out_std_dev_file = self.output_path("std_dev")

        self.rqmt = {"cpu": 2, "mem": 4, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        self.returnn_config.write("returnn.config")

        command = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "tools/dump-dataset.py"),
            "returnn.config",
            "--endseq -1",
            "--stats",
            "--dump_stats stats",
        ]

        create_executable("rnn.sh", command)
        subprocess.check_call(["./rnn.sh"])

        shutil.move("stats.mean.txt", self.out_mean_file.get_path())
        shutil.move("stats.std_dev.txt", self.out_std_dev_file.get_path())

        total_mean = 0
        total_var = 0

        with open(self.out_mean_file.get_path()) as mean_file, open(
            self.out_std_dev_file.get_path()
        ) as std_dev_file:

            # compute the total mean and std-dev in an iterative way
            for i, (mean, std_dev) in enumerate(zip(mean_file, std_dev_file)):
                mean = float(mean)
                var = float(std_dev.strip()) ** 2
                mean_variance = (total_mean - mean) ** 2
                adjusted_mean_variance = mean_variance * i / (i + 1)
                total_var = (total_var * i + var + adjusted_mean_variance) / (i + 1)
                total_mean = (total_mean * i + mean) / (i + 1)

            self.out_mean.set(total_mean)
            self.out_std_dev.set(numpy.sqrt(total_var))


class SpeakerLabelHDFFromBlissJob(Job):
    """
    Extract speakers from a bliss corpus and create an HDF file with a speaker index
    matching the speaker entry in the corpus speakers for each segment
    """

    def __init__(self, bliss_corpus, returnn_root=None):
        """
        :param bliss_corpus: bliss XML corpus where the speakers and segments are taken from
        :param returnn_root: used to import SimpleHDFWriter from a specific path if not installed in the worker env
        """
        self.bliss_corpus = bliss_corpus
        self.returnn_root = returnn_root
        self.out_speaker_hdf = self.output_path("speaker_labels.hdf")
        self.out_num_speakers = self.output_var("num_speakers")
        self.out_speaker_dict = self.output_path("speaker_dict.pkl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        bliss = corpus.Corpus()
        bliss.load(self.bliss_corpus.get_path())
        speaker_by_index = {}
        index_by_speaker = {}
        num_speakers = len(bliss.speakers)
        self.out_num_speakers.set(num_speakers)
        # speakers are stored as OrderedDict, so this is a safe operation
        for i, speaker in enumerate(bliss.all_speakers()):
            speaker_by_index[i] = speaker.name
            index_by_speaker[speaker.name] = i

        pickle.dump(speaker_by_index, uopen(self.out_speaker_dict, "wb"))

        SimpleHDFWriter = get_returnn_simple_hdf_writer(
            returnn_root=self.returnn_root.get_path() if self.returnn_root else None
        )
        hdf_writer = SimpleHDFWriter(
            self.out_speaker_hdf.get_path(), dim=num_speakers, ndim=1
        )

        for recording in bliss.all_recordings():
            for segment in recording.segments:
                speaker_name = segment.speaker_name or recording.speaker_name
                speaker_index = index_by_speaker[speaker_name]
                segment_name = segment.fullname()
                hdf_writer.insert_batch(
                    numpy.asarray([[speaker_index]], dtype="int32"), [1], [segment_name]
                )

        hdf_writer.close()
