__all__ = ["ExtractDatasetMeanStddevJob"]

from sisyphus import *

import pickle
import shutil
import subprocess
from typing import Optional, Any, Dict

import numpy

from i6_core.returnn.config import ReturnnConfig
from i6_core.lib import corpus
from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.util import (
    create_executable,
    get_returnn_python_exe,
    get_returnn_root,
    uopen,
)


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

    __sis_hash_exclude__ = {"data_key": "data"}

    def __init__(self, returnn_config, data_key="data", returnn_python_exe=None, returnn_root=None):
        """

        :param ReturnnConfig returnn_config:
        :param str data_key: the data key to extract the mean and std-dev from
        :param Optional[Path] returnn_python_exe:
        :param Optional[Path] returnn_root:
        """

        self.returnn_config = returnn_config
        self.data_key = data_key
        self.returnn_python_exe = get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = get_returnn_root(returnn_root)

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
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("tools/dump-dataset.py").get_path(),
            "returnn.config",
            "--endseq",
            "-1",
            "--type",
            "null",
            "--stats",
            "--dump_stats",
            "stats",
            "--key",
            self.data_key,
        ]

        create_executable("rnn.sh", command)
        subprocess.check_call(["./rnn.sh"])

        shutil.move("stats.mean.txt", self.out_mean_file.get_path())
        shutil.move("stats.std_dev.txt", self.out_std_dev_file.get_path())

        total_mean = 0
        total_var = 0

        with open(self.out_mean_file.get_path()) as mean_file, open(self.out_std_dev_file.get_path()) as std_dev_file:

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

    def __init__(self, bliss_corpus: tk.Path, returnn_root: Optional[tk.Path] = None):
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
        hdf_writer = SimpleHDFWriter(self.out_speaker_hdf.get_path(), dim=num_speakers, ndim=1)

        for recording in bliss.all_recordings():
            for segment in recording.segments:
                speaker_name = segment.speaker_name or recording.speaker_name
                speaker_index = index_by_speaker[speaker_name]
                segment_name = segment.fullname()
                hdf_writer.insert_batch(numpy.asarray([[speaker_index]], dtype="int32"), [1], [segment_name])

        hdf_writer.close()


class ExtractSeqLensJob(Job):
    """
    Extracts sequence lengths from a dataset for one specific key.
    """

    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        dataset: Dict[str, Any],
        post_dataset: Optional[Dict[str, Any]] = None,
        *,
        key: str,
        format: str,
        returnn_config: Optional[ReturnnConfig] = None,
    ):
        """
        :param dataset: dict for :func:`returnn.datasets.init_dataset`
        :param post_dataset: extension of the dataset dict, which is not hashed
        :param key: e.g. "data", "classes" or whatever the dataset provides
        :param format: "py" or "txt"
        :param returnn_config: for the RETURNN global config.
            This is optional and only needed if you use any custom functions (e.g. audio pre_process)
            which expect some configuration in the global config.
        """
        super().__init__()
        self.dataset = dataset
        self.post_dataset = post_dataset
        self.key = key
        assert format in {"py", "txt"}
        self.format = format
        self.returnn_config = returnn_config

        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_file = self.output_path(f"seq_lens.{format}")

        self.rqmt = {"gpu": 0, "cpu": 1, "mem": 4, "time": 1}

    @classmethod
    def hash(cls, parsed_args):
        """hash"""
        parsed_args = parsed_args.copy()
        parsed_args.pop("post_dataset")
        if not parsed_args["returnn_config"]:
            parsed_args.pop("returnn_config")
        return super().hash(parsed_args)

    def tasks(self):
        """tasks"""
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        """create files"""
        config = self.returnn_config or ReturnnConfig({})
        assert "dataset" not in config.config and "dataset" not in config.post_config
        dataset_dict = self.dataset.copy()
        if self.post_dataset:
            # The modification to the config here is not part of the hash anymore,
            # so merge dataset and post_dataset now.
            dataset_dict.update(self.post_dataset)
        config.config["dataset"] = dataset_dict
        config.write(self.out_returnn_config_file.get_path())

    def run(self):
        """run"""
        import tempfile
        import shutil
        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset

        config = Config()
        config.load_file(self.out_returnn_config_file.get_path())
        set_global_config(config)

        dataset_dict = config.typed_value("dataset")
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)
        dataset.init_seq_order(epoch=1)

        with tempfile.NamedTemporaryFile("w") as tmp_file:
            if self.format == "py":
                tmp_file.write("{\n")

            seq_idx = 0
            while dataset.is_less_than_num_seqs(seq_idx):
                dataset.load_seqs(seq_idx, seq_idx + 1)
                seq_tag = dataset.get_tag(seq_idx)
                seq_len = dataset.get_seq_length(seq_idx)
                assert self.key in seq_len.keys()
                seq_len_ = seq_len[self.key]
                if self.format == "py":
                    tmp_file.write(f"{seq_tag!r}: {seq_len_},\n")
                elif self.format == "txt":
                    tmp_file.write(f"{seq_len_}\n")
                else:
                    raise ValueError(f"{self}: invalid format {self.format!r}")
                seq_idx += 1

            if self.format == "py":
                tmp_file.write("}\n")
            tmp_file.flush()

            shutil.copyfile(tmp_file.name, self.out_file.get_path())
