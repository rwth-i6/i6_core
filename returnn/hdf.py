__all__ = ["ReturnnDumpHDFJob", "ReturnnRasrDumpHDFJob", "BlissToPcmHDFJob", "RasrAlignmentDumpHDFJob"]

from dataclasses import dataclass
from enum import Enum, auto
import glob
import math
import numpy as np
import os
import shutil
import soundfile as sf
import subprocess as sp
import tempfile
from typing import List, Optional

from .rasr_training import ReturnnRasrTrainingJob
from i6_core.lib import corpus
from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.lib.rasr_cache import FileArchive
import i6_core.rasr as rasr
from i6_core.util import instanciate_delayed, uopen, write_paths_to_file
from i6_core import util

from sisyphus import *

Path = setup_path(__package__)


class ReturnnDumpHDFJob(Job):
    """
    This Job is a wrapper around the RETURNN tool hdf_dump.py.
    It can be used to dump a dataset into an HDF directly from a string containing a RETURNN Dataset definition
    """

    def __init__(
        self,
        data,
        start_seq=None,
        end_seq=None,
        epoch=None,
        cpu=2,
        mem=8,
        file_size=100,
        time=4,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param dict|Path|str data: a dict, path, or string defining a RETURNN dataset
        :param start_seq: first sequence to dump in the dataset
        :param end_seq: last sequence to dump in the dataset
        :param int epoch: epoch to dump
        :param int cpu: number of CPU cores required
        :param int mem: RAM required in Gb
        :param int file_size: request file space on compute node in Gb
        :param int time: compute time in hours
        :param Optional[Path] returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Optional[Path] returnn_root: file path to the RETURNN repository root folder
        """
        self.data = data  # typing: dict|Path|str
        self.start_seq = start_seq
        self.end_seq = end_seq
        self.epoch = epoch

        self.rqmt = {
            "cpu": cpu,
            "mem": mem,
            "file_size": file_size,
            "time": time,
        }
        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)

        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        data = self.data
        if isinstance(data, dict):
            instanciate_delayed(data)
            with open("dataset.config", "wt") as dataset_file:
                dataset_file.write("#!rnn.py\n")
                dataset_file.write("train = %s\n" % str(data))
            data = "dataset.config"
        elif isinstance(data, tk.Path):
            data = data.get_path()

        (fd, tmp_hdf_file) = tempfile.mkstemp(prefix=gs.TMP_PREFIX, suffix=".hdf")
        os.close(fd)

        args = [
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("tools/hdf_dump.py").get_path(),
            data,
            tmp_hdf_file,
        ]
        if self.start_seq is not None:
            args += ["--start_seq", f"{self.start_seq}"]
        if self.end_seq is not None:
            args += ["--end_seq", f"{self.end_seq}"]
        if self.epoch is not None:
            args += ["--epoch", f"{self.epoch}"]

        sp.check_call(args)
        shutil.move(tmp_hdf_file, self.out_hdf.get_path())

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["cpu"]
        del parsed_args["mem"]
        del parsed_args["file_size"]
        del parsed_args["time"]
        del parsed_args["returnn_python_exe"]
        return super().hash(parsed_args)


class ReturnnRasrDumpHDFJob(ReturnnDumpHDFJob):
    """
    This Job extends the ReturnnDumpHDFJob to allow the use of ExternSprintDataset.

    """

    def __init__(
        self,
        crp,
        feature_flow,
        alignment,
        num_classes,
        buffer_size=200 * 1024,
        cpu=2,
        mem=8,
        file_size=100,
        time=4,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param rasr.CommonRasrParameters crp: common RASR parameters
        :param rasr.FlowNetwork feature_flow:
        :param Path alignment:
        :param int num_classes:
        :param int buffer_size:
        :param int cpu: number of CPU cores required
        :param int mem: RAM required in Gb
        :param int file_size: request file space on compute node in Gb
        :param int time: compute time in hours
        :param Optional[Path] returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Optional[Path] returnn_root: file path to the RETURNN repository root folder
        """

        data = {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
            "sprintConfigStr": "--config=rasr.config --*.LOGFILE=nn-trainer.log --*.TASK=1",
            "partitionEpoch": 1,
        }

        super(ReturnnRasrDumpHDFJob, self).__init__(
            data=data,
            cpu=cpu,
            mem=mem,
            file_size=file_size,
            time=time,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
        )

        self.crp = crp
        self.alignment = alignment
        self.rasr_exe = rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer")
        self.feature_flow = ReturnnRasrTrainingJob.create_flow(feature_flow, alignment)
        (self.rasr_config, self.rasr_post_config,) = ReturnnRasrTrainingJob.create_config(
            crp=crp,
            alignment=alignment,
            num_classes=num_classes,
            buffer_size=buffer_size,
            disregarded_classes=None,
            class_label_file=None,
            extra_rasr_config=None,
            extra_rasr_post_config=None,
            use_python_control=True,
            feature_flow=feature_flow,
        )

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        rasr.RasrCommand.write_config(self.rasr_config, self.rasr_post_config, "rasr.config")
        self.feature_flow.write_to_file("feature.flow")
        with open("dummy.flow", "wt") as f:
            f.write('<?xml version="1.0" ?>\n<network><out name="features" /></network>')


class BlissToPcmHDFJob(Job):
    """
    Gets audio files from a Bliss corpus and stores them as HDF file
    compatible with the RETURNN HDFDataset
    """

    class BaseStrategy:
        def __eq__(self, other):
            return type(other) == type(self)

    @dataclass(frozen=True)
    class PickNth(BaseStrategy):
        channel: int

        def __eq__(self, other):
            return super().__eq__(other) and other.channel == self.channel

    class RoundingScheme(Enum):
        start_and_duration = auto()
        rasr_compatible = auto()

    __sis_hash_exclude__ = {"multi_channel_strategy": BaseStrategy(), "rounding": RoundingScheme.start_and_duration}

    def __init__(
        self,
        bliss_corpus: tk.Path,
        segment_file: Optional[tk.Path] = None,
        output_dtype: str = "int16",
        multi_channel_strategy: BaseStrategy = BaseStrategy(),
        returnn_root: Optional[tk.Path] = None,
        rounding: RoundingScheme = RoundingScheme.start_and_duration,
    ):
        """

        :param bliss_corpus: Bliss corpus to read segments and audio files from
        :param segment_file: segment file that lists allowed segments
        :param output_dtype: dtype that should be written in the hdf (supports float64, float32, int32, int16)
        :param multi_channel_strategy: defines what should happen to multi-channel audio files.
            Currently implemented are:
            BaseStrategy(): no handling, assume only one channel
            PickNth(n): Takes audio from n-th channel
        :param returnn_root: RETURNN repository
        :param rounding: defines how timestamps should be rounded if they do not exactly fall onto a sample:
            start_and_duration will round down the start time and the duration of the segment
            rasr_compatible will round up the start time and round down the end time
        """
        self.set_vis_name("Dump audio to HDF")
        assert output_dtype in ["float64", "float32", "int32", "int16"]

        self.bliss_corpus = bliss_corpus
        self.segment_file = segment_file
        self.output_dtype = output_dtype
        self.multi_channel_strategy = multi_channel_strategy
        self.returnn_root = returnn_root
        self.rounding = rounding

        self.out_hdf = self.output_path("audio.hdf")

        self.rqmt = {}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        returnn_root = None if self.returnn_root is None else self.returnn_root.get_path()
        SimpleHDFWriter = get_returnn_simple_hdf_writer(returnn_root)

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        if self.segment_file:
            with uopen(self.segment_file, "rt") as f:
                segments_whitelist = set(l.strip() for l in f.readlines() if len(l.strip()) > 0)
        else:
            segments_whitelist = None

        out_hdf = SimpleHDFWriter(filename=self.out_hdf, dim=1)

        for recording in c.all_recordings():
            audio_file = recording.audio
            audio = sf.SoundFile(audio_file)

            for segment in recording.segments:
                if (not segments_whitelist) or (segment.fullname() in segments_whitelist):
                    if self.rounding == self.RoundingScheme.start_and_duration:
                        start = int(segment.start * audio.samplerate)
                        duration = int((segment.end - segment.start) * audio.samplerate)
                    elif self.rounding == self.RoundingScheme.rasr_compatible:
                        start = math.ceil(segment.start * audio.samplerate)
                        duration = math.floor(segment.end * audio.samplerate) - start
                    else:
                        raise NotImplementedError(f"RoundingScheme {self.rounding} not implemented.")
                    audio.seek(start)
                    data = audio.read(
                        duration,
                        always_2d=True,
                        dtype=self.output_dtype,
                    )
                    if isinstance(self.multi_channel_strategy, self.PickNth):
                        data = data[:, self.multi_channel_strategy.channel]
                    else:
                        assert data.shape[-1] == 1, "Audio has more than one channel, choose a multi_channel_strategy"
                    out_hdf.insert_batch(
                        inputs=data.reshape(1, -1, 1),
                        seq_len=[data.shape[0]],
                        seq_tag=[segment.fullname()],
                    )

            audio.close()

        out_hdf.close()


class RasrAlignmentDumpHDFJob(Job):
    """
    This Job reads Rasr alignment caches and dump them in hdf files.
    """

    __sis_hash_exclude__ = {"encoding": "ascii", "filter_list_keep": None, "sparse": False}

    def __init__(
        self,
        alignment_caches: List[tk.Path],
        allophone_file: tk.Path,
        state_tying_file: tk.Path,
        data_type: type = np.uint16,
        returnn_root: Optional[tk.Path] = None,
        encoding: str = "ascii",
        filter_list_keep: Optional[tk.Path] = None,
        sparse: bool = False,
    ):
        """
        :param alignment_caches: e.g. output of an AlignmentJob
        :param allophone_file: e.g. output of a StoreAllophonesJob
        :param state_tying_file: e.g. output of a DumpStateTyingJob
        :param data_type: type that is used to store the data
        :param returnn_root: file path to the RETURNN repository root folder
        :param encoding: encoding of the segment names in the cache
        :param filter_list_keep: list of segment names to dump
        :param sparse: writes the data to hdf in sparse format
        """
        self.alignment_caches = alignment_caches
        self.allophone_file = allophone_file
        self.state_tying_file = state_tying_file
        self.data_type = data_type
        self.returnn_root = returnn_root
        self.encoding = encoding
        self.filter_list_keep = filter_list_keep
        self.sparse = sparse

        self.out_hdf_files = [self.output_path(f"data.hdf.{d}") for d in range(len(alignment_caches))]
        self.out_excluded_segments = self.output_path(f"excluded.segments")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 0.5}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, args=range(1, (len(self.alignment_caches) + 1)))
        yield Task("merge", mini_task=True)

    def merge(self):
        excluded_segments = []
        excluded_files = glob.glob("excluded_segments.*")
        for p in excluded_files:
            if os.path.isfile(p):
                with open(p, "r") as f:
                    segments = f.read().splitlines()
                excluded_segments.extend(segments)

        write_paths_to_file(self.out_excluded_segments, excluded_segments)

    def run(self, task_id):
        state_tying = dict(
            (k, int(v)) for l in open(self.state_tying_file.get_path()) for k, v in [l.strip().split()[0:2]]
        )
        num_classes = max(state_tying.values()) + 1

        alignment_cache = FileArchive(self.alignment_caches[task_id - 1].get_path(), encoding=self.encoding)
        alignment_cache.setAllophones(self.allophone_file.get_path())
        if self.filter_list_keep is not None:
            keep_segments = set(open(self.filter_list_keep.get_path()).read().splitlines())
        else:
            keep_segments = None

        returnn_root = None if self.returnn_root is None else self.returnn_root.get_path()
        SimpleHDFWriter = get_returnn_simple_hdf_writer(returnn_root)
        out_hdf = SimpleHDFWriter(
            filename=self.out_hdf_files[task_id - 1],
            dim=num_classes if self.sparse else 1,
            ndim=1 if self.sparse else 2,
        )

        excluded_segments = []

        for file in alignment_cache.ft:
            info = alignment_cache.ft[file]
            seq_name = info.name

            if seq_name.endswith(".attribs"):
                continue
            if keep_segments is not None and seq_name not in keep_segments:
                excluded_segments.append(seq_name)
                continue

            # alignment
            targets = []
            alignment = alignment_cache.read(file, "align")
            if not len(alignment):
                excluded_segments.append(seq_name)
                continue
            alignmentStates = ["%s.%d" % (alignment_cache.allophones[t[1]], t[2]) for t in alignment]
            for allophone in alignmentStates:
                targets.append(state_tying[allophone])

            data = np.array(targets).astype(np.dtype(self.data_type))
            out_hdf.insert_batch(
                inputs=data.reshape(1, -1) if self.sparse else data.reshape(1, -1, 1),
                seq_len=[data.shape[0]],
                seq_tag=[seq_name],
            )

        out_hdf.close()

        if len(excluded_segments):
            write_paths_to_file(f"excluded_segments.{task_id}", excluded_segments)
