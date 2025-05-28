__all__ = [
    "ReturnnDumpHDFJob",
    "ReturnnRasrDumpHDFJob",
    "BlissToPcmHDFJob",
    "BlissToAudioHDFJob",
    "RasrAlignmentDumpHDFJob",
]

import array
from dataclasses import dataclass
from enum import Enum, auto
import glob
import io
import itertools
from logging import getLogger
import math
import multiprocessing
import librosa
import numpy as np
import os
import shutil
import soundfile as sf
import subprocess as sp
import sys
import tempfile
from typing import List, Optional, Sequence, Tuple
import wave

from .rasr_training import ReturnnRasrTrainingJob
from i6_core.lib import corpus
from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.lib.rasr_cache import FileArchive
import i6_core.rasr as rasr
from i6_core.util import instanciate_delayed, uopen, write_paths_to_file
from i6_core import util

from sisyphus import gs, tk, Job, Task, setup_path

_logging = getLogger(__name__)
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
        if isinstance(self.data, (dict, str)):
            yield Task("write_config", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def write_config(self):
        """
        Optionally writes a config if self.data is either of type str or a dict, i.e.g not a tk.Path
        """
        data = self.data
        instanciate_delayed(data)
        data = str(data)
        with open("dataset.config", "wt") as dataset_file:
            dataset_file.write("#!rnn.py\n")
            dataset_file.write("train = %s\n" % str(data))

    def run(self):
        if isinstance(self.data, tk.Path):
            data = self.data.get_path()
        else:
            data = "dataset.config"

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
        os.chmod(tmp_hdf_file, 0o644)
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
        self.rasr_config, self.rasr_post_config = ReturnnRasrTrainingJob.create_config(
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

    See BlissToAudioHDFJob for a faster and more robust version of this job.
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

    __sis_hash_exclude__ = {
        "multi_channel_strategy": BaseStrategy(),
        "rounding": RoundingScheme.start_and_duration,
        "round_factor": 1,
        "target_sampling_rate": None,
    }

    def __init__(
        self,
        bliss_corpus: tk.Path,
        segment_file: Optional[tk.Path] = None,
        output_dtype: str = "int16",
        multi_channel_strategy: BaseStrategy = BaseStrategy(),
        returnn_root: Optional[tk.Path] = None,
        rounding: RoundingScheme = RoundingScheme.start_and_duration,
        round_factor: int = 1,
        target_sampling_rate: Optional[int] = None,
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
        :param round_factor: do the rounding based on a sampling rate that is scaled down by this factor
        :param target_sampling_rate: desired sampling rate for the HDF, data will be resampled to this rate if needed
        """
        self.set_vis_name("Dump audio to HDF")
        assert output_dtype in ["float64", "float32", "int32", "int16"]

        self.bliss_corpus = bliss_corpus
        self.segment_file = segment_file
        self.output_dtype = output_dtype
        self.multi_channel_strategy = multi_channel_strategy
        self.returnn_root = returnn_root
        self.rounding = rounding
        self.round_factor = round_factor
        self.target_sampling_rate = target_sampling_rate

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
                segments_whitelist = {line.strip() for line in f.readlines() if len(line.strip()) > 0}
        else:
            segments_whitelist = None

        out_hdf = SimpleHDFWriter(filename=self.out_hdf, dim=1)

        for recording in c.all_recordings():
            audio_file = recording.audio
            audio = sf.SoundFile(audio_file)

            for segment in recording.segments:
                if (segments_whitelist is not None) and (segment.fullname() not in segments_whitelist):
                    continue

                # determine correct start and duration values
                if self.rounding == self.RoundingScheme.start_and_duration:
                    start = int(segment.start * audio.samplerate / self.round_factor) * self.round_factor
                    duration = (
                        int((segment.end - segment.start) * audio.samplerate / self.round_factor) * self.round_factor
                    )
                elif self.rounding == self.RoundingScheme.rasr_compatible:
                    start = math.ceil(segment.start * audio.samplerate / self.round_factor) * self.round_factor
                    duration = (
                        math.floor(segment.end * audio.samplerate / self.round_factor) * self.round_factor - start
                    )
                else:
                    raise NotImplementedError(f"RoundingScheme {self.rounding} not implemented.")

                # read audio data
                audio.seek(start)
                data = audio.read(duration, always_2d=True, dtype=self.output_dtype)
                if isinstance(self.multi_channel_strategy, self.PickNth):
                    data = data[:, self.multi_channel_strategy.channel]
                else:
                    assert data.shape[-1] == 1, (
                        "Audio has more than one channel, choose a supported multi_channel_strategy. "
                        f"Currently using {self.multi_channel_strategy}."
                    )

                # resample if necessary
                if (sr := self.target_sampling_rate) is not None and sr != audio.samplerate:
                    data = librosa.resample(
                        y=data.astype(float),
                        orig_sr=audio.samplerate,
                        target_sr=sr,
                        axis=0,
                    ).astype(self.output_dtype)

                # add audio to hdf
                out_hdf.insert_batch(
                    inputs=data.reshape(1, -1, 1),
                    seq_len=[data.shape[0]],
                    seq_tag=[segment.fullname()],
                )

            audio.close()

        out_hdf.close()


class BlissToAudioHDFJob(Job):
    """
    Gets audio files from a Bliss corpus and stores them as HDF file compatible with
    the RETURNN HDFDataset.

    More robust and faster version of `BlissToPcmHDFJob`, suitable for processing
    large scale corpora. The increased speed is mainly due to a better I/O
    efficiency. In some situations, `BlissToPcmHDFJob` will end up loading the same
    audio file multiple times from the disk, while this job takes care to only load
    each audio file once (per unit of `concurrent`).

    It, however, will place the segments in the HDF not in the order they occur in
    the split files, but in the order they occur in the corpus. If you depend on the
    order of the segments, you should use the split files as seq ordering files in
    training.

    Can optionally write compressed audio data to the HDF.

    See:
        - https://github.com/rwth-i6/i6_core/pull/593 for discussion,
        - https://github.com/rwth-i6/i6_core/pull/593#issuecomment-2883024538 for why
          this job is faster than `BlissToPcmHDFJob`.
    """

    @dataclass(frozen=True)
    class Mixdown(BlissToPcmHDFJob.BaseStrategy):
        """Multi channel strategy that instructs FFmpeg to mix down all channels to one channel."""

    def __init__(
        self,
        bliss_corpus: tk.Path,
        splits: Sequence[tk.Path],
        *,
        ffmpeg_output_args: Optional[Sequence[str]] = None,
        multi_channel_strategy: Optional[BlissToPcmHDFJob.BaseStrategy] = None,
        output_dtype: Optional[str] = "int16",
        returnn_root: Optional[tk.Path] = None,
        rounding: BlissToPcmHDFJob.RoundingScheme = BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
        round_factor: int = 1,
        target_sampling_rate: int = 16000,
        concurrent: int = 1,
        cpu_rqmt: int = 1 + 6,  # default: 3 workers per core, as the job is I/O bound, +1 for the main process
        mem_rqmt: int = 2 + (0.5 * 18),  # default: 2GB + 0.5GB per worker
        num_workers: int = 18,
    ):
        """
        :param bliss_corpus: Bliss corpus to read segments and audio files from
        :param splits: List of segment files that list the segments per HDF.
            The job creates one HDF per split.
        :param ffmpeg_output_args: Optional additional arguments to pass to FFmpeg.
            These arguments are passed to a second FFmpeg call that can be used to
            optionally compress the audio data and store it in an encoded form.
            E.g. to compress the audio data to OGG Vorbis, you can use:
            `ffmpeg_output_args=("-c:a", "libvorbis", "-qscale:a", "0", "-f", "ogg")`.
            When these arguments are set, the audio data is always written as raw bytes
            into the HDF (i.e. `dtype=numpy.uint8`).
            For this reason, `output_dtype` must be set to `None` when using this argument.
        :param multi_channel_strategy: defines what should happen to multi-channel audio files.
            Currently implemented are:
            Mixdown(): Mix down all channels to one channel, default.
            PickNth(n): Takes audio from n-th channel.
        :param output_dtype: dtype that should be written in the hdf (supports float64, float32, int16).
            If writing compressed data, must be set to None as the compressed audio is always written as uint8 (raw bytes).
        :param returnn_root: RETURNN repository
        :param rounding: defines how timestamps should be rounded if they do not exactly fall onto a sample:
            start_and_duration will round down the start time and the duration of the segment
            rasr_compatible will round up the start time and round down the end time
        :param round_factor: do the rounding based on a sampling rate that is scaled down by this factor
        :param target_sampling_rate: desired sampling rate for the HDF, data will be resampled to this rate if needed
        :param concurrent: Split up the list of splits into this many concurrent jobs.
            Recommended is about one unit of concurrency per 1000h of audio.
            This value affects how I/O efficient the job is. With increasing concurrency
            the I/O efficiency decreases as recordings may end up having to be read multiple
            times from disk.
            Within job concurrency is handled by the multiprocessing library, using `num_workers`
            as parallelism factor.
            Note that within-job concurrency is more I/O efficient than between-job concurrency,
            so prefer increasing `num_workers` over increasing `concurrent`, when possible.
        :param cpu_rqmt: How many CPUs to assign to the job.
            The job is mainly I/O limited, so it's okay to assign fewer CPUs than the number of worker processes.
        :param mem_rqmt: How much memory to assign to the job.
        :param num_workers: Num subprocs (multiprocessing.Pool size) used for parallel recording processing.
            It can be increased to e.g. match the number of CPU cores on a big cluster machine,
            and the job will stay I/O efficient.
        """
        self.bliss_corpus = bliss_corpus
        self.splits = splits
        if ffmpeg_output_args is not None:
            assert ffmpeg_output_args, "ffmpeg_output_args must not be empty"
            assert output_dtype is None, (
                "when using ffmpeg_output_args, output_dtype must be None as the data is always written as raw bytes."
            )
            self.output_dtype = "int16"
        else:
            assert output_dtype in ["float64", "float32", "int16"]
            self.output_dtype = output_dtype
        self.ffmpeg_output_args = ffmpeg_output_args
        self.multi_channel_strategy = multi_channel_strategy or BlissToAudioHDFJob.Mixdown()
        assert isinstance(self.multi_channel_strategy, (BlissToAudioHDFJob.Mixdown, BlissToPcmHDFJob.PickNth))
        self.rounding = rounding
        assert round_factor > 0
        self.round_factor = round_factor
        self.returnn_root = returnn_root
        self.target_sampling_rate = target_sampling_rate

        assert concurrent > 0
        self.concurrent = concurrent
        assert num_workers > 0
        self.num_workers = num_workers

        self.out_hdfs = [self.output_path(f"{i + 1:0d}.hdf") for i in range(len(splits))]

        self.rqmt = {"cpu": cpu_rqmt, "mem": mem_rqmt, "time": 48}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, args=range(self.concurrent), resume="run")

    def run(self, index: int):
        out_hdfs = list(util.chunks(self.out_hdfs, self.concurrent))[index]
        splits = list(util.chunks(self.splits, self.concurrent))[index]
        assert len(out_hdfs) == len(splits)
        _logging.info(f"Writing audio to {out_hdfs} from {splits}.")

        SimpleHDFWriter = get_returnn_simple_hdf_writer(self.returnn_root.get_path())
        hdf_writers = [SimpleHDFWriter(filename=out_hdf.get_path(), dim=1) for out_hdf in out_hdfs]
        segment_whitelists = []
        for split in splits:
            with uopen(split, "rt") as f:
                segments_whitelist = {line.strip() for line in f if len(line.strip()) > 0}
            segment_whitelists.append(segments_whitelist)

        assert len(segment_whitelists) == len(out_hdfs)
        all_whitelists = set.union(*segment_whitelists)

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        pool = multiprocessing.Pool(self.num_workers)
        recs = (
            # We send every segment of the recording grouped together with the audio to
            # avoid having to read the audio file more than once.
            #
            # We only pass down the required audio/segment metadata instead of the whole
            # `i6_core.lib.corpus.Segment` object to avoid excessive pickling overhead,
            # stalling the worker processes (if we don't).
            (rec.audio, segments)
            for rec in c.all_recordings()
            if len(
                segments := [
                    (full_name, segment.start, segment.end)
                    for segment in rec.segments
                    if (full_name := segment.fullname()) in all_whitelists
                ]
            )
            > 0
        )
        # use imap instead of imap_unordered to have reproducible results, even though this
        # causes a slight speed penalty
        for results in pool.imap(self._process_seq, recs, chunksize=8):
            for (segment_name, data), (hdf_writer, segments_whitelist) in itertools.product(
                results, zip(hdf_writers, segment_whitelists)
            ):
                if segment_name not in segments_whitelist:
                    continue
                _logging.info(f"Writing {segment_name} to {hdf_writer.filename}.")
                hdf_writer.insert_batch(
                    inputs=data.reshape(1, -1, 1),
                    seq_len=[data.shape[0]],
                    seq_tag=[segment_name],
                )
        for hdf_writer in hdf_writers:
            hdf_writer.close()

    def _process_seq(self, recording: Tuple[str, Sequence[Tuple[str, float, float]]]) -> List[Tuple[str, np.ndarray]]:
        audio_file, segments = recording

        assert isinstance(self.multi_channel_strategy, (BlissToPcmHDFJob.Mixdown, BlissToPcmHDFJob.PickNth)), (
            "unknown multi_channel_strategy"
        )
        channel_mix_args = (
            ["-ac", "1"]
            if isinstance(self.multi_channel_strategy, BlissToPcmHDFJob.Mixdown)
            else ["-map_channel", f"0.{self.multi_channel_strategy.channel}"]
        )

        # We preprocess all data with ffmpeg because it is more robust to different
        # audio formats like gsm-ms (which soundfile cannot read) or mp3 (standard format,
        # but soundfile cannot seek). We also resample it at the same time and merge channels.
        #
        # We then use the wave library to extract the segments because we want to slice the audio
        # data on a frame-by-frame basis (to match with RASR), while ffmpeg supports only
        # temporal slices.
        ffmpeg_proc = sp.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                audio_file,
                "-c:a",
                "pcm_s16le",
            ]
            + channel_mix_args
            + [
                "-af",
                "aresample=resampler=soxr",
                "-ar",
                str(self.target_sampling_rate),
                "-f",
                "wav",
                "-",
            ],
            check=True,
            stdout=sp.PIPE,
        )

        # we don't read with soundfile as the wave module is more robust
        with wave.open(io.BytesIO(ffmpeg_proc.stdout), "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == self.target_sampling_rate

            audio_data: bytes = audio.readframes(audio.getnframes())

        audio_data_int16 = array.array("h")
        audio_data_int16.frombytes(audio_data)
        if sys.byteorder == "big":
            audio_data_int16.byteswap()
        audio_data = np.array(audio_data_int16, dtype=np.int16)
        if self.output_dtype.startswith("float"):
            # scale to output range
            audio_data = audio_data.astype(self.output_dtype) / np.iinfo(audio_data.dtype).max

        def _process_segment(segment: Tuple[str, float, float]):
            segment_name, start, end = segment
            if self.rounding == BlissToPcmHDFJob.RoundingScheme.start_and_duration:
                start = int(start * self.target_sampling_rate / self.round_factor) * self.round_factor
                duration = int((end - start) * self.target_sampling_rate / self.round_factor) * self.round_factor
            elif self.rounding == BlissToPcmHDFJob.RoundingScheme.rasr_compatible:
                start = math.ceil(start * self.target_sampling_rate / self.round_factor) * self.round_factor
                duration = math.floor(end * self.target_sampling_rate / self.round_factor) * self.round_factor - start
            else:
                raise NotImplementedError(f"RoundingScheme {self.rounding} not implemented.")
            assert start + duration <= len(audio_data)
            data = audio_data[start : start + duration]

            if self.ffmpeg_output_args is not None:
                data_arr = array.array("h")
                data_arr.frombytes(data.tobytes())
                if sys.byteorder == "big":
                    data_arr.byteswap()
                data_bytes = data_arr.tobytes()
                ffmpeg_proc = sp.run(  # we do not use check_output to forward stderr to the job log
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ar",
                        str(self.target_sampling_rate),
                        "-f",
                        "s16le",
                        "-i",
                        "-",
                        *self.ffmpeg_output_args,
                        "-",
                    ],
                    check=True,
                    input=data_bytes,
                    stdout=sp.PIPE,
                )
                data = np.frombuffer(ffmpeg_proc.stdout, dtype=np.uint8)

            return (segment_name, data)

        return [_process_segment(segment) for segment in segments]

    @classmethod
    def hash(cls, kwargs):
        kwargs = kwargs.copy()
        kwargs.pop("concurrent", None)
        kwargs.pop("cpu_rqmt", None)
        kwargs.pop("mem_rqmt", None)
        kwargs.pop("num_workers", None)
        return super().hash(kwargs)


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
        self.out_excluded_segments = self.output_path("excluded.segments")

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
