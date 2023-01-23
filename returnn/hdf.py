__all__ = ["ReturnnDumpHDFJob", "ReturnnRasrDumpHDFJob", "BlissToPcmHDFJob"]

import os
import shutil
import soundfile as sf
import subprocess as sp
import tempfile
from typing import Optional

from .rasr_training import ReturnnRasrTrainingJob
from i6_core.lib import corpus
from i6_core.lib.hdf import get_returnn_simple_hdf_writer
import i6_core.rasr as rasr
from i6_core.util import instanciate_delayed, uopen

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
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
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
        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        data = self.data
        if isinstance(data, dict):
            instanciate_delayed(data)
            data = str(data)
        elif isinstance(data, tk.Path):
            data = data.get_path()

        (fd, tmp_hdf_file) = tempfile.mkstemp(prefix=gs.TMP_PREFIX, suffix=".hdf")
        os.close(fd)

        args = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "tools/hdf_dump.py"),
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
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        """

        data = {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": rasr.RasrCommand.select_exe(
                crp.nn_trainer_exe, "nn-trainer"
            ),
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
        (
            self.rasr_config,
            self.rasr_post_config,
        ) = ReturnnRasrTrainingJob.create_config(
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
        rasr.RasrCommand.write_config(
            self.rasr_config, self.rasr_post_config, "rasr.config"
        )
        self.feature_flow.write_to_file("feature.flow")
        with open("dummy.flow", "wt") as f:
            f.write(
                '<?xml version="1.0" ?>\n<network><out name="features" /></network>'
            )


class BlissToPcmHDFJob(Job):
    """
    Gets audio files from a Bliss corpus and stores them as HDF file
    compatible with the RETURNN HDFDataset
    """

    def __init__(
        self,
        bliss_corpus: tk.Path,
        segment_file: Optional[tk.Path] = None,
        output_dtype: str = "int16",
        returnn_root: Optional[tk.Path] = None,
    ):
        """

        :param bliss_corpus: Bliss corpus to read segments and audio files from
        :param segment_file: segment file that lists allowed segments
        :param output_dtype: dtype that should be written in the hdf (supports float64, float32, int32, int16)
        :param returnn_root: RETURNN repository
        """
        self.set_vis_name("Dump audio to HDF")
        assert output_dtype in ["float64", "float32", "int32", "int16"]

        self.bliss_corpus = bliss_corpus
        self.segment_file = segment_file
        self.output_dtype = output_dtype
        self.returnn_root = returnn_root
        self.rqmt = {}

        self.out_hdf = self.output_path("audio.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        returnn_root = (
            None if self.returnn_root is None else self.returnn_root.get_path()
        )
        SimpleHDFWriter = get_returnn_simple_hdf_writer(returnn_root)

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        if self.segment_file:
            with uopen(self.segment_file, "rt") as f:
                segments_whitelist = set(
                    l.strip() for l in f.readlines() if len(l.strip()) > 0
                )
        else:
            segments_whitelist = None

        out_hdf = SimpleHDFWriter(filename=self.out_hdf, dim=1)

        for recording in c.all_recordings():
            audio_file = recording.audio
            audio = sf.SoundFile(audio_file)
            assert audio.channels == 1, "Multichannel audio not yet implemented"

            for segment in recording.segments:
                if (not segments_whitelist) or (
                    segment.fullname() in segments_whitelist
                ):
                    audio.seek(int(segment.start * audio.samplerate))
                    data = audio.read(
                        int((segment.end - segment.start) * audio.samplerate),
                        dtype=self.output_dtype,
                    )
                    out_hdf.insert_batch(
                        inputs=data.reshape(1, -1, 1),
                        seq_len=[data.shape[0]],
                        seq_tag=[segment.fullname()],
                    )

            audio.close()

        out_hdf.close()
