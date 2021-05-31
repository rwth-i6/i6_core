__all__ = ["ReturnnDumpHDFJob", "ReturnnRasrDumpHDFJob"]

import os
import subprocess as sp

from .rasr_training import ReturnnRasrTrainingJob
import i6_core.rasr as rasr
from i6_core.util import relink

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

        :param str data: a string defining a RETURNN dataset
        :param start_seq: first sequence to dump in the dataset
        :param end_seq: last sequence to dump in the dataset
        :param int epoch: epoch to dump
        :param int cpu: number of CPU cores required
        :param int mem: RAM required
        :param int file_size: request file space on compute node
        :param int time: compute time
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
        if isinstance(data, str):
            data = str(data)
        else:
            data = tk.uncached_path(data)

        args = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "tools/hdf_dump.py"),
            data,
            "data.hdf",
        ]
        if self.start_seq is not None:
            args.append(f"--start_seq {self.start_seq}")
        if self.end_seq is not None:
            args.append(f"--end_seq {self.end_seq}")
        if self.epoch is not None:
            args.append(f"--epoch {self.epoch}")

        sp.check_call(args)
        relink("data.hdf", self.out_hdf.get_path())

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
        **kwargs,
    ):
        """

        :param rasr.CommonRasrParameters crp: common RASR parameters
        :param rasr.FlowNetwork feature_flow:
        :param Path alignment:
        :param int num_classes:
        :param int buffer_size:
        :param kwargs: parameters for ReturnnDumpHDFJob
        """
        super(ReturnnRasrDumpHDFJob, self).__init__(**kwargs)

        self.data["sprintConfigStr"] = "--config=rasr.config"

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
