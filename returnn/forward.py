__all__ = ["ReturnnForwardJob"]

from sisyphus import *

import copy
import glob
import os
import shutil
import subprocess as sp
import tempfile

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint
import i6_core.util as util

Path = setup_path(__package__)


class ReturnnForwardJob(Job):
    """
    Run a RETURNN "forward" pass to HDF with a specified model checkpoint.
    Also allows to run an "eval" task pass, which is similar to "forward" but treats all layers as in training mode,
    which can be used to e.g. do cheating experiments.

    Outputs:

    Dict[tk.Path] out_hdf_files: Dictionary of all output HDF files that were requested by the key list `hdf_outputs`
    tk.Path out_default_hdf: For forward (not eval) mode, this contains the default HDF file that is always written
        by RETURNN, independent of other settings.
    """

    def __init__(
        self,
        model_checkpoint,
        returnn_config,
        hdf_outputs=None,
        eval_mode=False,
        *,  # args below are keyword only
        log_verbosity=3,
        device="gpu",
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=4,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param Checkpoint model_checkpoint: Checkpoint object pointing to a stored RETURNN Tensorflow model
        :param ReturnnConfig returnn_config: RETURNN config dict
        :param dict returnn_post_config: RETURNN config dict (no hashing)
        :param list[str] hdf_outputs: list of additional hdf output layer file names that the network generates (e.g. attention.hdf);
          The hdf outputs have to be a valid subset or be equal to the hdf_dump_layers in the config.
        :param bool eval_mode: run forward in eval mode, the default hdf is not available in this case and no search will be done.
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :param int time_rqmt: job time requirement
        :param int mem_rqmt: job memory requirement
        :param int cpu_rqmt: job cpu requirement
        :param Path|str returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param Path|str returnn_root: path to the RETURNN src folder
        """
        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self._model_checkpoint = model_checkpoint
        self._returnn_config = returnn_config
        self._eval_mode = eval_mode
        self._log_verbosity = log_verbosity
        self._device = device

        self.out_returnn_config_file = self.output_path("returnn.config")

        self.out_hdf_files = {}
        hdf_outputs = hdf_outputs if hdf_outputs else []
        for output in hdf_outputs:
            self.out_hdf_files[output] = self.output_path(output)
        if not eval_mode:
            self.out_hdf_files["output.hdf"] = self.output_path("output.hdf")
            self.out_default_hdf = self.out_hdf_files["output.hdf"]

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        config = self.create_returnn_config(
            model_checkpoint=self._model_checkpoint,
            returnn_config=self._returnn_config,
            eval_mode=self._eval_mode,
            log_verbosity=self._log_verbosity,
            device=self._device,
        )
        config.write(self.out_returnn_config_file.get_path())

        cmd = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        assert os.path.exists(
            self._model_checkpoint.index_path.get_path()
        ), "Provided model does not exists: %s" % str(self._model_checkpoint)

    def run(self):
        # run everything in a TempDir as writing HDFs can cause heavy load
        with tempfile.TemporaryDirectory(prefix="work_") as d:
            print("using temp-dir: %s" % d)
            call = [
                tk.uncached_path(self.returnn_python_exe),
                os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
                self.out_returnn_config_file.get_path(),
            ]

            # stash a possible exception until we finished copying files from the temp work to the actual
            # work folder to be able to examine the files
            error = None
            try:
                sp.check_call(call, cwd=d)
            except Exception as e:
                print("Run crashed - copy temporary work folder")
                error = e

            # move log and tensorboard
            shutil.move(os.path.join(d, "returnn.log"), "returnn.log")
            tensorboard_dirs = glob.glob(os.path.join(d, "eval-*"))
            for dir in tensorboard_dirs:
                shutil.move(dir, os.path.basename(dir))

            # move hdf outputs to output folder
            for k, v in self.out_hdf_files.items():
                try:
                    shutil.move(os.path.join(d, k), v.get_path())
                except Exception as e:
                    if error is None:
                        # if we had an error before it is expected to have missing hdf files
                        # so we do not need to raise anything here then
                        raise e

            if error:
                raise error

    @classmethod
    def create_returnn_config(
        cls,
        model_checkpoint,
        returnn_config,
        eval_mode,
        log_verbosity,
        device,
        **kwargs,
    ):
        """

        :param Checkpoint model_checkpoint:
        :param ReturnnConfig returnn_config:
        :param int log_verbosity:
        :param str device:
        :param kwargs:
        :return:
        """
        assert device in ["gpu", "cpu"]
        assert "network" in returnn_config.config

        res = copy.deepcopy(returnn_config)

        config = {"load": model_checkpoint}

        post_config = {
            "device": device,
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
            "task": "eval" if eval_mode else "forward",
        }

        if not eval_mode:
            post_config["forward_override_hdf_output"] = True
            post_config["output_file"] = "output.hdf"

        config.update(returnn_config.config)
        post_config.update(returnn_config.post_config)

        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config": ReturnnForwardJob.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return super().hash(d)
