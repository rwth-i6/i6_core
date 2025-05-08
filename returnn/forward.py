"""
RETURNN forward jobs
"""

__all__ = ["ReturnnForwardJob", "ReturnnForwardJobV2"]

from sisyphus import *

import copy
import glob
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Union

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint
import i6_core.util as util


Path = setup_path(__package__)

Checkpoint = Union[TfCheckpoint, PtCheckpoint, tk.Path]


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
        model_checkpoint: Optional[Checkpoint],
        returnn_config: ReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        hdf_outputs: Optional[List[str]] = None,
        eval_mode: bool = False,
        *,  # args below are keyword only
        log_verbosity: int = 5,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
    ):
        """

        :param model_checkpoint: Checkpoint object pointing to a stored RETURNN Tensorflow/PyTorch model
            or None if network has no parameters or should be randomly initialized
        :param returnn_config: RETURNN config object
        :param returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param returnn_root: path to the RETURNN src folder
        :param hdf_outputs: list of additional hdf output layer file names that the network generates (e.g. attention.hdf);
          The hdf outputs have to be a valid subset or be equal to the hdf_dump_layers in the config.
        :param eval_mode: run forward in eval mode, the default hdf is not available in this case and no search will be done.
        :param log_verbosity: RETURNN log verbosity
        :param device: RETURNN device, cpu or gpu
        :param time_rqmt: job time requirement in hours
        :param mem_rqmt: job memory requirement in GB
        :param cpu_rqmt: job cpu requirement
        """
        self.returnn_config = returnn_config
        if model_checkpoint is None:
            assert not eval_mode, "Eval requires a checkpoint"
        self.model_checkpoint = model_checkpoint
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root
        self.eval_mode = eval_mode
        self.log_verbosity = log_verbosity
        self.device = device

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
            model_checkpoint=self.model_checkpoint,
            returnn_config=self.returnn_config,
            eval_mode=self.eval_mode,
            log_verbosity=self.log_verbosity,
            device=self.device,
        )
        config.write(self.out_returnn_config_file.get_path())

        cmd = [
            self.returnn_python_exe.get_path(),
            os.path.join(self.returnn_root.get_path(), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        if self.model_checkpoint is not None:
            assert os.path.exists(_get_model_path(self.model_checkpoint).get_path()), (
                f"Provided model checkpoint does not exists: {self.model_checkpoint}"
            )

    def run(self):
        # run everything in a TempDir as writing HDFs can cause heavy load
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as d:
            print("using temp-dir: %s" % d)
            call = [
                self.returnn_python_exe.get_path(),
                os.path.join(self.returnn_root.get_path(), "rnn.py"),
                self.out_returnn_config_file.get_path(),
            ]

            try:
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(self.rqmt["cpu"])
                env["MKL_NUM_THREADS"] = str(self.rqmt["cpu"])
                subprocess.check_call(call, cwd=d, env=env)
            except Exception as e:
                print("Run crashed - copy temporary work folder as 'crash_dir'")
                shutil.copytree(d, "crash_dir")
                raise e

            # move log and tensorboard
            shutil.move(os.path.join(d, "returnn.log"), "returnn.log")
            tensorboard_dirs = glob.glob(os.path.join(d, "eval-*"))
            for dir in tensorboard_dirs:
                shutil.move(dir, os.path.basename(dir))

            # move hdf outputs to output folder
            for k, v in self.out_hdf_files.items():
                shutil.move(os.path.join(d, k), v.get_path())

    @classmethod
    def create_returnn_config(
        cls,
        model_checkpoint: Optional[Checkpoint],
        returnn_config: ReturnnConfig,
        eval_mode: bool,
        log_verbosity: int,
        device: str,
        **_kwargs_unused,
    ) -> ReturnnConfig:
        """
        Update the config locally to make it ready for the forward/eval task.
        The resulting config will be used for hashing.

        :param model_checkpoint:
        :param returnn_config:
        :param eval_mode:
        :param log_verbosity:
        :param device:
        :return:
        """
        assert device in ["gpu", "cpu"]
        assert "task" not in returnn_config.config
        assert "load" not in returnn_config.config
        assert "model" not in returnn_config.config

        res = copy.deepcopy(returnn_config)

        if model_checkpoint is not None:
            config = {
                "load": model_checkpoint,
                "task": "eval" if eval_mode else "forward",
            }
        else:
            config = {"task": "forward", "allow_random_model_init": True}

        post_config = {
            "device": device,
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
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


class ReturnnForwardJobV2(Job):
    """
    Generic forward job.

    The user specifies the outputs in the RETURNN config
    via `forward_callback`.
    That is expected to be an instance of `returnn.forward_iface.ForwardCallbackIface`
    or a callable/function which returns such an instance.

    The callback is supposed to generate the output files in the current directory.
    The current directory will be a local temporary directory
    and the files are moved to the output directory at the end.

    Nothing is enforced here by intention, to keep it generic.
    The task by default is set to "forward",
    but other tasks of RETURNN might be used as well.
    """

    def __init__(
        self,
        *,
        model_checkpoint: Optional[Checkpoint],
        returnn_config: ReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        output_files: List[str],
        log_verbosity: int = 5,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
    ):
        """
        :param model_checkpoint: Checkpoint object pointing to a stored RETURNN Tensorflow/PyTorch model
            or None if network has no parameters or should be randomly initialized
        :param returnn_config: RETURNN config object
        :param returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param returnn_root: path to the RETURNN src folder
        :param output_files: list of output file names that will be generated. These are just the basenames,
            and they are supposed to be created in the current directory.
        :param log_verbosity: RETURNN log verbosity
        :param device: RETURNN device, cpu or gpu
        :param time_rqmt: job time requirement in hours
        :param mem_rqmt: job memory requirement in GB
        :param cpu_rqmt: job cpu requirement
        """
        self.returnn_config = returnn_config
        self.model_checkpoint = model_checkpoint
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root
        self.log_verbosity = log_verbosity
        self.device = device

        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_files = {output: self.output_path(output) for output in output_files}

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
        """create files"""
        config = self.create_returnn_config(
            model_checkpoint=self.model_checkpoint,
            returnn_config=self.returnn_config,
            log_verbosity=self.log_verbosity,
            device=self.device,
        )
        config.write(self.out_returnn_config_file.get_path())

        cmd = [
            self.returnn_python_exe.get_path(),
            os.path.join(self.returnn_root.get_path(), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        if self.model_checkpoint is not None:
            assert os.path.exists(_get_model_path(self.model_checkpoint).get_path()), (
                f"Provided model checkpoint does not exists: {self.model_checkpoint}"
            )

    def run(self):
        """run"""
        # run everything in a TempDir as writing files can cause heavy load
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
            print("using temp-dir: %s" % tmp_dir)
            call = [
                self.returnn_python_exe.get_path(),
                os.path.join(self.returnn_root.get_path(), "rnn.py"),
                self.out_returnn_config_file.get_path(),
            ]

            try:
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(self.rqmt["cpu"])
                env["MKL_NUM_THREADS"] = str(self.rqmt["cpu"])
                subprocess.check_call(call, cwd=tmp_dir, env=env)
            except Exception:
                print("Run crashed - copy temporary work folder as 'crash_dir'")
                if os.path.exists("crash_dir"):
                    shutil.rmtree("crash_dir")
                shutil.copytree(tmp_dir, "crash_dir", dirs_exist_ok=True)
                raise

            # move outputs to output folder
            for k, v in self.out_files.items():
                assert os.path.exists(f"{tmp_dir}/{k}"), f"Output file {k} does not exist"
                shutil.move(f"{tmp_dir}/{k}", v.get_path())

            # copy logs and anything else. don't make assumptions on filenames
            shutil.copytree(tmp_dir, ".", dirs_exist_ok=True)

    @classmethod
    def create_returnn_config(
        cls,
        *,
        model_checkpoint: Optional[Checkpoint],
        returnn_config: ReturnnConfig,
        log_verbosity: int,
        device: str,
        **_kwargs,
    ):
        """
        Update the config locally to make it ready for the forward/eval task.
        The resulting config will be used for hashing.

        :param model_checkpoint:
        :param returnn_config:
        :param log_verbosity:
        :param device:
        :return:
        """
        assert "load" not in returnn_config.config
        assert "model" not in returnn_config.config

        res = copy.deepcopy(returnn_config)

        res.config.setdefault("task", "forward")
        if model_checkpoint is not None:
            res.config["load"] = model_checkpoint
        else:
            res.config.setdefault("allow_random_model_init", True)

        res.post_config.setdefault("device", device)
        res.post_config.setdefault("log", ["./returnn.log"])
        res.post_config.setdefault("tf_log_dir", "returnn-tf-log")
        res.post_config.setdefault("log_verbosity", log_verbosity)

        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config": cls.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return super().hash(d)


def _get_model_path(model: Checkpoint) -> tk.Path:
    if isinstance(model, tk.Path):
        return model
    if isinstance(model, TfCheckpoint):
        return model.index_path
    if isinstance(model, PtCheckpoint):
        return model.path
    raise TypeError(f"Unknown model checkpoint type: {type(model)}")
