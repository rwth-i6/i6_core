__all__ = [
    "AverageTFCheckpointsJob",
    "AverageTorchCheckpointsJob",
    "Checkpoint",
    "GetBestEpochJob",
    "GetBestTFCheckpointJob",
    "GetBestPtCheckpointJob",
    "PtCheckpoint",
    "ReturnnModel",
    "ReturnnTrainingFromFileJob",
    "ReturnnTrainingJob",
]

import copy
from dataclasses import dataclass
import sys
import os
import shutil
import subprocess as sp
import numpy as np
from typing import Dict, Sequence, Iterable, List, Optional, Union

from sisyphus import *

import i6_core.util as util
from .config import ReturnnConfig

Path = setup_path(__package__)


class ReturnnModel:
    """
    Defines a RETURNN model as config, checkpoint meta file and epoch

    This is deprecated, use :class:`Checkpoint` instead.
    """

    def __init__(self, returnn_config_file: Path, model: Path, epoch: int):
        """

        :param returnn_config_file: Path to a returnn config file
        :param model: Path to a RETURNN checkpoint (only the .meta for Tensorflow)
        :param epoch:
        """
        self.returnn_config_file = returnn_config_file
        self.model = model
        self.epoch = epoch


class Checkpoint:
    """
    Checkpoint object which holds the (Tensorflow) index file path as tk.Path,
    and will return the checkpoint path as common prefix of the .index/.meta/.data[...]

    A checkpoint object should be directly assigned to a RasrConfig entry (do not call `.ckpt_path`)
    so that the hash will resolve correctly
    """

    def __init__(self, index_path):
        """
        :param Path index_path:
        """
        self.index_path = index_path

    def _sis_hash(self):
        return self.index_path._sis_hash()

    @property
    def ckpt_path(self):
        return self.index_path.get_path()[: -len(".index")]

    def __str__(self):
        return self.ckpt_path

    def __repr__(self):
        return "'%s'" % self.ckpt_path

    def exists(self):
        return os.path.exists(self.index_path.get_path())


class PtCheckpoint:
    """
    Checkpoint object pointing to a PyTorch checkpoint .pt file
    """

    def __init__(self, path: tk.Path):
        """
        :param path: .pt file
        """
        self.path = path

    def _sis_hash(self):
        return self.path._sis_hash()

    def __str__(self):
        return self.path.get()

    def __repr__(self):
        return "'%s'" % self.path

    def exists(self):
        return os.path.exists(self.path.get_path())


class ReturnnTrainingJob(Job):
    """
    Train a RETURNN model using the rnn.py entry point.

    Only returnn_config, returnn_python_exe and returnn_root influence the hash.

    The outputs provided are:

     - out_returnn_config_file: the finalized Returnn config which is used for the rnn.py call
     - out_learning_rates: the file containing the learning rates and training scores (e.g. use to select the best checkpoint or generate plots)
     - out_model_dir: the model directory, which can be used in succeeding jobs to select certain models or do combinations
        note that the model dir is DIRECTLY AVAILABLE when the job starts running, so jobs that do not have other conditions
        need to implement an "update" method to check if the required checkpoints are already existing
     - out_checkpoints: a dictionary containing all created checkpoints. Note that when using the automatic checkpoint cleaning
        function of Returnn not all checkpoints are actually available.
    """

    __sis_hash_exclude__ = {"distributed_launch_cmd": "mpirun"}

    def __init__(
        self,
        returnn_config: ReturnnConfig,
        *,  # args below are keyword only
        log_verbosity: int = 3,
        device: str = "gpu",
        num_epochs: int = 1,
        save_interval: int = 1,
        keep_epochs: Optional[Iterable[int]] = None,
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
        distributed_launch_cmd: str = "mpirun",
        horovod_num_processes: Optional[int] = None,
        multi_node_slots: Optional[int] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        """

        :param returnn_config:
        :param log_verbosity: RETURNN log verbosity from 1 (least verbose) to 5 (most verbose)
        :param device: "cpu" or "gpu"
        :param num_epochs: number of epochs to run, will also set `num_epochs` in the config file.
            Note that this value is NOT HASHED, so that this number can be increased to continue the training.
        :param save_interval: save a checkpoint each n-th epoch
        :param keep_epochs: specify which checkpoints are kept, use None for the RETURNN default
            This will also limit the available output checkpoints to those defined. If you want to specify the keep
            behavior without this limitation, provide `cleanup_old_models/keep` in the post-config and use `None` here.
        :param time_rqmt:
        :param mem_rqmt:
        :param cpu_rqmt:
        :param distributed_launch_cmd: the command used to launch training jobs, only used if horovod_num_processes is not None
            Possible values: "mpirun": use mpirun, c.f. https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php
                             "torchrun": use torchrun, c.f. https://pytorch.org/docs/stable/elastic/run.html
        :param horovod_num_processes: If used without multi_node_slots, then single node, otherwise multi node.
        :param multi_node_slots: multi-node multi-GPU training. See Sisyphus rqmt documentation.
            Currently only with Horovod,
            and horovod_num_processes should be set as well, usually to the same value.
            See https://returnn.readthedocs.io/en/latest/advanced/multi_gpu.html.
        :param returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param returnn_root: file path to the RETURNN repository root folder
        """
        assert isinstance(returnn_config, ReturnnConfig)
        assert distributed_launch_cmd in ["mpirun", "torchrun"]
        self.check_blacklisted_parameters(returnn_config)
        kwargs = locals()
        del kwargs["self"]

        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)
        self.distributed_launch_cmd = distributed_launch_cmd
        self.horovod_num_processes = horovod_num_processes
        self.multi_node_slots = multi_node_slots
        self.returnn_config = ReturnnTrainingJob.create_returnn_config(**kwargs)

        stored_epochs = list(range(save_interval, num_epochs, save_interval)) + [num_epochs]
        if keep_epochs is None:
            self.keep_epochs = set(stored_epochs)
        else:
            self.keep_epochs = set(keep_epochs)

        suffix = ".meta" if self.returnn_config.get("use_tensorflow", False) else ""

        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_learning_rates = self.output_path("learning_rates")
        self.out_model_dir = self.output_path("models", directory=True)
        if self.returnn_config.get("use_tensorflow", False) or self.returnn_config.get("backend", None) == "tensorflow":
            self.out_checkpoints = {
                k: Checkpoint(index_path)
                for k in stored_epochs
                if k in self.keep_epochs
                for index_path in [self.output_path("models/epoch.%.3d.index" % k)]
            }

            # Deprecated, remove when possible
            self.out_models = {
                k: ReturnnModel(
                    self.out_returnn_config_file,
                    self.output_path("models/epoch.%.3d%s" % (k, suffix)),
                    k,
                )
                for k in stored_epochs
                if k in self.keep_epochs
            }
        elif self.returnn_config.get("backend", None) == "torch":
            self.out_checkpoints = {
                k: PtCheckpoint(pt_path)
                for k in stored_epochs
                if k in self.keep_epochs
                for pt_path in [self.output_path("models/epoch.%.3d.pt" % k)]
            }
            self.out_models = None
        else:
            raise ValueError("'backend' not specified in config")

        self.out_plot_se = self.output_path("score_and_error.png")
        self.out_plot_lr = self.output_path("learning_rate.png")

        self.returnn_config.post_config["model"] = os.path.join(self.out_model_dir.get_path(), "epoch")

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

        if self.multi_node_slots:
            assert self.horovod_num_processes, "multi_node_slots only supported together with Horovod currently"
            assert self.horovod_num_processes >= self.multi_node_slots
            assert self.horovod_num_processes % self.multi_node_slots == 0
            self.rqmt["multi_node_slots"] = self.multi_node_slots

        if (self.horovod_num_processes or 1) > (self.multi_node_slots or 1):
            assert self.horovod_num_processes % (self.multi_node_slots or 1) == 0
            self.rqmt["cpu"] *= self.horovod_num_processes // (self.multi_node_slots or 1)
            self.rqmt["gpu"] *= self.horovod_num_processes // (self.multi_node_slots or 1)
            self.rqmt["mem"] *= self.horovod_num_processes // (self.multi_node_slots or 1)

    def _get_run_cmd(self):
        run_cmd = [
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("rnn.py").get_path(),
            self.out_returnn_config_file.get_path(),
        ]

        if self.horovod_num_processes:
            if self.distributed_launch_cmd == "torchrun":
                # use torchrun to lauch DDP training when the backend is torch
                # Instead of using the torchrun binary, directly execute the corresponding Python module
                # and directly use the correct Python environment.
                prefix = [self.returnn_python_exe.get_path(), "-mtorch.distributed.run"]
                if (self.multi_node_slots or 1) == 1:
                    prefix += ["--standalone"]
                prefix += [
                    f"--nnodes={self.multi_node_slots or 1}",
                    f"--nproc-per-node={self.horovod_num_processes}",
                ]
                run_cmd = prefix + run_cmd[1:]
            elif self.distributed_launch_cmd == "mpirun":
                # Normally, if the engine (e.g. SGE or Slurm) is configured correctly,
                # it automatically provides the information on multiple nodes to mpirun,
                # so it is not needed to explicitly pass on any hostnames here.
                run_cmd = [
                    "mpirun",
                    "-np",
                    str(self.horovod_num_processes),
                    "-bind-to",
                    "none",
                    "-map-by",
                    "slot",
                    "-mca",
                    "pml",
                    "ob1",
                    "-mca",
                    "btl",
                    "^openib",
                    "--report-bindings",
                ] + run_cmd
            else:
                raise ValueError(f"invalid distributed_launch_cmd {self.distributed_launch_cmd!r}")

        return run_cmd

    def info(self):
        def try_load_lr_log(file_path: str) -> Optional[dict]:
            # Used in parsing the learning rates
            @dataclass
            class EpochData:
                learningRate: float
                error: Dict[str, float]

            try:
                with open(file_path, "rt") as file:
                    return eval(
                        file.read().strip(),
                        {"EpochData": EpochData, "nan": float("nan"), "inf": float("inf"), "np": np},
                    )
            except FileExistsError:
                return None
            except FileNotFoundError:
                return None

        lr_file = os.path.join(
            self._sis_path(gs.JOB_WORK_DIR),
            self.returnn_config.get("learning_rate_file", "learning_rates"),
        )
        epochs = try_load_lr_log(lr_file)

        if epochs is None:
            return None

        if not isinstance(epochs, dict):
            raise TypeError(f"parsed learning rates must be a Dict[int, EpochData] but found {type(epochs)}")

        available_epochs = {ep: data for ep, data in epochs.items() if len(data.error) > 0}

        max_available_ep = max(available_epochs) if len(available_epochs) > 0 else 0
        max_ep = max(self.out_checkpoints)

        return f"ep {max_available_ep}/{max_ep}"

    def path_available(self, path):
        # if job is finished the path is available
        res = super().path_available(path)
        if res:
            return res

        # learning rate files are only available at the end
        if path == self.out_learning_rates:
            return super().path_available(path)

        # maybe the file already exists
        res = os.path.exists(path.get_path())
        if res:
            return res

        # maybe the model is just a pretrain model
        file = os.path.basename(path.get_path())
        directory = os.path.dirname(path.get_path())
        if file.startswith("epoch."):
            segments = file.split(".")
            pretrain_file = ".".join([segments[0], "pretrain", segments[1]])
            pretrain_path = os.path.join(directory, pretrain_file)
            return os.path.exists(pretrain_path)

        return False

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)
        yield Task("plot", resume="plot", mini_task=True)

    def create_files(self):
        self.returnn_config.write(self.out_returnn_config_file.get_path())

        util.create_executable("rnn.sh", self._get_run_cmd())

    @staticmethod
    def _relink(src, dst):
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)

    def run(self):
        if self.multi_node_slots:
            # Some useful debugging, specifically for SGE parallel environment (PE).
            if "PE_HOSTFILE" in os.environ:
                print("PE_HOSTFILE =", os.environ["PE_HOSTFILE"])
                if os.environ["PE_HOSTFILE"]:
                    try:
                        print("Content:")
                        with open(os.environ["PE_HOSTFILE"]) as f:
                            print(f.read())
                    except Exception as exc:
                        print("Cannot read:", exc)
            sys.stdout.flush()

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.rqmt["cpu"])
        env["MKL_NUM_THREADS"] = str(self.rqmt["cpu"])
        sp.check_call(self._get_run_cmd(), env=env)

        lrf = self.returnn_config.get("learning_rate_file", "learning_rates")
        self._relink(lrf, self.out_learning_rates.get_path())

    def plot(self):
        def EpochData(learningRate, error):
            return {"learning_rate": learningRate, "error": error}

        with open(self.out_learning_rates.get_path(), "rt") as f:
            text = f.read()

        data = eval(text, {"EpochData": EpochData, "nan": float("nan"), "inf": float("inf"), "np": np})

        epochs = list(sorted(data.keys()))
        train_score_keys = [k for k in data[epochs[0]]["error"] if k.startswith("train_score")]
        dev_score_keys = [k for k in data[epochs[0]]["error"] if k.startswith("dev_score")]
        dev_error_keys = [k for k in data[epochs[0]]["error"] if k.startswith("dev_error")]

        train_scores = [
            [(epoch, data[epoch]["error"][tsk]) for epoch in epochs if tsk in data[epoch]["error"]]
            for tsk in train_score_keys
        ]
        dev_scores = [
            [(epoch, data[epoch]["error"][dsk]) for epoch in epochs if dsk in data[epoch]["error"]]
            for dsk in dev_score_keys
        ]
        dev_errors = [
            [(epoch, data[epoch]["error"][dek]) for epoch in epochs if dek in data[epoch]["error"]]
            for dek in dev_error_keys
        ]
        learing_rates = [data[epoch]["learning_rate"] for epoch in epochs]

        colors = ["#2A4D6E", "#AA3C39", "#93A537"]  # blue red yellowgreen

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        for ts in train_scores:
            ax1.plot([d[0] for d in ts], [d[1] for d in ts], "o-", color=colors[0])
        for ds in dev_scores:
            ax1.plot([d[0] for d in ds], [d[1] for d in ds], "o-", color=colors[1])
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("scores", color=colors[0])
        for tl in ax1.get_yticklabels():
            tl.set_color(colors[0])

        if len(dev_errors) > 0 and any(len(de) > 0 for de in dev_errors):
            ax2 = ax1.twinx()
            ax2.set_ylabel("dev error", color=colors[2])
            for de in dev_errors:
                ax2.plot([d[0] for d in de], [d[1] for d in de], "o-", color=colors[2])
            for tl in ax2.get_yticklabels():
                tl.set_color(colors[2])

        fig.savefig(fname=self.out_plot_se.get_path())

        fig, ax1 = plt.subplots()
        ax1.semilogy(epochs, learing_rates, "ro-")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("learning_rate")

        fig.savefig(fname=self.out_plot_lr.get_path())

    @classmethod
    def create_returnn_config(
        cls,
        returnn_config,
        log_verbosity,
        device,
        num_epochs,
        save_interval,
        keep_epochs,
        horovod_num_processes,
        **kwargs,
    ):
        assert device in ["gpu", "cpu"]

        res = copy.deepcopy(returnn_config)

        config = {
            "task": "train",
            "target": "classes",
            "learning_rate_file": "learning_rates",
        }

        post_config = {
            "device": device,
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
            "num_epochs": num_epochs,
            "save_interval": save_interval,
        }

        if horovod_num_processes is not None:
            config["use_horovod"] = True

        config.update(copy.deepcopy(returnn_config.config))
        if returnn_config.post_config is not None:
            post_config.update(copy.deepcopy(returnn_config.post_config))

        if keep_epochs is not None:
            if not "cleanup_old_models" in post_config or isinstance(post_config["cleanup_old_models"], bool):
                assert (
                    post_config.get("cleanup_old_models", True) == True
                ), "'cleanup_old_models' can not be False if 'keep_epochs' is specified"
                post_config["cleanup_old_models"] = {"keep": keep_epochs}
            elif isinstance(post_config["cleanup_old_models"], dict):
                assert (
                    "keep" not in post_config["cleanup_old_models"]
                ), "you can only provide either 'keep_epochs' or 'cleanup_old_models/keep', but not both"
                post_config["cleanup_old_models"]["keep"] = keep_epochs
            else:
                assert False, "invalid type of cleanup_old_models: %s" % type(post_config["cleanup_old_models"])

        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    def check_blacklisted_parameters(self, returnn_config):
        """
        Check for parameters that should not be set in the config directly

        :param ReturnnConfig returnn_config:
        :return:
        """
        blacklisted_keys = [
            "log_verbosity",
            "device",
            "num_epochs",
            "save_interval",
            "keep_epochs",
        ]
        for key in blacklisted_keys:
            assert returnn_config.get(key) is None, (
                "please define %s only as parameter to ReturnnTrainingJob directly" % key
            )

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config": ReturnnTrainingJob.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        if kwargs["horovod_num_processes"] is not None:
            d["horovod_num_processes"] = kwargs["horovod_num_processes"]

        if kwargs["multi_node_slots"] is not None:
            d["multi_node_slots"] = kwargs["multi_node_slots"]

        return super().hash(d)


class ReturnnTrainingFromFileJob(Job):
    """
    The Job allows to directly execute returnn config files. The config files have to have the line
    `ext_model = config.value("ext_model", None)` and `model = ext_model` to correctly set the model path

    If the learning rate file should be available, add
    `ext_learning_rate_file = config.value("ext_learning_rate_file", None)` and
    `learning_rate_file = ext_learning_rate_file`

    Other externally controllable parameters may also defined in the same way, and can be set by providing the parameter
    value in the parameter_dict. The "ext_" prefix is used for naming convention only, but should be used for all
    external parameters to clearly mark them instead of simply overwriting any normal parameter.

    Also make sure that task="train" is set.
    """

    def __init__(
        self,
        returnn_config_file,
        parameter_dict,
        time_rqmt=4,
        mem_rqmt=4,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param tk.Path|str returnn_config_file: a returnn training config file
        :param dict parameter_dict: provide external parameters to the rnn.py call
        :param int|str time_rqmt:
        :param int|str mem_rqmt:
        :param Optional[Path] returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Optional[Path] returnn_root: file path to the RETURNN repository root folder
        """

        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)
        self.returnn_config_file_in = returnn_config_file
        self.parameter_dict = parameter_dict
        if self.parameter_dict is None:
            self.parameter_dict = {}

        self.returnn_config_file = self.output_path("returnn.config")

        self.rqmt = {"gpu": 1, "cpu": 2, "mem": mem_rqmt, "time": time_rqmt}

        self.learning_rates = self.output_path("learning_rates")
        self.model_dir = self.output_path("models", directory=True)

        self.parameter_dict["ext_model"] = self.model_dir.get() + "/epoch"
        self.parameter_dict["ext_learning_rate_file"] = self.learning_rates.get()

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def path_available(self, path):
        # if job is finised the path is available
        res = super().path_available(path)
        if res:
            return res

        # learning rate files are only available at the end
        if path == self.learning_rates:
            return super().path_available(path)

        # maybe the file already exists
        res = os.path.exists(path.get_path())
        if res:
            return res

        # maybe the model is just a pretrain model
        file = os.path.basename(path.get_path())
        directory = os.path.dirname(path.get_path())
        if file.startswith("epoch."):
            segments = file.split(".")
            pretrain_file = ".".join([segments[0], "pretrain", segments[1]])
            pretrain_path = os.path.join(directory, pretrain_file)
            return os.path.exists(pretrain_path)

        return False

    def get_parameter_list(self):
        parameter_list = []
        for k, v in sorted(self.parameter_dict.items()):
            if isinstance(v, (tk.Variable, tk.Path)):
                v = v.get()
            elif isinstance(v, (list, dict, tuple)):
                v = '"%s"' % str(v).replace(" ", "")

            if isinstance(v, (float, int)) and v < 0:
                v = "+" + str(v)
            else:
                v = str(v)

            parameter_list.append("++%s" % k)
            parameter_list.append(v)

        return parameter_list

    def create_files(self):
        # returnn
        shutil.copy(
            tk.uncached_path(self.returnn_config_file_in),
            self.returnn_config_file.get_path(),
        )

        parameter_list = self.get_parameter_list()
        cmd = [
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("rnn.py").get_path(),
            self.returnn_config_file.get_path(),
        ] + parameter_list

        util.create_executable("rnn.sh", cmd)

    def run(self):
        sp.check_call(["./rnn.sh"])

    @classmethod
    def hash(cls, kwargs):

        d = {
            "returnn_config_file": kwargs["returnn_config_file"],
            "parameter_dict": kwargs["parameter_dict"],
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return super().hash(d)


class GetBestEpochJob(Job):
    """
    Provided a RETURNN model directory and a score key, finds the best epoch.
    The sorting is lower=better, so to access the model with the highest values use negative index values (e.g. -1 for
    the model with the highest score, error or "loss")

    """

    def __init__(self, model_dir: tk.Path, learning_rates: tk.Path, key: str, index: int = 0):
        """
        :param model_dir: model_dir output from a RETURNNTrainingJob
        :param learning_rates: learning_rates output from a RETURNNTrainingJob
        :param key: a key from the learning rate file that is used to sort the models,
            e.g. "dev_score_output/output_prob"
        :param index: index of the sorted list to access, 0 for the lowest, -1 for the highest score/error/loss
        """
        self.model_dir = model_dir
        self.learning_rates = learning_rates
        self.index = index
        self.out_epoch = self.output_var("epoch")
        self.key = key

    def run(self):
        # this has to be defined in order for "eval" to work
        def EpochData(learningRate, error):
            return {"learning_rate": learningRate, "error": error}

        with open(self.learning_rates.get_path(), "rt") as f:
            text = f.read()

        data = eval(text, {"EpochData": EpochData, "nan": float("nan"), "inf": float("inf"), "np": np})

        epochs = list(sorted(data.keys()))

        # some epochs might not have all keys, we require that the last entry has the key, other epochs which
        # do not have it might be ignored
        available_keys = data[epochs[-1]]["error"]
        if self.key not in available_keys:
            raise KeyError(
                f"{self.key} is not available in the provided learning_rates file {self.learning_rates.get_path()}"
            )

        scores = [(epoch, data[epoch]["error"][self.key]) for epoch in epochs if self.key in data[epoch]["error"]]
        sorted_scores = list(sorted(scores, key=lambda x: x[1]))

        self.out_epoch.set(sorted_scores[self.index][0])

    def tasks(self):
        yield Task("run", mini_task=True)


class GetBestTFCheckpointJob(GetBestEpochJob):
    """
    Returns the best checkpoint given a training model dir and a learning-rates file
    The best checkpoint will be HARD-linked if possible, so that no space is wasted but also the model not
    deleted in case that the training folder is removed.
    """

    def __init__(self, model_dir: tk.Path, learning_rates: tk.Path, key: str, index: int = 0):
        """

        :param Path model_dir: model_dir output from a RETURNNTrainingJob
        :param Path learning_rates: learning_rates output from a RETURNNTrainingJob
        :param str key: a key from the learning rate file that is used to sort the models
            e.g. "dev_score_output/output_prob"
        :param int index: index of the sorted list to access, 0 for the lowest, -1 for the highest score
        """
        super().__init__(model_dir, learning_rates, key, index)
        self._out_model_dir = self.output_path("model", directory=True)

        # Note: checkpoint.index (without epoch number) is only a symlink which is possibly resolved by RETURNN
        # See also https://github.com/rwth-i6/returnn/issues/1194 for the current behavior
        self.out_checkpoint = Checkpoint(self.output_path("model/checkpoint.index"))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        super().run()

        try:
            os.link(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get()),
                os.path.join(
                    self._out_model_dir.get_path(),
                    "epoch.%.3d.index" % self.out_epoch.get(),
                ),
            )
            os.link(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
                os.path.join(
                    self._out_model_dir.get_path(),
                    "epoch.%.3d.meta" % self.out_epoch.get(),
                ),
            )
            os.link(
                os.path.join(
                    self.model_dir.get_path(),
                    "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get(),
                ),
                os.path.join(
                    self._out_model_dir.get_path(),
                    "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get(),
                ),
            )
        except OSError:
            # the hardlink will fail when there was an imported job on a different filesystem,
            # thus do a copy instead then
            shutil.copy(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get()),
                os.path.join(
                    self._out_model_dir.get_path(),
                    "epoch.%.3d.index" % self.out_epoch.get(),
                ),
            )
            shutil.copy(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
                os.path.join(
                    self._out_model_dir.get_path(),
                    "epoch.%.3d.meta" % self.out_epoch.get(),
                ),
            )
            shutil.copy(
                os.path.join(
                    self.model_dir.get_path(),
                    "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get(),
                ),
                os.path.join(
                    self._out_model_dir.get_path(),
                    "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get(),
                ),
            )

        os.symlink(
            os.path.join(
                self._out_model_dir.get_path(),
                "epoch.%.3d.index" % self.out_epoch.get(),
            ),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.index"),
        )
        os.symlink(
            os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.meta"),
        )
        os.symlink(
            os.path.join(
                self._out_model_dir.get_path(),
                "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get(),
            ),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.data-00000-of-00001"),
        )


class GetBestPtCheckpointJob(GetBestEpochJob):
    """
    Analog to GetBestTFCheckpointJob, just for torch checkpoints.
    """

    def __init__(self, model_dir: tk.Path, learning_rates: tk.Path, key: str, index: int = 0):
        """

        :param Path model_dir: model_dir output from a ReturnnTrainingJob
        :param Path learning_rates: learning_rates output from a ReturnnTrainingJob
        :param str key: a key from the learning rate file that is used to sort the models
            e.g. "dev_score_output/output_prob"
        :param int index: index of the sorted list to access, 0 for the lowest, -1 for the highest score
        """
        super().__init__(model_dir, learning_rates, key, index)
        self.out_checkpoint = PtCheckpoint(self.output_path("checkpoint.pt"))

    def run(self):
        super().run()

        try:
            os.link(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.pt" % self.out_epoch.get()),
                self.out_checkpoint.path,
            )
        except OSError:
            # the hardlink will fail when there was an imported job on a different filesystem,
            # thus do a copy instead then
            shutil.copy(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.pt" % self.out_epoch.get()),
                self.out_checkpoint.path,
            )


class AverageTFCheckpointsJob(Job):
    """
    Compute the average of multiple specified Tensorflow checkpoints using the tf_avg_checkpoints script from Returnn
    """

    def __init__(
        self,
        model_dir: tk.Path,
        epochs: List[Union[int, tk.Variable]],
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
    ):
        """

        :param model_dir: model dir from `ReturnnTrainingJob`
        :param epochs: manually specified epochs or `out_epoch` from `GetBestEpochJob`
        :param returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param returnn_root: file path to the RETURNN repository root folder
        """
        self.model_dir = model_dir
        self.epochs = epochs
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root

        self._out_model_dir = self.output_path("model", directory=True)
        self.out_checkpoint = Checkpoint(self.output_path("model/average.index"))

        self.rqmt = {"cpu": 1, "time": 0.5, "mem": 2 * len(epochs)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        epochs = util.instanciate_delayed(self.epochs)
        max_epoch = max(epochs)

        # we are writing a checkpoint with the maximum epoch index in the file name because Returnn
        # resolves symlinks and reads the name to determine the "checkpoint epoch"
        out_path = os.path.join(self._out_model_dir.get_path(), "epoch.%03d" % max_epoch)
        args = [
            self.returnn_python_exe.get_path(),
            os.path.join(self.returnn_root.get_path(), "tools/tf_avg_checkpoints.py"),
            "--checkpoints",
            ",".join(["%03d" % epoch for epoch in epochs]),
            "--prefix",
            self.model_dir.get_path() + "/epoch.",
            "--output_path",
            out_path,
        ]
        os.symlink(out_path + ".index", self.out_checkpoint.index_path.get_path())
        os.symlink(out_path + ".meta", self.out_checkpoint.ckpt_path + ".meta")
        os.symlink(
            out_path + ".data-00000-of-00001",
            self.out_checkpoint.ckpt_path + ".data-00000-of-00001",
        )

        # The env override is needed if this job is run locally on a node with a GPU installed
        sp.check_call(args, env={"CUDA_VISIBLE_DEVICES": ""})


class AverageTorchCheckpointsJob(Job):
    """
    average Torch model checkpoints
    """

    def __init__(
        self,
        *,
        checkpoints: Sequence[Union[tk.Path, PtCheckpoint]],
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
    ):
        """
        :param checkpoints: input checkpoints
        :param returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param returnn_root: file path to the RETURNN repository root folder
        """
        self.checkpoints = [ckpt if isinstance(ckpt, PtCheckpoint) else PtCheckpoint(ckpt) for ckpt in checkpoints]
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root

        self.out_checkpoint = PtCheckpoint(self.output_path("model/average.pt"))

        self.rqmt = {"cpu": 1, "time": 0.5, "mem": 5}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        os.makedirs(os.path.dirname(self.out_checkpoint.path.get_path()), exist_ok=True)
        args = [
            self.returnn_python_exe.get_path(),
            os.path.join(self.returnn_root.get_path(), "tools/torch_avg_checkpoints.py"),
            "--checkpoints",
            *[ckpt.path.get_path() for ckpt in self.checkpoints],
            "--output_path",
            self.out_checkpoint.path.get_path(),
        ]
        sp.check_call(args)
