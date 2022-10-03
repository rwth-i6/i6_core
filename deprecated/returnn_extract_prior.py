import copy
import numpy as np
import os
import subprocess as sp

from sisyphus import Job, Task, gs, tk

from i6_core import util
from i6_core.returnn import ReturnnConfig


class ReturnnComputePriorJob(Job):
    """
    Given a model checkpoint, run compute_prior task with RETURNN
    """

    def __init__(
        self,
        model_checkpoint,
        returnn_config,
        prior_data=None,
        *,
        log_verbosity=3,
        device="gpu",
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param dict[str]|None prior_data: dataset used to compute prior (None = use one train epoch)
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :param float|int time_rqmt: job time requirement in hours
        :param float|int mem_rqmt: job memory requirement in GB
        :param float|int cpu_rqmt: job cpu requirement in GB
        :param tk.Path|str|None returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param tk.Path|str|None returnn_root: path to the RETURNN src folder
        """
        assert isinstance(returnn_config, ReturnnConfig)
        kwargs = locals()
        del kwargs["self"]

        self.model_checkpoint = model_checkpoint

        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )

        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.returnn_config = ReturnnComputePriorJob.create_returnn_config(**kwargs)

        self.out_returnn_config_file = self.output_path("returnn.config")

        self.out_prior_txt_file = self.output_path("prior.txt")
        self.out_prior_xml_file = self.output_path("prior.xml")
        self.out_prior_png_file = self.output_path("prior.png")

        self.returnn_config.post_config["output_file"] = self.out_prior_txt_file

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)
        yield Task("plot", resume="plot", mini_task=True)

    def create_files(self):
        config = self.returnn_config
        config.write(self.out_returnn_config_file.get_path())

        cmd = self._get_run_cmd()
        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        assert os.path.exists(
            tk.uncached_path(self.model_checkpoint.index_path)
        ), "Provided model does not exists: %s" % str(self.model_checkpoint)

    def run(self):
        cmd = self._get_run_cmd()
        sp.check_call(cmd)

        with open(self.out_prior_txt_file.get_path(), "rt") as f:
            merged_scores = np.loadtxt(f, delimiter=" ")

        with open(self.out_prior_xml_file.get_path(), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n'
                % len(merged_scores)
            )
            f.write(" ".join("%.20e" % s for s in merged_scores) + "\n")
            f.write("</vector-f32>")

    def plot(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(self.out_prior_txt_file.get_path(), "rt") as f:
            merged_scores = np.loadtxt(f, delimiter=" ")

        xdata = range(len(merged_scores))
        plt.semilogy(xdata, np.exp(merged_scores))
        plt.xlabel("emission idx")
        plt.ylabel("prior")
        plt.grid(True)
        plt.savefig(self.out_prior_png_file.get_path())

    def _get_run_cmd(self):
        return [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]

    @classmethod
    def create_returnn_config(
        cls,
        model_checkpoint,
        returnn_config,
        prior_data,
        log_verbosity,
        device,
        **kwargs,
    ):
        """
        Creates compute_prior RETURNN config
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param dict[str]|None prior_data: dataset used to compute prior (None = use one train epoch)
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :rtype: ReturnnConfig
        """
        assert device in ["gpu", "cpu"]
        original_config = returnn_config.config
        assert "network" in original_config

        config = copy.deepcopy(original_config)
        config["load"] = model_checkpoint.ckpt_path
        config["task"] = "compute_priors"

        if prior_data is not None:
            config["train"] = prior_data

        post_config = {
            "device": device,
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
        }

        post_config.update(copy.deepcopy(returnn_config.post_config))

        res = copy.deepcopy(returnn_config)
        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config": ReturnnComputePriorJob.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }
        return super().hash(d)
