__all__ = ["ReturnnCalculatePerplexityJob"]

import shutil
import subprocess as sp
from typing import Union

from sisyphus import Job, Task, setup_path, tk

import i6_core.util as util

from .config import ReturnnConfig
from .training import PtCheckpoint, Checkpoint

Path = setup_path(__package__)


class ReturnnCalculatePerplexityJob(Job):
    """
    Calculates the perplexity of a language model trained in RETURNN
    on an evaluation data set
    """

    def __init__(
        self,
        returnn_config: ReturnnConfig,
        returnn_model: Union[PtCheckpoint, Checkpoint],
        eval_dataset: tk.Path,
        *,
        log_verbosity: int = 3,
        returnn_root: tk.Path,
        returnn_python_exe: tk.Path,
    ):
        returnn_config.config.pop("train")
        returnn_config.config.pop("dev")
        returnn_config.config["eval_datasets"] = {"eval": eval_dataset}

        # TODO verify paths
        if isinstance(returnn_model, PtCheckpoint):
            model_path = returnn_model.path
            self.add_input(returnn_model.path)
        elif isinstance(returnn_model, Checkpoint):
            model_path = returnn_model.index_path
            self.add_input(returnn_model.index_path)
        else:
            raise NotImplementedError(f"returnn model has unknown type: {type(returnn_model)}")

        returnn_config.config["model"] = model_path

        returnn_config.post_config["log_verbosity"] = log_verbosity

        self.returnn_config = returnn_config

        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root

        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_returnn_log = self.output_path("returnn.log")
        self.out_perplexities = self.output_var("ppl_score")

        self.rqmt = {"gpu": 0, "cpu": 2, "mem": 4, "time": 4}

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)
        yield Task("gather", mini_task=True)

    def _get_run_cmd(self):
        run_cmd = [
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("rnn.py").get_path(),
            self.out_returnn_config_file.get_path(),
            "++task eval",
        ]
        return run_cmd

    def create_files(self):
        self.returnn_config.write(self.out_returnn_config_file.get_path())

        util.create_executable("rnn.sh", self._get_run_cmd())

    def run(self):
        sp.check_call(self._get_run_cmd())

        shutil.move("returnn_log", self.out_returnn_log.get_path())

    def gather(self):
        for data_key in self.out_perplexities.keys():
            print(data_key)

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["log_verbosity"]
        return super().hash(parsed_args)
