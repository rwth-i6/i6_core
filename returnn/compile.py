__all__ = ["CompileTFGraphJob"]

from sisyphus import *

Path = setup_path(__package__)

import copy
import os
import subprocess as sp

from .config import ReturnnConfig


class CompileTFGraphJob(Job):
    def __init__(
        self,
        returnn_config,
        train=0,
        eval=0,
        search=0,
        verbosity=4,
        summaries_tensor_name=None,
        output_format="meta",
        returnn_python_exe=None,
        returnn_root=None,
    ):
        self.returnn_config = returnn_config
        self.train = train
        self.eval = eval
        self.search = search
        self.verbosity = verbosity
        self.summaries_tensor_name = summaries_tensor_name
        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.graph = self.output_path("graph.%s" % output_format)
        self.model_params = self.output_var("model_params.pickle", pickle=True)
        self.state_vars = self.output_var("state_vars.pickle", pickle=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        returnn_config_path = self.returnn_config
        if isinstance(self.returnn_config, ReturnnConfig):
            returnn_config_path = "returnn.config"
            self.returnn_config.write(returnn_config_path)

        args = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(
                tk.uncached_path(self.returnn_root), "tools/compile_tf_graph.py"
            ),
            returnn_config_path,
            "--train=%d" % self.train,
            "--eval=%d" % self.eval,
            "--search=%d" % self.search,
            "--verbosity=%d" % self.verbosity,
            "--output_file=%s" % self.graph.get_path(),
            "--output_file_model_params_list=model_params",
            "--output_file_state_vars_list=state_vars",
        ]
        if self.summaries_tensor_name is not None:
            args.append("--summaries_tensor_name=%s" % self.summaries_tensor_name)

        sp.check_call(args)

        with open("model_params", "rt") as input:
            lines = [l.strip() for l in input if len(l.strip()) > 0]
            self.model_params.set(lines)
        with open("state_vars", "rt") as input:
            lines = [l.strip() for l in input if len(l.strip()) > 0]
            self.state_vars.set(lines)

    @classmethod
    def hash(cls, kwargs):
        c = copy.copy(kwargs)
        del c["verbosity"]
        return super().hash(c)
