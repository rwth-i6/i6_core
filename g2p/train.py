__all__ = ["TrainG2PModelJob"]

import os
import subprocess as sp

from sisyphus import *

Path = setup_path(__package__)


class TrainG2PModelJob(Job):
    def __init__(
        self,
        train_lexicon,
        num_ramp_ups=4,
        min_iter=1,
        max_iter=60,
        devel="5%",
        size_constrains="0,1,0,1",
        extra_args=None,
        g2p_path=None,
        g2p_python=None,
    ):
        if extra_args is None:
            extra_args = []
        if g2p_path is None:
            g2p_path = tk.gs.G2P_PATH
        if g2p_python is None:
            g2p_python = tk.gs.G2P_PYTHON

        self.train_lexicon = train_lexicon
        self.num_ramp_ups = num_ramp_ups
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.devel = devel
        self.size_constrains = size_constrains
        self.extra_args = extra_args
        self.g2p_path = g2p_path
        self.g2p_python = g2p_python

        self.g2p_models = [
            self.output_path("model-%d" % idx) for idx in range(self.num_ramp_ups + 1)
        ]
        self.error_rates = [
            self.output_var("err-%d" % idx) for idx in range(self.num_ramp_ups + 1)
        ]
        self.best_model = self.output_path("model-best")
        self.best_error_rate = self.output_var("err-best")

        self.rqmt = {
            "time": max(0.5, (self.max_iter / 20.0) * (self.num_ramp_ups + 1)),
            "cpu": 1,
            "mem": 2,
        }

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        for idx in range(self.num_ramp_ups + 1):
            if os.path.exists(self.g2p_models[idx].get_path()):
                continue

            args = [
                self.g2p_python,
                self.g2p_path,
                "-e",
                "utf-8",
                "-i",
                str(self.min_iter),
                "-I",
                str(self.max_iter),
                "-d",
                self.devel,
                "-s",
                self.size_constrains,
                "-n",
                "tmp-model",
                "-S",
                "-t",
                tk.uncached_path(self.train_lexicon),
            ]
            if idx > 0:
                args += ["-r", "-m", self.g2p_models[idx - 1].get_path()]
            args += self.extra_args

            if os.path.exists("tmp-model"):
                os.unlink("tmp-model")

            with open("stdout.%d" % idx, "w") as out:
                sp.check_call(args, stdout=out)

            with open("stdout.%d" % idx, "rt") as log:
                for line in log:
                    if "total symbol errors" in line:
                        error_rate = float(line.split("(")[1].split("%")[0])
                        self.error_rates[idx].set(error_rate)

            os.rename("tmp-model", self.g2p_models[idx].get_path())

        best = min(
            ((idx, err_var.get()) for idx, err_var in enumerate(self.error_rates)),
            key=lambda t: t[1],
        )
        os.symlink("model-%d" % best[0], self.best_model.get_path())
        self.best_error_rate.set(best[1])
