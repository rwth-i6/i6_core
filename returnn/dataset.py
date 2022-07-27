__all__ = ["ExtractDatasetMeanStddevJob"]

from sisyphus import *

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import numpy

from i6_core.returnn import ReturnnConfig, ReturnnForwardJob, Checkpoint
from i6_core.util import create_executable


class ExtractDatasetMeanStddevJob(Job):
    """
    Runs the RETURNN tool dump-dataset with statistic extraction.
    Collects mean and std-var for each feature as file and in total as sisyphus var.

    Outputs:

    Variable out_mean: a global mean over all sequences and features
    Variable out_std_dev: a global std-dev over all sequences and features

    Path out_mean_file: a text file with #feature entries for the mean
    Path out_std_dev_file: a text file with #features entries for the standard deviation
    """

    def __init__(self, returnn_config, returnn_python_exe=None, returnn_root=None):
        """

        :param ReturnnConfig returnn_config:
        :param Path|str|None returnn_python_exe:
        :param Path|str|None returnn_root:
        """

        self.returnn_config = returnn_config
        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.out_mean = self.output_var("mean_var")
        self.out_std_dev = self.output_var("std_dev_var")
        self.out_mean_file = self.output_path("mean")
        self.out_std_dev_file = self.output_path("std_dev")

        self.rqmt = {"cpu": 2, "mem": 4, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        self.returnn_config.write("returnn.config")

        command = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "tools/dump-dataset.py"),
            "returnn.config",
            "--endseq -1",
            "--stats",
            "--dump_stats stats",
        ]

        create_executable("rnn.sh", command)
        subprocess.check_call(["./rnn.sh"])

        shutil.move("stats.mean.txt", self.out_mean_file.get_path())
        shutil.move("stats.std_dev.txt", self.out_std_dev_file.get_path())

        total_mean = 0
        total_var = 0

        with open(self.out_mean_file.get_path()) as mean_file, open(
            self.out_std_dev_file.get_path()
        ) as std_dev_file:

            # compute the total mean and std-dev in an iterative way
            for i, (mean, std_dev) in enumerate(zip(mean_file, std_dev_file)):
                mean = float(mean)
                var = float(std_dev.strip()) ** 2
                mean_variance = (total_mean - mean) ** 2
                adjusted_mean_variance = mean_variance * i / (i + 1)
                total_var = (total_var * i + var + adjusted_mean_variance) / (i + 1)
                total_mean = (total_mean * i + mean) / (i + 1)

            self.out_mean.set(total_mean)
            self.out_std_dev.set(numpy.sqrt(total_var))


def get_returnn_length_hdfs(
    dataset_dict: Dict[str, Any],
    extern_data: Dict[str, Any],
    dataset_keys: List,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    job_alias: Optional[str] = None,
    mem_rqmt: float = 4,
    time_rqmt: float = 4,
) -> Dict[str, tk.Path]:
    """
    Uses returnn to extract the length of sequences in the datasets after feature extraction, returns a separate hdf
    file per dataset key
    :param dataset_dict: RETURNN config dict for dataset
    :param extern_data: extern data dict matching the dataset
    :param dataset_keys: keys of the datastreams
    :param returnn_exe:
    :param returnn_root:
    :param job_alias: full alias of the forward job
    :param mem_rqmt: job memory requirement in GB
    :param time_rqmt: job time requirement in hours
    :return:
    """
    post_config = {"use_tensorflow": True}
    config = {
        "eval": dataset_dict,
        "network": {
            "output": {
                "class": "length",
                "axis": "T",
                "from": f"data:{dataset_keys[0]}",
            },
        },
        "extern_data": extern_data,
        "max_seqs": 50
    }
    for idx, key in enumerate(dataset_keys[1:]):
        config["network"][f"length_{idx}"] = {
            "class": "length",
            "axis": "T",
            "from": f"data:{key}",
        }
        config["network"][f"dump_{idx}"] = {
            "class": "hdf_dump",
            "filename": f"{key}.hdf",
            "from": f"length_data_{idx}",
        }

    hdf_outputs = [k + ".hdf" for k in dataset_keys]
    config = ReturnnConfig(config=config, post_config=post_config)
    forward_job = ReturnnForwardJob(
        model_checkpoint=None,
        returnn_config=config,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        hdf_outputs=hdf_outputs,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
    )
    if job_alias is not None:
        forward_job.add_alias(job_alias)
    # remove .hdf extension and map default output to first dataset key
    hdf_dict = {k[:-4]: v for k, v in forward_job.out_hdf_files.items()}
    hdf_dict[dataset_keys[0]] = hdf_dict["output"]
    del hdf_dict["output"]
    return hdf_dict
