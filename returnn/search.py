__all__ = ["ReturnnSearchFromFileJob", "ReturnnComputeWERJob"]

from sisyphus import *

Path = setup_path(__package__)

import copy
import logging
import os
import stat
import shutil
import subprocess as sp


class ReturnnSearchFromFileJob(Job):
    """
    Run search mode with a given RETURNN config.

    If the model path and epoch should be defined from the outside, please use
    `ext_model`for the model path and `ext_load_epoch` for the epoch in the config and parameter dict.

    As the existance of the model will be checked via `update`, it is possible to define checkpoints
    that do not exist yet, or checkpoint that can be automatically deleted during the training.

    """

    def __init__(
        self,
        returnn_config_file,
        parameter_dict=None,
        output_mode="py",
        default_model_name="epoch",
        time_rqmt=4,
        mem_rqmt=4,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param tk.Path returnn_config_file: a returnn training config file
        :param dict parameter_dict: provide external parameters to the rnn.py call
        :param str output_mode: "py" or "txt"
        :param float|int time_rqmt:
        :param float|int mem_rqmt:
        :param tk.Path|str returnn_python_exe: RETURNN python executable
        :param tk.Path|str returnn_root: RETURNN source root
        """

        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.returnn_config_file_in = returnn_config_file
        self.default_model_name = default_model_name
        self.parameter_dict = parameter_dict
        if self.parameter_dict is None:
            self.parameter_dict = {}

        self.returnn_config_file = self.output_path("returnn.config")

        self.rqmt = {"gpu": 1, "cpu": 2, "mem": mem_rqmt, "time": time_rqmt}

        assert output_mode in ["py", "txt"]
        self.search_results = self.output_path("search_results.%s" % output_mode)

        self.parameter_dict["search_output_file"] = tk.uncached_path(
            self.search_results
        )
        self.parameter_dict["search_output_file_format"] = output_mode

    def update(self):
        if (
            "ext_model" in self.parameter_dict
            and "ext_load_epoch" in self.parameter_dict
        ):
            epoch = self.parameter_dict["ext_load_epoch"]
            epoch = epoch.get() if isinstance(epoch, tk.Variable) else epoch
            model_dir = self.parameter_dict["ext_model"]
            if isinstance(model_dir, tk.Path):
                self.add_input(
                    Path(
                        tk.uncached_path(model_dir)
                        + "/%s.%03d.index" % (self.default_model_name, epoch),
                        creator=model_dir.creator,
                    )
                )
            else:
                self.add_input(
                    Path(
                        tk.uncached_path(model_dir)
                        + "/%s.%03d.index" % (self.default_model_name, epoch)
                    )
                )

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def get_parameter_list(self):
        parameter_list = []
        for k, v in sorted(self.parameter_dict.items()):
            if isinstance(v, tk.Variable):
                v = v.get()
            elif isinstance(v, tk.Path):
                v = tk.uncached_path(v)
            elif isinstance(v, (list, dict, tuple)):
                v = '"%s"' % str(v).replace(" ", "")

            if isinstance(v, (float, int)) and v < 0:
                v = "+" + str(v)
            else:
                v = str(v)

            if k == "ext_model" and not v.endswith("/%s" % self.default_model_name):
                v = v + "/%s" % self.default_model_name

            parameter_list += ["++%s" % k, v]

        return parameter_list

    def create_files(self):
        # returnn
        shutil.copy(
            tk.uncached_path(self.returnn_config_file_in),
            tk.uncached_path(self.returnn_config_file),
        )

        parameter_list = self.get_parameter_list()

        with open("rnn.sh", "wt") as f:
            f.write(
                "#!/usr/bin/env bash\n%s"
                % " ".join(
                    [
                        tk.uncached_path(self.returnn_python_exe),
                        os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
                        self.returnn_config_file.get_path(),
                    ]
                    + parameter_list
                )
            )
        os.chmod(
            "rnn.sh",
            stat.S_IRUSR
            | stat.S_IRGRP
            | stat.S_IROTH
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )

    def run(self):
        sp.check_call(["./rnn.sh"])

    @classmethod
    def hash(cls, kwargs):
        kwargs = copy.copy(kwargs)
        kwargs.pop("default_model_name")
        kwargs.pop("time_rqmt")
        kwargs.pop("mem_rqmt")
        return super().hash(kwargs)


class SearchBPEtoWordsJob(Job):
    """
    converts BPE tokens to words in the python format dict from the returnn search
    """

    def __init__(self, search_py_output):
        """

        :param Path search_py_output: a search output file from RETURNN in python format
        """
        self.search_py_output = search_py_output
        self.word_search_results = self.output_path("word_search_results.py")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        d = eval(open(tk.uncached_path(self.search_py_output), "r").read())
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(tk.uncached_path(self.word_search_results))
        with open(tk.uncached_path(self.word_search_results), "w") as out:
            out.write("{\n")
            for seq_tag, txt in sorted(d.items()):
                if "#" in seq_tag:
                    tag_split = seq_tag.split("/")
                    recording_name, segment_name = tag_split[2].split("#")
                    seq_tag = tag_split[0] + "/" + recording_name + "/" + segment_name
                out.write("%r: %r,\n" % (seq_tag, txt.replace("@@ ", "")))
            out.write("}\n")


class ReturnnComputeWERJob(Job):
    """
    Computes WER using the calculate-word-error-rate.py tool from RETURNN
    """

    def __init__(
        self, hypothesis, reference, returnn_python_exe=None, returnn_root=None
    ):
        """

        :param Path hypothesis: python-style search output from RETURNN
        :param Path reference: python-style text dictionary (use e.g. BlissExtractTextDictionary)
        :param str|Path returnn_python_exe: RETURNN python executable
        :param str|Path returnn_root: RETURNN source root
        """
        self.hypothesis = hypothesis
        self.reference = reference

        self.returnn_python_exe = (
            returnn_python_exe if returnn_python_exe else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = returnn_root if returnn_root else gs.RETURNN_ROOT

        self.wer = self.output_path("wer")

    def run(self):
        call = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(str(self.returnn_root), "tools/calculate-word-error-rate.py"),
            "--expect_full",
            "--hyps",
            tk.uncached_path(self.hypothesis),
            "--refs",
            tk.uncached_path(self.reference),
            "--out",
            tk.uncached_path(self.wer),
        ]
        logging.info("run %s" % " ".join(call))
        sp.check_call(call)

    def tasks(self):
        yield Task("run", mini_task=True)
