__all__ = ["CompileTFGraphJob", "CompileNativeOpJob", "TorchOnnxExportJob"]

from sisyphus import *

Path = setup_path(__package__)

import copy
import logging
import os
import shutil
import subprocess as sp
from typing import Optional, Sequence

import i6_core.util as util

from .config import ReturnnConfig
from .training import PtCheckpoint


class CompileTFGraphJob(Job):
    """
    This Job is a wrapper around the RETURNN tool compile_tf_graph.py

    """

    __sis_hash_exclude__ = {"device": None, "epoch": None, "rec_step_by_step": None, "rec_json_info": False}

    def __init__(
        self,
        returnn_config,
        train=0,
        eval=0,
        search=0,
        epoch=None,
        verbosity=4,
        device=None,
        summaries_tensor_name=None,
        output_format="meta",
        returnn_python_exe=None,
        returnn_root=None,
        rec_step_by_step=None,
        rec_json_info=False,
    ):
        """

        :param ReturnnConfig|Path|str returnn_config: Path to a RETURNN config file
        :param int train:
        :param int eval:
        :param int search:
        :param int|tk.Variable|None epoch: compile a specific epoch for networks that might change with every epoch
        :param int log_verbosity: RETURNN log verbosity from 1 (least verbose) to 5 (most verbose)
        :param str|None device: optimize graph for cpu or gpu. If `None`, defaults to cpu for current RETURNN.
            For any RETURNN version before `cd4bc382`, the behavior will depend on the `device` entry in the
            `returnn_conig`, or on the availability of a GPU on the execution host if not defined at all.
        :param summaries_tensor_name:
        :param str output_format: graph output format, one of ["pb", "pbtxt", "meta", "metatxt"]
        :param Optional[Path] returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Optional[Path] returnn_root: file path to the RETURNN repository root folder
        :param Optional[str] rec_step_by_step: name of rec layer for step-by-step graph
        :param bool rec_json_info: whether to enable rec json info for step-by-step graph compilation
        """
        self.returnn_config = returnn_config
        self.train = train
        self.eval = eval
        self.search = search
        self.epoch = epoch
        self.verbosity = verbosity
        self.device = device
        self.summaries_tensor_name = summaries_tensor_name
        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)
        self.rec_step_by_step = rec_step_by_step
        self.rec_json_info = rec_json_info

        self.out_graph = self.output_path("graph.%s" % output_format)
        self.out_model_params = self.output_var("model_params.pickle", pickle=True)
        self.out_state_vars = self.output_var("state_vars.pickle", pickle=True)
        self.out_returnn_config = self.output_path("returnn.config")
        if self.rec_json_info:
            self.out_rec_json_info = self.output_path("rec.info")

        self.rqmt = None

    def tasks(self):
        if self.rqmt:
            yield Task("run", resume="run", rqmt=self.rqmt)
        else:
            yield Task("run", resume="run", mini_task=True)

    def run(self):
        if isinstance(self.returnn_config, tk.Path):
            returnn_config_path = self.returnn_config.get_path()
            shutil.copy(returnn_config_path, self.out_returnn_config.get_path())

        elif isinstance(self.returnn_config, ReturnnConfig):
            returnn_config_path = self.out_returnn_config.get_path()
            self.returnn_config.write(returnn_config_path)

        else:
            returnn_config_path = self.returnn_config
            shutil.copy(self.returnn_config, self.out_returnn_config.get_path())

        args = [
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("tools/compile_tf_graph.py").get_path(),
            returnn_config_path,
            "--train=%d" % self.train,
            "--eval=%d" % self.eval,
            "--search=%d" % self.search,
            "--verbosity=%d" % self.verbosity,
            "--output_file=%s" % self.out_graph.get_path(),
            "--output_file_model_params_list=model_params",
            "--output_file_state_vars_list=state_vars",
        ]
        if self.device is not None:
            args.append("--device=%s" % self.device)
        if self.epoch is not None:
            args.append("--epoch=%d" % util.instanciate_delayed(self.epoch))
        if self.summaries_tensor_name is not None:
            args.append("--summaries_tensor_name=%s" % self.summaries_tensor_name)
        if self.rec_step_by_step is not None:
            args.append(f"--rec_step_by_step={self.rec_step_by_step}")
            if self.rec_json_info:
                args.append(f"--rec_step_by_step_output_file={self.out_rec_json_info.get_path()}")

        util.create_executable("run.sh", args)

        sp.check_call(args)

        with open("model_params", "rt") as input:
            lines = [l.strip() for l in input if len(l.strip()) > 0]
            self.out_model_params.set(lines)
        with open("state_vars", "rt") as input:
            lines = [l.strip() for l in input if len(l.strip()) > 0]
            self.out_state_vars.set(lines)

    @classmethod
    def hash(cls, kwargs):
        c = copy.copy(kwargs)
        del c["verbosity"]
        return super().hash(c)


class CompileNativeOpJob(Job):
    """
    Compile a RETURNN native op into a shared object file.
    """

    __sis_hash_exclude__ = {"search_numpy_blas": True, "blas_lib": None}

    def __init__(
        self,
        native_op,
        returnn_python_exe=None,
        returnn_root=None,
        search_numpy_blas=True,
        blas_lib=None,
    ):
        """
        :param str native_op: Name of the native op to compile (e.g. NativeLstm2)
        :param Optional[Path] returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Optional[Path] returnn_root: file path to the RETURNN repository root folder
        :param bool search_numpy_blas: search for blas lib in numpy's .libs folder
        :param Path|str blas_lib: explicit path to the blas library to use
        """
        self.native_op = native_op
        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)
        self.search_numpy_blas = search_numpy_blas
        self.blas_lib = blas_lib

        self.out_op = self.output_path("%s.so" % native_op)
        self.out_grad_op = self.output_path("GradOf%s.so" % native_op)

        self.rqmt = None

    def tasks(self):
        if self.rqmt is None:
            yield Task("run", resume="run", mini_task=True)
        else:
            yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        cmd = [
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("tools/compile_native_op.py").get_path(),
            "--native_op",
            self.native_op,
            "--output_file",
            "compile.out",
        ]
        if not self.search_numpy_blas:
            cmd += ["--no_search_for_numpy_blas"]
        if self.blas_lib is not None:
            cmd += ["--blas_lib", tk.uncached_path(self.blas_lib)]
        logging.info(cmd)

        util.create_executable("compile.sh", cmd)  # convenience file for manual execution
        sp.run(cmd, check=True)

        with open("compile.out", "rt") as f:
            files = [l.strip() for l in f]

        if len(files) > 0:
            shutil.move(files[0], self.out_op.get_path())
        if len(files) > 1:
            shutil.move(files[1], self.out_grad_op.get_path())


class TorchOnnxExportJob(Job):
    """
    Export an ONNX model using the appropriate RETURNN tool script.

    Currently only supports PyTorch via tools/torch_export_to_onnx.py
    """

    __sis_hash_exclude__ = {"input_names": None, "output_names": None}

    def __init__(
        self,
        *,
        returnn_config: ReturnnConfig,
        checkpoint: PtCheckpoint,
        input_names: Optional[Sequence[str]] = None,
        output_names: Optional[Sequence[str]] = None,
        device: str = "cpu",
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        """

        :param returnn_config: RETURNN config object
        :param checkpoint: Path to the checkpoint for export
        :param device: target device for graph creation
        :param input_names: sequence of model input names
        :param output_names: sequence of model output names.
            If not specified, will automatically determine from model_outputs when available in returnn_config.config.
        :param returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param returnn_root: file path to the RETURNN repository root folder
        """

        self.returnn_config = returnn_config
        self.checkpoint = checkpoint
        self.input_names = input_names
        self.output_names = output_names
        self.device = device
        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)

        self.out_returnn_config = self.output_path("returnn.config")
        self.out_onnx_model = self.output_path("model.onnx")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        returnn_config_path = self.out_returnn_config.get_path()
        self.returnn_config.write(returnn_config_path)

        cmd = [
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("tools/torch_export_to_onnx.py").get_path(),
            returnn_config_path,
            str(self.checkpoint),
            self.out_onnx_model.get_path(),
            "--device",
            self.device,
            "--verbosity",
            "5",
        ]
        if self.input_names:
            cmd += ["--input_names", ",".join(self.input_names)]
        if self.output_names:
            cmd += ["--output_names", ",".join(self.output_names)]

        util.create_executable("compile.sh", cmd)  # convenience file for manual execution
        sp.run(cmd, check=True)
