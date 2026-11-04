"""
RETURNN forward jobs
"""

__all__ = ["ReturnnForwardJob", "ReturnnForwardJobV2", "BatchedReturnnForwardJob", "BatchedReturnnForwardDynamicJob"]

import copy
import glob
import os
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Union

from sisyphus import Job, Task, gs, setup_path, tk
from sisyphus.delayed_ops import DelayedBase

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint as TfCheckpoint
from i6_core.returnn.training import PtCheckpoint
import i6_core.util as util


Path = setup_path(__package__)

Checkpoint = Union[TfCheckpoint, PtCheckpoint, tk.Path, DelayedBase]


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
    """
    :param model: Base model checkpoint represented as a python/sisyphus object.
    :return: Path on disk to the model checkpoint.
    """

    if isinstance(model, DelayedBase) and not isinstance(model, tk.Path):
        # For instance, sisyphus.delayed_ops.DelayedGetItem could wrap around a tk.Path or PtCheckpoint.
        model = model.get()

    # Unwrap the internal tk.Path from any checkpoint that the user could have provided.
    if isinstance(model, TfCheckpoint):
        model = model.index_path
    elif isinstance(model, PtCheckpoint):
        model = model.path
    elif isinstance(model, tk.Path):
        # Already in tk.Path, which is what we need to return.
        pass
    else:
        raise TypeError(f"Expected model after unwrapping to be of type tk.Path but found model of type {type(model)}.")

    return model


class BatchedReturnnForwardJob(Job):
    """
    Runs a set of RETURNN forward work items (e.g. recognition of several test sets or data shards) on all GPUs
    of one node.

    Each work item is a ready-to-run RETURNN forward config (e.g. from
    :meth:`ReturnnForwardJobV2.create_returnn_config`, i.e. with ``task=forward`` and the checkpoint) together
    with the file(s) it writes into its working directory. The job places each item's output(s) at a declared
    ``output_path``.

    How the items are run:

    - one worker process per GPU (or several, see ``workers_per_gpu``), all pulling from a shared work queue:
      each worker claims the next unclaimed item (an ``O_EXCL`` marker file), so the GPUs share the work
      dynamically; items with a ``cost`` estimate (:class:`BatchedReturnnForwardDynamicJob`) are queued
      longest-first,
    - resumable: an item whose output file(s) already exist is skipped (outputs are moved into place
      atomically, so existence means done),
    - stops before the time limit, like RETURNN's ``stop_for_resubmission``: when the average item time
      times ``_stop_safety_factor`` exceeds the remaining Slurm time, a worker finishes its current item and
      waits until the other workers have done the same; then the job exits as interrupted and Sisyphus
      resubmits it (``Task("run", resume="run")``),
    - ``completed_fraction`` = items done / all items, for the Sisyphus progress display.

    Several workers per GPU (``workers_per_gpu``) are only useful under NVIDIA MPS (``use_mps``), which lets
    the kernels of several processes on one GPU run concurrently; this helps if a single process does not
    keep the GPU busy (e.g. decoding with many small kernels). Without MPS, the GPU time-slices between the
    processes, which is slower than one process per GPU.

    By default, each item runs as its own rnn.py process, i.e. RETURNN starts up and loads the model again for
    every item. With ``persistent_workers``, a worker runs its items in its own process instead and does this only
    once (if the configs of all items are the same except ``forward_data``, otherwise each item still gets its
    own process), which saves the startup time of all but the first item of each worker.

    Hashing: only the work items and ``returnn_python_exe`` / ``returnn_root``; how they are run (node
    size, workers per GPU, MPS, persistent workers) does not change the result and is not hashed.
    """

    _stop_safety_factor = 1.2  # same default as RETURNN's stop_for_resubmission_safety_factor
    _stop_exit_code = 3

    def __init__(
        self,
        work_items: Dict[str, Dict[str, Any]],
        *,
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
        num_gpus: int = 4,
        cpu_rqmt: Optional[int] = None,
        mem_rqmt: Optional[float] = None,
        time_rqmt: float = 11.95,
        workers_per_gpu: int = 1,
        use_mps: bool = False,
        persistent_workers: bool = False,
    ):
        """
        :param work_items: mapping ``key -> {"returnn_config": ReturnnConfig,
            "output_files": list[str], "model_checkpoint": Optional[tk.Path]}``.
            ``key`` must be a valid path component (no ``/``). ``returnn_config`` must already be a
            forward-ready config (e.g. from :meth:`ReturnnForwardJobV2.create_returnn_config`); it is
            written to ``items/<key>/returnn.config`` and run with ``rnn.py`` in a fresh tmp cwd, so
            the file(s) named in ``output_files`` (written relative to cwd) are picked up there and
            moved to their declared ``output_path`` destinations. ``model_checkpoint`` is optional
            and only used for an existence pre-check (the checkpoint is already embedded in the
            config); the config is what is hashed.
        :param returnn_python_exe:
        :param returnn_root:
        :param num_gpus: number of GPUs of the node to run the workers on. The defaults of the requirements
            correspond to a full JUPITER node (4x GH200, 72 cores and ~100 GB host memory per GPU, 12 h limit)
        :param cpu_rqmt: job cpu requirement (all workers together), default 72 per GPU
        :param mem_rqmt: job memory requirement in GB (all workers together), default 100 per GPU
        :param time_rqmt: job time requirement in hours (the job stops early and is resubmitted, see above)
        :param workers_per_gpu: worker processes per GPU, see :func:`_spawn_and_wait_workers`
        :param use_mps: run the workers under NVIDIA MPS, see :func:`_spawn_and_wait_workers`
        :param persistent_workers: run the items of a worker in the worker process itself, so RETURNN is
            initialized and the model is loaded only once per worker instead of once per item, see
            :func:`_worker_main`. Only used if the configs of all items are the same except ``forward_data``
        """
        super().__init__()
        assert num_gpus >= 1, num_gpus
        assert workers_per_gpu >= 1, workers_per_gpu
        self.num_gpus = num_gpus
        self.workers_per_gpu = workers_per_gpu
        self.use_mps = use_mps
        self.persistent_workers = persistent_workers
        assert work_items, "BatchedReturnnForwardJob: work_items dict must be non-empty"
        for key, item in work_items.items():
            assert isinstance(key, str) and key and "/" not in key and "\\" not in key, (
                f"work item key {key!r} must be a non-empty path component"
            )
            assert isinstance(item.get("returnn_config"), ReturnnConfig), f"item {key}: returnn_config required"
            out_files = item.get("output_files")
            assert isinstance(out_files, (list, tuple)) and out_files, (
                f"item {key}: output_files must be a non-empty list"
            )
            for fn in out_files:
                assert isinstance(fn, str) and fn, f"item {key}: output_files entries must be non-empty str"

        self.work_items = work_items
        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)

        # Per-item outputs: dict key -> dict output_filename -> Path inside the Job's output/.
        import os

        self.out_files: Dict[str, Dict[str, tk.Path]] = {
            key: {fn: self.output_path(os.path.join("outputs", key, fn)) for fn in item["output_files"]}
            for key, item in work_items.items()
        }

        self.rqmt = {
            "gpu": num_gpus,
            "cpu": cpu_rqmt if cpu_rqmt is not None else 72 * num_gpus,
            "mem": mem_rqmt if mem_rqmt is not None else 100 * num_gpus,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_count(self) -> int:
        """:return: number of work items whose output file(s) are all already written."""
        import os

        n = 0
        for outs in self.out_files.values():
            if all(os.path.exists(p.get_path()) for p in outs.values()):
                n += 1
        return n

    def completed_fraction(self) -> float:
        """:return: fraction of items done (for sis progress/ETA). Mirrors ReturnnTrainingJob."""
        return self.completed_count() / len(self.work_items)

    def create_files(self):
        """Write one RETURNN forward config per work item + the worker manifest and driver."""
        import os

        items = []
        for key, item in self.work_items.items():
            item_dir = os.path.join("items", key)
            os.makedirs(item_dir, exist_ok=True)
            cfg_path = os.path.join(item_dir, "returnn.config")
            signature = _config_signature(item["returnn_config"])
            item["returnn_config"].write(cfg_path)
            items.append(
                {
                    "key": key,
                    "config": os.path.abspath(cfg_path),
                    # Where each file the config writes (relative to cwd) must end up.
                    "outputs": {fn: self.out_files[key][fn].get_path() for fn in item["output_files"]},
                    "signature": signature,
                }
            )
            ckpt = item.get("model_checkpoint")
            if ckpt is not None:
                ckpt_path = ckpt.get_path() if isinstance(ckpt, tk.Path) else getattr(ckpt, "path", ckpt).get_path()
                assert os.path.exists(ckpt_path), "item %s: checkpoint missing: %s" % (key, ckpt_path)

        _write_manifest(
            items=items,
            returnn_root_path=self.returnn_root.get_path(),
            returnn_python_exe_path=self.returnn_python_exe.get_path(),
        )
        _write_worker_driver()

    def run(self):
        """Spawn ``workers_per_gpu`` workers per GPU; round-robin items; resubmit if stopped early for walltime."""
        codes = _spawn_and_wait_workers(
            num_gpus=self.num_gpus,
            returnn_python_exe_path=self.returnn_python_exe.get_path(),
            stop_safety_factor=self._stop_safety_factor,
            stop_exit_code=self._stop_exit_code,
            workers_per_gpu=self.workers_per_gpu,
            use_mps=self.use_mps,
            persistent=self.persistent_workers,
        )
        _finish_run(
            codes=codes,
            n_done=self.completed_count(),
            n_total=len(self.work_items),
            stop_exit_code=self._stop_exit_code,
            label="BatchedReturnnForwardJob",
        )

    @classmethod
    def hash(cls, kwargs):
        d = {
            "work_items": kwargs["work_items"],
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }
        return super().hash(d)


class BatchedReturnnForwardDynamicJob(Job):
    """
    Like :class:`BatchedReturnnForwardJob`, but the set of work items (cells) is NOT known at
    __init__ -- it is enumerated at run time by a user callable. Hence, a single directory output
    ``out/`` into which each cell writes its file(s) at a cell-chosen relative path, instead of one
    declared ``output_path`` per (statically known) cell. Use this when the cell list is only
    discoverable at run time (e.g. one cell per dynamically-discovered data shard); for a statically
    known cell set use :class:`BatchedReturnnForwardJob`.

    The cell enumeration separates cheap listing from expensive config building, so
    ``completed_fraction`` / resumability never build configs. ``enumerate_cells()`` returns an
    ordered ``list[(key, make_config, outputs)]`` -- optionally ``(key, make_config, outputs, cost)``
    -- where

    - ``key`` is a path component (no ``/``) naming the cell,
    - ``make_config`` is a zero-arg callable returning the cell's forward-ready ``ReturnnConfig``
      (already through :meth:`ReturnnForwardJobV2.create_returnn_config`); invoked only for cells
      whose output(s) are not yet present,
    - ``outputs`` maps each cwd filename the config writes to its destination path *relative to* this
      job's ``out/`` directory,
    - ``cost`` (optional) estimates the cell's run time, bigger = longer, in any unit (e.g. its Arrow
      shard's byte size); cells are queued longest-first by it, see :func:`_write_manifest`. Worth
      supplying whenever cells differ a lot in size -- it is what keeps all GPUs busy to the end.

    ``enumerate_cells`` itself must be cheap + deterministic + picklable + hashable (e.g. a
    ``functools.partial`` of a module-level function); it is what the job hashes on (plus
    ``returnn_python_exe`` / ``returnn_root``). Same multi-GPU node engine + walltime barrier-stop +
    resumability (a cell whose output(s) exist is skipped) as :class:`BatchedReturnnForwardJob`.
    """

    _stop_safety_factor = 1.2
    _stop_exit_code = 3

    def __init__(
        self,
        enumerate_cells: Callable[[], List[Any]],
        *,
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
        num_gpus: int = 4,
        cpu_rqmt: Optional[int] = None,
        mem_rqmt: Optional[float] = None,
        time_rqmt: float = 11.95,
        workers_per_gpu: int = 1,
        use_mps: bool = False,
        persistent_workers: bool = False,
        file_cache_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        :param enumerate_cells: zero-arg callable returning ``list[(key, make_config, outputs)]``
            (see the class docstring). Invoked at run time (returnn on sys.path) and for
            ``completed_fraction``, so it must be cheap, deterministic, picklable + hashable.
        :param returnn_python_exe:
        :param returnn_root:
        :param num_gpus: number of GPUs of the node to run the workers on. The defaults of the requirements
            correspond to a full JUPITER node (4x GH200, 72 cores and ~100 GB host memory per GPU, 12 h limit)
        :param cpu_rqmt: job cpu requirement (all workers together), default 72 per GPU
        :param mem_rqmt: job memory requirement in GB (all workers together), default 100 per GPU
        :param time_rqmt: job time requirement in hours (the job stops early and is resubmitted)
        :param workers_per_gpu: worker processes per GPU, see :func:`_spawn_and_wait_workers`
        :param use_mps: run the workers under NVIDIA MPS, see :func:`_spawn_and_wait_workers`
        :param persistent_workers: run the items of a worker in the worker process itself, so RETURNN is
            initialized and the model is loaded only once per worker instead of once per item, see
            :func:`_worker_main`. Only used if the configs of all items are the same except ``forward_data``
        :param file_cache_opts: RETURNN ``file_cache_opts``, set in each cell's ``post_config`` (unless the cell
            sets it itself). Only has an effect if the cells read their data via RETURNN's file cache. Default
            (None): ``{"cleanup_files_wanted_older_than_days": 1 / (24 * 60)}``, i.e. the files of finished cells
            may be removed from the cache after a minute instead of after a day: the workers share one local
            cache directory, and with RETURNN's default it would fill up within a multi-hour run (each cell's
            data is read once, so evicting early costs nothing; files in use are never evicted).
            ``{}``: RETURNN's defaults.
        """
        super().__init__()
        assert num_gpus >= 1, num_gpus
        assert workers_per_gpu >= 1, workers_per_gpu
        self.num_gpus = num_gpus
        self.workers_per_gpu = workers_per_gpu
        self.use_mps = use_mps
        self.persistent_workers = persistent_workers
        self.file_cache_opts = (
            file_cache_opts
            if file_cache_opts is not None
            else {"cleanup_files_wanted_older_than_days": 1.0 / (24 * 60)}
        )
        self.enumerate_cells = enumerate_cells
        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)

        # Single dynamic-count directory output: filled with each cell's file(s) in run().
        self.out_dir = self.output_path("out", directory=True)

        self.rqmt = {
            "gpu": num_gpus,
            "cpu": cpu_rqmt if cpu_rqmt is not None else 72 * num_gpus,
            "mem": mem_rqmt if mem_rqmt is not None else 100 * num_gpus,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def _cell_dest(self, rel_dest: str) -> str:
        """:return: absolute destination of a cell output file (``rel_dest`` is under ``out/``)."""
        import os

        return os.path.join(self.out_dir.get_path(), rel_dest)

    def _cell_done(self, outputs: Dict[str, str]) -> bool:
        """:return: whether all of a cell's output files already exist."""
        import os

        return all(os.path.exists(self._cell_dest(rel)) for rel in outputs.values())

    def completed_count(self) -> int:
        """:return: number of cells whose output file(s) are all already written."""
        return sum(1 for cell in self.enumerate_cells() if self._cell_done(cell[2]))

    def completed_fraction(self) -> float:
        """:return: fraction of cells done (for sis progress/ETA). Mirrors ReturnnTrainingJob."""
        cells = self.enumerate_cells()
        if not cells:
            return 0.0
        return sum(1 for cell in cells if self._cell_done(cell[2])) / len(cells)

    def create_files(self):
        """Write just the worker driver; the per-cell configs + manifest are built in run() (dynamic)."""
        _write_worker_driver()

    def run(self):
        """Enumerate cells, build a config per not-yet-done cell, run them all across the node."""
        import os
        import sys

        # Config building (whatever enumerate_cells' make_config does) may import returnn.
        sys.path.insert(0, self.returnn_root.get_path())
        os.makedirs(self.out_dir.get_path(), exist_ok=True)

        cells = self.enumerate_cells()
        assert cells, "enumerate_cells() returned no cells"

        items = []
        for cell in cells:
            key, make_config, outputs = cell[0], cell[1], cell[2]
            dests = {fn: self._cell_dest(rel) for fn, rel in outputs.items()}
            if all(os.path.exists(dest) for dest in dests.values()):
                continue  # resumable: this cell already done
            cfg = make_config()
            if self.file_cache_opts:
                cfg.post_config.setdefault("file_cache_opts", self.file_cache_opts)
            item_dir = os.path.join("items", key)
            os.makedirs(item_dir, exist_ok=True)
            cfg_path = os.path.join(item_dir, "returnn.config")
            signature = _config_signature(cfg)
            cfg.write(cfg_path)
            # cost (optional 4th element) -> queue order, see _write_manifest
            items.append(
                {
                    "key": key,
                    "config": os.path.abspath(cfg_path),
                    "outputs": dests,
                    "cost": cell[3] if len(cell) > 3 else None,
                    "signature": signature,
                }
            )

        if items:
            _write_manifest(
                items=items,
                returnn_root_path=self.returnn_root.get_path(),
                returnn_python_exe_path=self.returnn_python_exe.get_path(),
            )
            codes = _spawn_and_wait_workers(
                num_gpus=self.num_gpus,
                returnn_python_exe_path=self.returnn_python_exe.get_path(),
                stop_safety_factor=self._stop_safety_factor,
                stop_exit_code=self._stop_exit_code,
                workers_per_gpu=self.workers_per_gpu,
                use_mps=self.use_mps,
                persistent=self.persistent_workers,
            )
        else:
            codes = [0]  # nothing left (all cells already present)

        _finish_run(
            codes=codes,
            n_done=self.completed_count(),
            n_total=len(cells),
            stop_exit_code=self._stop_exit_code,
            label="BatchedReturnnForwardDynamicJob",
        )

    @classmethod
    def hash(cls, kwargs):
        d = {
            "enumerate_cells": kwargs["enumerate_cells"],
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }
        return super().hash(d)


def _worker_main():
    """
    Entry point of the per-GPU worker (the generated ``worker.py`` imports and calls this).

    One worker process per GPU, all pulling from one shared work queue: walk the manifest's items in
    order (longest-first, see :func:`_write_manifest`), skip the ones whose output file(s) already
    exist (resumable) or that a peer has claimed, claim the next one atomically
    (:func:`_worker_claim`), run its config, repeat until the queue is drained; and
    barrier-stop together with the other workers when low on walltime. Pulling instead of striding
    means a worker that draws a cheap item comes back for more, so no GPU idles while a peer still
    has hours of work queued. Args ``--rank/--world/--manifest/--claims/--safety/--persistent`` come from
    argv; ``--claims`` is a per-run directory (stale claims of an earlier run would deadlock the queue).

    Each item runs either as its own rnn.py process (:func:`_worker_run_item`) or, with ``--persistent`` and
    if the configs of all items are the same except ``forward_data``, in this process
    (:func:`_worker_run_item_in_process`): RETURNN is initialized and the model is loaded for the first item
    only, the later items only create their dataset and run the forward. This saves the startup (imports,
    CUDA init, model and checkpoint loading, and everything the forward step caches, e.g. a RASR search)
    for all but the first item of a worker. An error in an item ends the worker in both modes.
    """
    import argparse
    import json
    import os
    import sys
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--safety", type=float, default=1.2)
    ap.add_argument("--persistent", action="store_true")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    sys.path.insert(0, manifest["returnn_root"])

    from returnn.util.basic import slurm_time_left_sec

    items = manifest["items"]
    python_exe = manifest["python_exe"]
    rnn_py = manifest["rnn_py"]
    job_dir = os.path.dirname(os.path.abspath(args.manifest))

    persistent = args.persistent
    if persistent:
        signatures = {item.get("signature") for item in items}
        if None in signatures or len(signatures) > 1:
            if args.rank == 0:
                print(
                    "BatchedReturnnForward: the item configs differ in more than forward_data, "
                    "running each item as its own rnn.py process",
                    flush=True,
                )
            persistent = False
    in_process_state = {}

    ema = None
    for si, item in enumerate(items):
        if _worker_item_done(item):  # atomic os.replace -> existence == complete
            continue
        if ema is not None:  # walltime-aware stop, mirrors returnn _maybe_stop_for_resubmission
            left = slurm_time_left_sec()
            if left is not None and left < ema * args.safety:
                # before claiming: a claim held by a stopped worker would hide that item from the
                # peers that do still have walltime left
                _worker_barrier_and_exit(args.rank, args.world, job_dir, ema)
        if not _worker_claim(args.claims, si):  # a peer got there first
            continue
        t0 = time.time()
        if persistent:
            _worker_run_item_in_process(in_process_state, item)
        else:
            _worker_run_item(python_exe, rnn_py, item)
        dt = time.time() - t0
        ema = dt if ema is None else 0.5 * ema + 0.5 * dt

    if in_process_state:
        _worker_finish_in_process(in_process_state)
    # Queue drained: register so a stopping peer's barrier can complete, then exit clean.
    open(os.path.join(job_dir, "stopping.rank%i" % args.rank), "w").close()
    sys.exit(0)


def _worker_item_done(item):
    """:return: whether all of a work item's declared output files already exist."""
    import os

    return all(os.path.exists(dest) for dest in item["outputs"].values())


def _worker_claim(claims_dir, item_index):
    """
    Try to take item ``item_index`` out of the shared queue, exactly once across all workers.

    ``O_CREAT|O_EXCL`` on a shared filesystem is the atomic test-and-set: the one worker whose create
    succeeds owns the item, everyone else sees EEXIST and moves on. ``claims_dir`` is per run (see
    :func:`_spawn_and_wait_workers`), so a resubmitted run starts from an empty queue state and
    re-runs whatever the killed run had claimed but not finished.

    :return: whether this worker now owns the item
    """
    import os

    try:
        fd = os.open(os.path.join(claims_dir, "%i" % item_index), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def _worker_run_item(python_exe, rnn_py, item):
    """Run one item's config with rnn.py in a fresh tmp cwd, then atomically move its outputs into place."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([python_exe, rnn_py, item["config"]], cwd=tmp)
        _worker_move_outputs(tmp, item)


def _worker_run_item_in_process(state, item):
    """
    Run one item's forward in this process (persistent worker), in a fresh tmp cwd, then atomically move its
    outputs into place.

    The first call initializes RETURNN with the item's config (like rnn.py) and builds the model
    (``engine.init_network_from_config``); this stays for all later items, whose configs are the same except
    ``forward_data`` (checked via the item signatures in :func:`_worker_main`). So per item only the dataset and
    the forward callback are created, as in RETURNN's ``execute_main_task`` for ``task="forward"``.

    :param dict state: empty before the first item, then holds the worker's RETURNN working directory and the
        original cwd
    :param dict item: manifest entry
    """
    import os
    import tempfile

    import returnn.__main__ as rnn_main
    from returnn.config import Config

    if not state:
        # RETURNN opens its log file (e.g. ./returnn.log) relative to the cwd at init: keep it in a worker dir
        state["orig_cwd"] = os.getcwd()
        state["work_dir"] = tempfile.mkdtemp(prefix="returnn_worker_")
        os.chdir(state["work_dir"])
        rnn_main.init(config_filename=item["config"], extra_greeting="BatchedReturnnForward: persistent worker")
        assert rnn_main.config.value("task", "train") == "forward", "persistent workers need task=forward"
        rnn_main.engine.init_network_from_config(rnn_main.config)
        forward_data = rnn_main.config.opt_typed_value("forward_data")
    else:
        item_config = Config()
        item_config.load_file(item["config"])
        forward_data = item_config.opt_typed_value("forward_data")
        rnn_main.config.set("forward_data", forward_data)
    assert isinstance(forward_data, dict), "item %s: forward_data must be a dataset dict" % item["key"]

    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # the forward callback writes its output file(s) relative to the cwd
        try:
            dataset = rnn_main.init_dataset(forward_data)
            forward_callback = rnn_main.config.typed_value("forward_callback")
            assert forward_callback, "no forward_callback specified"
            if callable(forward_callback):
                forward_callback = forward_callback()
            rnn_main.engine.forward_with_callback(
                dataset=dataset,
                callback=forward_callback,
                allow_skipping_seqs=rnn_main.config.bool("allow_skipping_seqs_in_forward", False),
            )
            del dataset, forward_callback
        finally:
            os.chdir(state["work_dir"])
        _worker_move_outputs(tmp, item)


def _worker_finish_in_process(state):
    """Shut down the RETURNN instance of a persistent worker (:func:`_worker_run_item_in_process`)."""
    import os
    import shutil

    import returnn.__main__ as rnn_main

    rnn_main.finalize()
    os.chdir(state["orig_cwd"])  # RETURNN's exit handlers still use the cwd
    shutil.rmtree(state["work_dir"], ignore_errors=True)


def _worker_move_outputs(tmp, item):
    """Atomically move the output file(s) an item wrote into ``tmp`` to their destinations."""
    import os
    import shutil

    for fn, dest in item["outputs"].items():
        src = os.path.join(tmp, fn)
        assert os.path.exists(src), "item produced no %s: %s" % (fn, item["config"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        staged = dest + ".inprogress"  # same fs as final -> os.replace is atomic
        shutil.move(src, staged)
        os.replace(staged, dest)


def _config_signature(returnn_config: ReturnnConfig) -> Optional[str]:
    """
    :return: fingerprint of an item's config without ``forward_data``, so that items with the same fingerprint
        can share one RETURNN instance and model (persistent workers); None if ``forward_data`` is not a dataset
        dict (e.g. ``"eval"``), then the item needs its own process
    """
    import hashlib

    if not isinstance(returnn_config.get("forward_data"), dict):
        return None
    cfg = copy.deepcopy(returnn_config)  # _serialize() modifies the config: merges post_config into config
    for key in cfg.post_config:  # already merged if the config was written before
        cfg.config.pop(key, None)
    cfg.config.pop("forward_data", None)
    cfg.post_config.pop("forward_data", None)
    return hashlib.sha256(cfg._serialize().encode("utf8")).hexdigest()


def _worker_barrier_and_exit(rank, world, job_dir, ema):
    """
    Stop cleanly when low on walltime, in lockstep with the other workers.

    Register at the filesystem barrier, then wait for the other still-active workers to reach their
    own stop point (finish their in-flight item) so all exit together with no aborted partial item;
    bounded by a timeout. Exits with code 3, which must match
    :attr:`BatchedReturnnForwardJob._stop_exit_code` -- run() treats it as a clean low-walltime stop
    (resubmit), not a failure.
    """
    import glob
    import os
    import sys
    import time

    open(os.path.join(job_dir, "stopping.rank%i" % rank), "w").close()
    deadline = time.time() + max(600.0, 3.0 * (ema or 0.0))
    while time.time() < deadline:
        if len(glob.glob(os.path.join(job_dir, "stopping.rank*"))) >= world:
            break
        time.sleep(5.0)
    sys.exit(3)


def _write_manifest(*, items: List[Dict[str, Any]], returnn_root_path: str, returnn_python_exe_path: str):
    """
    Write the worker ``manifest.json`` (the work items + the returnn/python paths used to run them).

    The item order is the shared queue's order, so items carrying a ``"cost"`` estimate (bigger =
    longer, arbitrary unit) are written longest-first: with the pull queue in :func:`_worker_main`
    that is LPT scheduling, whose makespan is within 4/3 of optimal. Items without a ``"cost"`` keep
    the caller's order (sorting is stable, and a cost-less item set is left alone entirely).
    """
    import os
    import json

    if any(item.get("cost") is not None for item in items):
        items = sorted(items, key=lambda item: -(item.get("cost") or 0.0))

    manifest = {
        "rnn_py": os.path.join(returnn_root_path, "rnn.py"),
        "returnn_root": returnn_root_path,
        "python_exe": returnn_python_exe_path,
        "items": items,
    }
    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _write_worker_driver():
    """
    Write the per-GPU worker driver ``worker.py``.

    Serialize an import of the worker entry point with i6_core's serialization_v2, so the logic lives
    as a real function in this module (:func:`_worker_main`) instead of a source-string blob. The
    generated worker.py is run by a plain Python (not via rnn.py). serialization_v2 resolves the import
    of this module (and its sys.path); we add the remaining non-base sys.path entries (sisyphus,
    returnn, ...) explicitly, then append the (``__main__``-guarded) call.
    """
    import sys
    from i6_core.serialization.serialization_v2 import serialize_config, get_base_sys_path_list

    extra_sys_paths = [p for p in sys.path if p not in get_base_sys_path_list()]
    worker_code = (
        serialize_config({"_worker_main": _worker_main}, extra_sys_paths=extra_sys_paths)
        # guarded: subprocesses started with multiprocessing "spawn" (e.g. by RETURNN's data loading in a persistent
        # worker) import this file as their main module and must not run the worker again
        + '\nif __name__ == "__main__":\n    _worker_main()\n'
    )
    with open("worker.py", "w") as f:
        f.write(worker_code)


def _spawn_and_wait_workers(
    *,
    num_gpus: int,
    returnn_python_exe_path: str,
    stop_safety_factor: float,
    stop_exit_code: int,
    workers_per_gpu: int = 1,
    use_mps: bool = False,
    persistent: bool = False,
) -> List[int]:
    """
    Run the workers (:func:`_run_workers`), optionally under a private NVIDIA MPS daemon.

    With ``use_mps``, one MPS control daemon is started for this job (all GPUs of the node; pipe and log
    directories in node-local /tmp, so it only serves this job's processes) and stopped again at the end,
    also on failure. The workers and their rnn.py children inherit the daemon's environment and become MPS
    clients, so the kernels of several processes on one GPU run concurrently. If MPS cannot be started,
    the job falls back to one worker per GPU without MPS (several workers per GPU without MPS are slower
    than one, because the GPU then time-slices between the processes).

    :param workers_per_gpu: worker processes per GPU, all pulling from the same queue
    :param use_mps: run the workers under MPS
    :param persistent: persistent workers, see :func:`_worker_main`
    """
    mps_env = _start_mps_daemon(num_gpus=num_gpus) if use_mps else None
    if workers_per_gpu > 1 and mps_env is None:
        print(
            "BatchedReturnnForward: %i workers per GPU need MPS (%s), using 1 worker per GPU"
            % (workers_per_gpu, "not started" if use_mps else "use_mps=False"),
            flush=True,
        )
        workers_per_gpu = 1
    try:
        return _run_workers(
            num_gpus=num_gpus,
            returnn_python_exe_path=returnn_python_exe_path,
            stop_safety_factor=stop_safety_factor,
            stop_exit_code=stop_exit_code,
            workers_per_gpu=workers_per_gpu,
            extra_env=mps_env or {},
            persistent=persistent,
        )
    finally:
        if mps_env is not None:
            _stop_mps_daemon(mps_env)


def _start_mps_daemon(*, num_gpus: int) -> Optional[Dict[str, str]]:
    """
    Start a private NVIDIA MPS control daemon for this job.

    :return: the environment variables its clients need (pipe and log directory), or None if MPS is not
        usable here
    """
    import os
    import shutil
    import subprocess
    import tempfile

    if shutil.which("nvidia-cuda-mps-control") is None:
        print("BatchedReturnnForward: MPS: nvidia-cuda-mps-control not found", flush=True)
        return None
    # Clients address GPUs relative to the devices the MPS server sees. With the whole node
    # (CUDA_VISIBLE_DEVICES unset or 0..n-1) this equals the absolute index used in _run_workers.
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_cvd and parent_cvd.split(",") != [str(i) for i in range(num_gpus)]:
        print("BatchedReturnnForward: MPS: unexpected CUDA_VISIBLE_DEVICES=%s, not using MPS" % parent_cvd, flush=True)
        return None
    base_dir = tempfile.mkdtemp(prefix="mps_", dir="/tmp")  # node-local, not $TMPDIR (may be shared FS)
    env = {
        "CUDA_MPS_PIPE_DIRECTORY": os.path.join(base_dir, "pipe"),
        "CUDA_MPS_LOG_DIRECTORY": os.path.join(base_dir, "log"),
    }
    for d in env.values():
        os.makedirs(d)
    res = subprocess.run(["nvidia-cuda-mps-control", "-d"], env={**os.environ, **env}, capture_output=True, text=True)
    if res.returncode != 0:
        print(
            "BatchedReturnnForward: MPS: daemon start failed (%i): %s"
            % (res.returncode, (res.stdout + res.stderr).strip()),
            flush=True,
        )
        shutil.rmtree(base_dir, ignore_errors=True)
        return None
    print("BatchedReturnnForward: MPS: daemon started, pipe directory %s" % env["CUDA_MPS_PIPE_DIRECTORY"], flush=True)
    return env


def _stop_mps_daemon(env: Dict[str, str]):
    """Stop the MPS daemon of :func:`_start_mps_daemon`, keep its logs in ``mps_logs/`` of the job dir."""
    import os
    import shutil
    import subprocess

    try:
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input="quit\n",
            env={**os.environ, **env},
            text=True,
            timeout=60,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        print("BatchedReturnnForward: MPS: daemon did not quit within 60 s", flush=True)
    shutil.copytree(env["CUDA_MPS_LOG_DIRECTORY"], "mps_logs", dirs_exist_ok=True)
    shutil.rmtree(os.path.dirname(env["CUDA_MPS_PIPE_DIRECTORY"]), ignore_errors=True)
    print("BatchedReturnnForward: MPS: daemon stopped", flush=True)


def _run_workers(
    *,
    num_gpus: int,
    returnn_python_exe_path: str,
    stop_safety_factor: float,
    stop_exit_code: int,
    workers_per_gpu: int,
    extra_env: Dict[str, str],
    persistent: bool = False,
) -> List[int]:
    """
    Clear stale barrier/claim state, spawn ``workers_per_gpu`` workers per GPU (shared queue), wait; return
    exit codes.

    Fail-fast: if any worker exits with an unexpected code (not 0, not the clean low-walltime
    ``stop_exit_code``), terminate every still-running worker -- whole process group, so its in-flight
    rnn.py child dies too -- instead of letting survivors grind through their full strides. Such a
    failure is of unknown origin (e.g. /tmp ENOSPC, where continuing would only keep the disk full),
    so the job should fail as a unit. Killed workers report their negative terminating signal as code.
    """
    import os
    import glob
    import shutil
    import signal
    import subprocess
    import time

    # Clear stale barrier markers from a previous (interrupted) run.
    for f in glob.glob("stopping.rank*"):
        os.remove(f)
    # Fresh work-queue claims for this run: an earlier run's claims would hide its unfinished items
    # from every worker and drain the queue instantly. No worker of an earlier run is alive here --
    # _spawn_and_wait_workers only returns once all of them are reaped.
    claims_dir = os.path.abspath("claims")
    shutil.rmtree(claims_dir, ignore_errors=True)
    os.makedirs(claims_dir)

    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = parent_cvd.split(",") if parent_cvd else [str(i) for i in range(num_gpus)]
    world = len(gpu_ids) * workers_per_gpu
    manifest = os.path.abspath("manifest.json")
    worker = os.path.abspath("worker.py")

    procs = []
    for r in range(world):
        env = {**os.environ, **extra_env}
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[r % len(gpu_ids)]  # ranks 0..n-1 on GPUs 0..n-1, then again
        cmd = [
            returnn_python_exe_path,
            worker,
            "--rank",
            str(r),
            "--world",
            str(world),
            "--manifest",
            manifest,
            "--claims",
            claims_dir,
            "--safety",
            str(stop_safety_factor),
        ]
        if persistent:
            cmd.append("--persistent")
        # Own session/process group so a fail-fast kill takes the worker's rnn.py child with it.
        procs.append(subprocess.Popen(cmd, env=env, start_new_session=True))

    codes: List[Optional[int]] = [None] * world
    while any(c is None for c in codes):
        for i, p in enumerate(procs):
            if codes[i] is None:
                codes[i] = p.poll()
        if any(c is not None and c not in (0, stop_exit_code) for c in codes):
            break  # unexpected failure -> stop every still-running worker
        if any(c is None for c in codes):
            time.sleep(2.0)

    for i, p in enumerate(procs):  # no-op on the all-clean path (loop drained naturally)
        if codes[i] is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    for i, p in enumerate(procs):
        if codes[i] is None:
            try:
                codes[i] = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                codes[i] = p.wait()
    return codes


def _finish_run(*, codes: List[int], n_done: int, n_total: int, stop_exit_code: int, label: str):
    """
    Shared post-spawn verdict for both engine jobs.

    All items written -> return (done). Else if every worker exited cleanly (0) or via the clean
    low-walltime stop (``stop_exit_code``), raise KeyboardInterrupt so sis resubmits ``run`` and the
    existing-output skip continues from the remaining items. Otherwise a real worker failure.
    """
    if n_done >= n_total:
        return
    if all(c in (0, stop_exit_code) for c in codes):
        raise KeyboardInterrupt("%s: stop for resubmission (%i/%i done)" % (label, n_done, n_total))
    raise RuntimeError("%s: worker failure codes=%s (%i/%i done)" % (label, codes, n_done, n_total))
