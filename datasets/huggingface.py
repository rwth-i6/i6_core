"""
https://huggingface.co/docs/datasets/
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Union, Callable, Sequence, Dict
from sisyphus import Job, Task, Path, gs
from sisyphus.delayed_ops import DelayedBase

from i6_core.util import instanciate_delayed

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

    TransformFuncT = Union[Callable[[DatasetDict], DatasetDict], Callable[[Dataset], Dataset]]


def load_hf_dataset(path, **opts):
    """
    If `HF_HUB_OFFLINE` is set to true, only use the local files in `HF_HOME` to load the dataset.
    Otherwise, `load_dataset` tries to load metadata from the internet.

    This can be used for nodes which don't have internet access (e.g. JUPITER in Jülich).
    """
    import os
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    try:
        # noinspection PyProtectedMember
        from huggingface_hub import is_offline_mode  # type: ignore[attr-defined]
    except ImportError:
        # ``is_offline_mode`` is not exported in earlier ``huggingface_hub`` releases;
        # mirror its semantics by checking the ``HF_HUB_OFFLINE`` env var directly.
        def is_offline_mode() -> bool:
            return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")

    if is_offline_mode():
        # `snapshot_download` just locates the dataset files, if they already exist on disk
        # `load_dataset` does more, including a check for metadata online, which would not work without internet access
        path = snapshot_download(
            repo_id=path,
            repo_type="dataset",
            local_files_only=True,
        )

    return load_dataset(path, **opts)


_hf_hub_transient_retries_patched = False


def patch_hf_hub_transient_retries(*, max_retries: int = 8, max_wait_time: float = 30.0):
    """
    Make ``huggingface_hub`` survive mid-download connection drops.

    Observed on large WebDataset tar shards (~3-4GB): the CDN closes the connection mid-body,
    raising ``httpx.RemoteProtocolError("peer closed connection without sending complete message
    body")``. Two gaps in ``huggingface_hub`` (checked with 1.16.1) turn that into a hard failure,
    losing the whole (many hours long) shard:

    - ``_DEFAULT_RETRY_ON_EXCEPTIONS`` is ``(httpx.TimeoutException, httpx.NetworkError)``, which
      does not cover ``httpx.RemoteProtocolError``, so the backoff in ``http_backoff`` /
      ``http_stream_backoff`` does not apply to it.
    - ``HfFileSystemStreamFile.read`` retries a failed stream read exactly once.

    Both are patched here. The stream read reopens the connection with a ``Range`` header at the
    current offset, so a retry resumes instead of restarting the download.

    Best effort: any failure to patch (e.g. after a ``huggingface_hub`` refactoring) is logged and
    ignored, leaving the original behaviour in place. Idempotent.
    """
    global _hf_hub_transient_retries_patched
    if _hf_hub_transient_retries_patched:
        return
    _hf_hub_transient_retries_patched = True

    import logging
    import time
    import httpx

    try:
        from huggingface_hub.utils import _http as hf_http

        retry_on = tuple(hf_http._DEFAULT_RETRY_ON_EXCEPTIONS)
        if httpx.RemoteProtocolError not in retry_on:
            retry_on += (httpx.RemoteProtocolError,)
        hf_http._DEFAULT_RETRY_ON_EXCEPTIONS = retry_on
        # The default is bound at def time, so updating the module global alone has no effect.
        # http_stream_backoff is a @contextmanager, i.e. its defaults live on the wrapped function.
        patched_funcs = 0
        for func_name in ("_http_backoff_base", "http_backoff", "http_stream_backoff"):
            func = getattr(hf_http, func_name, None)
            func = getattr(func, "__wrapped__", func)
            kwdefaults = getattr(func, "__kwdefaults__", None)
            if kwdefaults and "retry_on_exceptions" in kwdefaults:
                kwdefaults["retry_on_exceptions"] = retry_on
                patched_funcs += 1
        assert patched_funcs == 3, f"patched {patched_funcs} of 3 huggingface_hub backoff functions"
    except Exception as exc:
        logging.warning(f"Could not patch huggingface_hub retry_on_exceptions: {exc}")

    try:
        from huggingface_hub.hf_file_system import HfFileSystemStreamFile

        orig_read = HfFileSystemStreamFile.read

        def read(self, length=-1):
            wait_time = 1.0
            for attempt in range(max_retries):
                try:
                    return orig_read(self, length)
                except (httpx.HTTPError, OSError) as exc:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(
                        f"'{exc}' while reading {getattr(self, 'path', '?')} at offset {self.loc},"
                        f" resuming in {wait_time}s [retry {attempt + 1}/{max_retries - 1}]"
                    )
                    # Force _open_connection() on the next attempt: it sends a Range header from
                    # self.loc, i.e. the read resumes rather than restarting.
                    self.response = None
                    time.sleep(wait_time)
                    wait_time = min(max_wait_time, wait_time * 2)

        HfFileSystemStreamFile.read = read
    except Exception as exc:
        logging.warning(f"Could not patch HfFileSystemStreamFile.read: {exc}")


def _shard_data_files(data_files, *, concurrent: int, task_id: int):
    """
    Round-robin partition ``data_files`` (a list of files, or dict split name -> list of files) into
    ``concurrent`` buckets, and return the bucket for ``task_id`` (1-indexed, as in :class:`Task`).
    """
    if isinstance(data_files, dict):
        return {split: files[task_id - 1 :: concurrent] for split, files in data_files.items()}
    return data_files[task_id - 1 :: concurrent]


class DownloadAndPrepareHuggingFaceDatasetJob(Job):
    """
    https://huggingface.co/docs/datasets/
    https://huggingface.co/datasets

    pip install datasets

    Basically wraps ``datasets.load_dataset(...).save_to_disk(out_dir)``.

    Example for Librispeech:

    DownloadAndPrepareHuggingFaceDatasetJob("librispeech_asr", "clean")
    https://github.com/huggingface/datasets/issues/4179
    """

    __sis_hash_exclude__ = {"split": None, "token": None, "trust_remote_code": None}

    def __init__(
        self,
        path: Union[str, DelayedBase],
        name: Optional[str] = None,
        *,
        data_files: Optional[Any] = None,
        revision: Optional[str] = None,
        split: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        trust_remote_code: Optional[bool] = None,
        time_rqmt: float = 1,
        mem_rqmt: float = 2,
        cpu_rqmt: int = 2,
        mini_task: bool = True,
    ):
        """
        :param path: Path or name of the dataset, parameter passed to :func:`load_dataset`
        :param name: Name of the dataset configuration, parameter passed to :func:`load_dataset`
        :param data_files: Path(s) to the source data file(s), parameter passed to :func:`load_dataset`
        :param revision: Version of the dataset script, parameter passed to :func:`load_dataset`
        :param split: Specifies the split to download e.g "test", parameter passed to :func:`load_dataset`
        :param token: To use as Bearer token for remote files on the Datasets Hub, parameter passed to :func:`load_dataset`
            If set to True, or if unset, it will use the standard HF methods to determine the token.
            E.g. it will look for the HF_TOKEN env var,
            or it will look into the HF home dir (set via HF_HOME env, or as default ~/.cache/huggingface).
            Do ``python -m huggingface_hub.commands.huggingface_cli login``.
            See HF :func:`get_token`.
            You should *not* set some token in public recipes.
        :param trust_remote_code: whether to trust remote code, parameter passed to :func:`load_dataset`
        :param time_rqmt:
        :param mem_rqmt:
        :param cpu_rqmt:
        :param mini_task: the job should be run as mini_task
        """
        super().__init__()
        self.path = path
        self.name = name
        self.data_files = data_files
        self.revision = revision
        self.split = split
        self.token = token
        self.trust_remote_code = trust_remote_code

        self.rqmt = {"cpu": cpu_rqmt, "mem": mem_rqmt, "time": time_rqmt}
        self.mini_task = mini_task

        self.out_dir = self.output_path("dataset", directory=True)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=self.mini_task)

    def run(self):
        import tempfile
        from datasets import load_dataset

        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
            ds = load_dataset(
                instanciate_delayed(self.path),
                self.name,
                data_files=instanciate_delayed(self.data_files),
                revision=self.revision,
                cache_dir=tmp_dir,
                split=self.split,
                token=self.token,
                **({"trust_remote_code": self.trust_remote_code} if self.trust_remote_code is not None else {}),
            )
            print("Dataset:")
            print(ds)

            print("Saving...")
            ds.save_to_disk(self.out_dir.get())

            print("Done.")

    @classmethod
    def hash(cls, kwargs):
        # All other options are ignored for the hash, as they should not have an influence on the result.
        d = {
            "path": kwargs["path"],
            "name": kwargs["name"],
            "data_files": kwargs["data_files"],
            "revision": kwargs["revision"],
            "split": kwargs["split"],
            "token": kwargs["token"],
        }
        return super().hash(d)


class TransformAndMapHuggingFaceDatasetJob(Job):
    """
    Runs some functions (e.g. filtering, mapping, renaming columns, ...) on a HF dataset.

    The map is handled with special logic, as this involves writing to disk.
    We write to the work dir via cache_file_name(s).
    Then we do a save_to_disk to the final output dir.
    Then we clean up the work dir again.
    """

    __sis_hash_exclude__ = {"concurrent": None}

    def __init__(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        *,
        load_dataset_opts: Optional[Dict[str, Any]] = None,  # e.g. "split", "revision", ...
        non_hashed_load_dataset_opts: Optional[Dict[str, Any]] = None,  # e.g. {"num_proc": 8}
        transform: Union[None, TransformFuncT, Sequence[TransformFuncT]] = None,
        map_func: Optional[Callable] = None,
        map_opts: Union[
            None, Dict[str, Any], Callable[[Dataset], Dict[str, Any]], Callable[[DatasetDict], Dict[str, Any]]
        ] = None,
        non_hashed_map_opts: Optional[Dict[str, Any]] = None,
        num_shards: Union[None, int, Dict[str, int]] = None,
        max_shard_size: Union[None, str, int] = None,
        concurrent: Optional[int] = None,
    ):
        """
        :param path: for :func:`datasets.load_dataset`,
            :func:`datasets.Dataset.load_from_disk` or :func:`datasets.DatasetDict.load_from_disk`.
            We automatically detect which one to use.
        :param name: for :func:`datasets.load_dataset`
        :param load_dataset_opts: other options for :func:`datasets.load_dataset`
            or :func:`datasets.Dataset.load_from_disk` or :func:`datasets.DatasetDict.load_from_disk`.
            E.g. "split", "revision", ...
        :param non_hashed_load_dataset_opts: like ``load_dataset_opts``, but not hashed.
            E.g. ``{"num_proc": 8}``.
        :param transform: function or list of functions to transform the dataset
            ((Dataset) -> Dataset or (DatasetDict) -> DatasetDict).
            E.g. filtering, renaming columns, ...
        :param map_func: function to map the dataset examples, or batch of examples.
            This is passed to :func:`datasets.Dataset.map` or :func:`datasets.DatasetDict.map`.
            None (default) means identity.
        :param map_opts: further options passed :func:`datasets.Dataset.map` or :func:`datasets.DatasetDict.map`,
            or a function that returns such options (e.g. depending on the dataset size).
            E.g. ``{"batched": True, "batch_size": 1000}``.
        :param non_hashed_map_opts: like ``map_opts``, but not hashed.
        :param num_shards: how many shards to write via :func:`datasets.Dataset.save_to_disk`
            or :func:`datasets.DatasetDict.save_to_disk`.
            If not given, will be auto-detected based on the dataset size and ``max_shard_size``.
        :param max_shard_size: maximum size of each shard.
            If not given, will use ``"500MB"``.
        :param concurrent: if given, splits ``load_dataset_opts["data_files"]`` (which must be set --
            either a list of files, or a dict mapping split name to a list of files) round-robin into
            this many buckets. Each bucket is processed as its own task (like a SLURM array job) and
            written to its own output dir (``self.out_dirs[i]`` instead of a single ``self.out_dir``),
            so e.g. a huge dataset can be downloaded/processed as many independent, independently
            resumable, and independently progress-trackable jobs instead of one big one -- and the
            resulting per-bucket dirs double as shards for training.
        """
        super().__init__()

        if max_shard_size is not None and num_shards is not None:
            raise ValueError(f"{self}: please specify either max_shard_size or num_shards, but not both.")
        if concurrent is not None:
            assert concurrent > 0, f"{self}: concurrent must be > 0, got {concurrent}"
            assert load_dataset_opts and load_dataset_opts.get("data_files"), (
                f"{self}: concurrent={concurrent} requires load_dataset_opts['data_files']"
                f" (the list/dict of files to split into {concurrent} buckets)"
            )

        self.path = path
        self.name = name
        self.load_dataset_opts = load_dataset_opts
        self.non_hashed_load_dataset_opts = non_hashed_load_dataset_opts
        self.transform = transform
        self.map_func = map_func
        self.map_opts = map_opts
        self.non_hashed_map_opts = non_hashed_map_opts
        self.num_shards = num_shards
        self.max_shard_size = max_shard_size
        self.concurrent = concurrent

        self.rqmt = {"cpu": 16, "mem": 16, "time": 12}

        if concurrent is not None:
            self.out_dirs = {i: self.output_path(f"dataset_shard_{i}", directory=True) for i in range(concurrent)}
        else:
            self.out_dir = self.output_path("dataset", directory=True)

    @classmethod
    def hash(cls, kwargs):
        kwargs.pop("non_hashed_load_dataset_opts")
        kwargs.pop("non_hashed_map_opts")
        return super().hash(kwargs)

    def tasks(self):
        if self.concurrent is not None:
            yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        else:
            yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self, task_id: Optional[int] = None):
        import os
        import shutil
        from datasets import load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
        from datasets.utils.py_utils import convert_file_size_to_int
        from datasets import config

        patch_hf_hub_transient_retries()

        dataset_path = instanciate_delayed(self.path)
        split = None
        load_dataset_opts = (instanciate_delayed(self.load_dataset_opts) or {}).copy()
        load_dataset_opts.update(instanciate_delayed(self.non_hashed_load_dataset_opts) or {})
        if self.name is not None:
            load_dataset_opts["name"] = self.name
        if callable(load_dataset_opts.get("features")):
            # `features` (a datasets.Features/Audio/Value/... object) can't be stored directly as a
            # job attribute: sisyphus's generic object-state reflection (used e.g. to find tk.Path
            # inputs, regardless of whether the containing dict is hashed) walks the whole job
            # __dict__ and fails on the underlying pyarrow DataType objects. So `features` may instead
            # be a plain top-level function that builds and returns the Features object -- exactly
            # like `map_func`/`transform`, which are already handled this way -- called here instead
            # of being stored anywhere.
            load_dataset_opts["features"] = load_dataset_opts["features"]()
        if task_id is not None:
            load_dataset_opts["data_files"] = _shard_data_files(
                load_dataset_opts["data_files"], concurrent=self.concurrent, task_id=task_id
            )
        if "split" in load_dataset_opts:
            split = load_dataset_opts["split"]
        path_ext = f"{dataset_path}/{split}" if split is not None else dataset_path
        ds = None
        if os.path.exists(path_ext):
            if os.path.isdir(path_ext):
                if os.path.exists(f"{path_ext}/{config.DATASET_INFO_FILENAME}") and os.path.exists(
                    f"{path_ext}/{config.DATASET_STATE_JSON_FILENAME}"
                ):
                    load_dataset_opts.pop("split", None)
                    ds = Dataset.load_from_disk(path_ext, **load_dataset_opts)
                elif os.path.exists(f"{path_ext}/{config.DATASETDICT_JSON_FILENAME}"):
                    load_dataset_opts.pop("split", None)
                    ds = DatasetDict.load_from_disk(path_ext, **load_dataset_opts)
            elif path_ext.endswith(".arrow"):
                load_dataset_opts.pop("split", None)
                ds = Dataset.from_file(path_ext, **load_dataset_opts)

        if ds is None:
            # Use load_dataset.
            # That can potentially download the dataset, so make sure that HF_HOME is set.
            assert os.environ.get("HF_HOME"), (
                "HF_HOME env var not set,"
                " set this in your settings.py DEFAULT_ENVIRONMENT_SET"
                " (if not CLEANUP_ENVIRONMENT, otherwise in your current env),"
                " or via job.set_env"
            )

            ds = load_hf_dataset(dataset_path, **load_dataset_opts)
            # A streaming load (load_dataset_opts={"streaming": True}) yields Iterable* here; a
            # transform must materialize it back to Dataset/DatasetDict (e.g. via Dataset.from_generator)
            # before we reach .map()/.save_to_disk() below, which require a map-style dataset.
            assert isinstance(ds, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict))

        if self.transform:
            if callable(self.transform):
                ds = self.transform(ds)
                assert isinstance(ds, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict)), (
                    f"After {self.transform} got {type(ds)}"
                )
            else:
                for func in self.transform:
                    ds = func(ds)
                    assert isinstance(ds, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict)), (
                        f"After {func} got {type(ds)}"
                    )

        assert isinstance(ds, (Dataset, DatasetDict)), (
            f"{self}: dataset must be materialized (Dataset/DatasetDict) before .map()/.save_to_disk(), got {type(ds)}."
            f" If loading via streaming=True, one of the transform funcs must materialize it"
            f" (e.g. via Dataset.from_generator)."
        )

        # We create this tmp dir inside the job work dir,
        # because this might need a lot of space, e.g. several TB, e.g. 2TB for Loquacious,
        # which is often more than what we have available on the local disk (/var/tmp or so).
        # Suffixed by task_id when concurrent (array-job-like) tasks are used: those all run in the
        # same job work dir, so a fixed name would collide/interfere between concurrent task instances.
        work_out_d = "tmp-map-output" if task_id is None else f"tmp-map-output-{task_id}"
        if os.path.exists(work_out_d):
            shutil.rmtree(work_out_d)
        os.makedirs(work_out_d)
        map_opts = self.map_opts
        if callable(map_opts):
            map_opts = map_opts(ds)
        map_extra_opts = {}
        if self.non_hashed_map_opts and "num_proc" in self.non_hashed_map_opts:
            num_proc = self.non_hashed_map_opts["num_proc"]
        else:
            num_proc = self.rqmt["cpu"] * 2
            map_extra_opts["num_proc"] = num_proc
        if self.map_func:
            ds = ds.map(
                self.map_func,
                **(map_opts or {}),
                **(self.non_hashed_map_opts or {}),
                **({"cache_file_name": f"{work_out_d}/data.arrow"} if isinstance(ds, Dataset) else {}),
                **(
                    {"cache_file_names": {k: f"{work_out_d}/data-{k}.arrow" for k in ds.keys()}}
                    if isinstance(ds, DatasetDict)
                    else {}
                ),
                **map_extra_opts,
            )

        num_shards = self.num_shards
        max_shard_size = self.max_shard_size or "500MB"
        max_shard_size = convert_file_size_to_int(max_shard_size)
        if num_shards is None:
            # This code is adapted from Dataset.save_to_disk to determine the number of shards.
            # We make this independent of num_proc (because num_proc is not hashed).
            if isinstance(ds, DatasetDict):
                # noinspection PyProtectedMember
                num_shards = {k: int(ds_._estimate_nbytes() / max_shard_size) + 1 for k, ds_ in ds.items()}
            elif isinstance(ds, Dataset):
                # noinspection PyProtectedMember
                num_shards = int(ds._estimate_nbytes() / max_shard_size) + 1
            else:
                raise TypeError(f"Unexpected type: {type(ds)}")

        out_dir = self.out_dirs[task_id - 1] if task_id is not None else self.out_dir
        ds.save_to_disk(out_dir.get_path(), num_shards=num_shards, num_proc=num_proc)
        del ds
        shutil.rmtree(work_out_d)


class ExtractTextFromHuggingFaceDatasetJob(Job):
    """
    Extract a text column from a HF dataset and write it to a gzipped text file.
    """

    __sis_hash_exclude__ = {"revision": None}

    def __init__(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        *,
        split: Optional[str] = "train",
        column_name: str = "text",
        revision: Optional[str] = None,
    ):
        """
        :param path: for :func:`datasets.load_dataset`
        :param name: for :func:`datasets.load_dataset`
        :param split: for :func:`datasets.load_dataset`
        :param column_name: name of the text column to extract
        :param revision: dataset version (git revision) of the dataset
        """
        super().__init__()
        self.path = path
        self.name = name
        self.split = split
        self.column_name = column_name
        self.revision = revision

        self.rqmt = {"cpu": 4, "mem": 8, "time": 10}

        self.out_text = self.output_path("text.txt.gz")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        import sys
        import gzip
        import time
        from datasets import load_dataset, Dataset

        ds = load_hf_dataset(self.path, name=self.name, split=self.split, revision=self.revision)
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)} {ds}"
        assert self.column_name in ds.column_names, f"Column name {self.column_name} not in columns {ds.column_names}"

        def _hms(s):
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            return "%d:%02d:%02d" % (h, m, s)

        size = ds.num_rows
        start_time = time.monotonic()
        with gzip.open(self.out_text.get_path(), "wt", encoding="utf-8") as f:
            for i, item in enumerate(ds):
                if (i + 1) % 10000 == 0 or i + 1 == size:
                    elapsed = time.monotonic() - start_time
                    speed = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (size - (i + 1)) / speed if speed > 0 else float("inf")
                    eta_str = _hms(eta) if eta != float("inf") else "inf"
                    print(f"Line {i + 1}/{size}, {((i + 1) / size * 100):.1f}%, {speed:.1f} it/s, ETA {eta_str}")
                    sys.stdout.flush()
                f.write(item[self.column_name])
                f.write("\n")
