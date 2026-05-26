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
        concurrent: int = 1,
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
        :param concurrent: number of Sisyphus workers to use for the transform/map step.
        """
        super().__init__()

        if max_shard_size is not None and num_shards is not None:
            raise ValueError(f"{self}: please specify either max_shard_size or num_shards, but not both.")
        if concurrent < 1:
            raise ValueError(f"{self}: concurrent must be at least 1, got {concurrent!r}.")

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
        self.merge_rqmt = {"cpu": 1, "mem": 16, "time": 4}

        self.out_dir = self.output_path("dataset", directory=True)

    @classmethod
    def hash(cls, kwargs):
        kwargs.pop("non_hashed_load_dataset_opts")
        kwargs.pop("non_hashed_map_opts")
        kwargs.pop("concurrent", None)
        return super().hash(kwargs)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        if self.concurrent > 1:
            yield Task("merge", resume="merge", rqmt=self.merge_rqmt)

    def _load_dataset(self):
        import os
        from datasets import load_dataset, Dataset, DatasetDict
        from datasets import config

        dataset_path = instanciate_delayed(self.path)
        split = None
        load_dataset_opts = (instanciate_delayed(self.load_dataset_opts) or {}).copy()
        load_dataset_opts.update(instanciate_delayed(self.non_hashed_load_dataset_opts) or {})
        if self.name is not None:
            load_dataset_opts["name"] = self.name
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

            ds = load_dataset(dataset_path, **load_dataset_opts)
            assert isinstance(ds, (Dataset, DatasetDict))

        return ds

    def _apply_transform(self, ds):
        from datasets import Dataset, DatasetDict

        if self.transform:
            if callable(self.transform):
                ds = self.transform(ds)
                assert isinstance(ds, (Dataset, DatasetDict)), f"After {self.transform} got {type(ds)}"
            else:
                for func in self.transform:
                    ds = func(ds)
                    assert isinstance(ds, (Dataset, DatasetDict)), f"After {func} got {type(ds)}"

        return ds

    def _shard_dataset(self, ds, task_id: int):
        from datasets import Dataset, DatasetDict

        if self.concurrent == 1:
            return ds
        shard_index = task_id - 1
        if isinstance(ds, Dataset):
            return ds.shard(num_shards=self.concurrent, index=shard_index, contiguous=True)
        if isinstance(ds, DatasetDict):
            return DatasetDict(
                {
                    key: dataset.shard(num_shards=self.concurrent, index=shard_index, contiguous=True)
                    for key, dataset in ds.items()
                }
            )
        raise TypeError(f"Unexpected type: {type(ds)}")

    def _get_map_opts(self, ds):
        map_opts = self.map_opts
        if callable(map_opts):
            map_opts = map_opts(ds)
        return map_opts or {}

    def _get_map_num_proc(self):
        map_extra_opts = {}
        if self.non_hashed_map_opts and "num_proc" in self.non_hashed_map_opts:
            num_proc = self.non_hashed_map_opts["num_proc"]
        else:
            num_proc = self.rqmt["cpu"] * 2
            map_extra_opts["num_proc"] = num_proc
        return num_proc, map_extra_opts

    def _get_num_shards(self, ds, num_proc: int):
        from datasets import Dataset, DatasetDict
        from datasets.utils.py_utils import convert_file_size_to_int

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
        return num_shards, num_proc

    def _worker_out_dir(self, task_id: int) -> str:
        return f"worker-{task_id}-dataset"

    def run(self, task_id: int):
        import os
        import shutil
        from datasets import Dataset, DatasetDict

        assert 1 <= task_id <= self.concurrent
        ds = self._load_dataset()
        ds = self._apply_transform(ds)
        map_opts = self._get_map_opts(ds)
        ds = self._shard_dataset(ds, task_id)

        # We create this tmp dir inside the job work dir,
        # because this might need a lot of space, e.g. several TB, e.g. 2TB for Loquacious,
        # which is often more than what we have available on the local disk (/var/tmp or so).
        work_out_d = f"tmp-map-output-{task_id}"
        if os.path.exists(work_out_d):
            shutil.rmtree(work_out_d)
        os.makedirs(work_out_d)
        num_proc, map_extra_opts = self._get_map_num_proc()
        if self.map_func:
            ds = ds.map(
                self.map_func,
                **map_opts,
                **(self.non_hashed_map_opts or {}),
                **({"cache_file_name": f"{work_out_d}/data-{task_id}.arrow"} if isinstance(ds, Dataset) else {}),
                **(
                    {"cache_file_names": {k: f"{work_out_d}/data-{task_id}-{k}.arrow" for k in ds.keys()}}
                    if isinstance(ds, DatasetDict)
                    else {}
                ),
                **map_extra_opts,
            )

        if self.concurrent > 1:
            out_dir = self._worker_out_dir(task_id)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
        else:
            out_dir = self.out_dir.get_path()

        num_shards, num_proc = self._get_num_shards(ds, num_proc)
        ds.save_to_disk(out_dir, num_shards=num_shards, num_proc=num_proc)
        del ds
        shutil.rmtree(work_out_d)

    def merge(self):
        import os
        import shutil
        from datasets import Dataset, DatasetDict, concatenate_datasets

        worker_dirs = [self._worker_out_dir(task_id) for task_id in range(1, self.concurrent + 1)]
        datasets = []
        for worker_dir in worker_dirs:
            if not os.path.exists(worker_dir):
                raise FileNotFoundError(f"Missing worker output directory {worker_dir!r}")
            datasets.append(self._load_dataset_from_disk(worker_dir))

        first_ds = datasets[0]
        if isinstance(first_ds, Dataset):
            ds = concatenate_datasets(datasets)
        elif isinstance(first_ds, DatasetDict):
            ds = DatasetDict(
                {
                    key: concatenate_datasets([worker_ds[key] for worker_ds in datasets])
                    for key in first_ds.keys()
                }
            )
        else:
            raise TypeError(f"Unexpected type: {type(first_ds)}")

        if os.path.exists(self.out_dir.get_path()):
            shutil.rmtree(self.out_dir.get_path())
        num_proc, _ = self._get_map_num_proc()
        num_shards, num_proc = self._get_num_shards(ds, num_proc)
        ds.save_to_disk(self.out_dir.get_path(), num_shards=num_shards, num_proc=num_proc)

    @staticmethod
    def _load_dataset_from_disk(path):
        import os
        from datasets import Dataset, DatasetDict
        from datasets import config

        if os.path.exists(f"{path}/{config.DATASET_INFO_FILENAME}") and os.path.exists(
            f"{path}/{config.DATASET_STATE_JSON_FILENAME}"
        ):
            return Dataset.load_from_disk(path)
        if os.path.exists(f"{path}/{config.DATASETDICT_JSON_FILENAME}"):
            return DatasetDict.load_from_disk(path)
        raise FileNotFoundError(f"Could not find a saved Hugging Face dataset in {path!r}")


class ExtractTextFromHuggingFaceDatasetJob(Job):
    """
    Extract a text column from a HF dataset and write it to a gzipped text file.
    """

    def __init__(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        *,
        split: Optional[str] = "train",
        column_name: str = "text",
    ):
        """
        :param path: for :func:`datasets.load_dataset`
        :param name: for :func:`datasets.load_dataset`
        :param split: for :func:`datasets.load_dataset`
        :param column_name: name of the text column to extract
        """
        super().__init__()
        self.path = path
        self.name = name
        self.split = split
        self.column_name = column_name

        self.rqmt = {"cpu": 4, "mem": 8, "time": 10}

        self.out_text = self.output_path("text.txt.gz")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        import sys
        import gzip
        import time
        from datasets import load_dataset, Dataset

        ds = load_dataset(self.path, self.name, split=self.split)
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
