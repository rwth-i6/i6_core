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

    We do a map at the end, which supports to directly save to disk (via cache_file_name(s)),
    without an additional save_to_disk
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
    ):
        super().__init__()

        self.path = path
        self.name = name
        self.load_dataset_opts = load_dataset_opts
        self.non_hashed_load_dataset_opts = non_hashed_load_dataset_opts
        self.transform = transform
        self.map_func = map_func
        self.map_opts = map_opts
        self.non_hashed_map_opts = non_hashed_map_opts

        self.rqmt = {"cpu": 2, "mem": 2, "time": 1}

        self.out_dir = self.output_path("dataset", directory=True)

    @classmethod
    def hash(cls, kwargs):
        kwargs.pop("non_hashed_load_dataset_opts")
        kwargs.pop("non_hashed_map_opts")
        return super().hash(kwargs)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        import os
        from datasets import load_dataset, Dataset, DatasetDict

        assert os.environ.get("HF_HOME"), (
            "HF_HOME env var not set,"
            " set this in your settings.py DEFAULT_ENVIRONMENT_SET"
            " (if not CLEANUP_ENVIRONMENT, otherwise in your current env),"
            " or via job.set_env"
        )

        ds = load_dataset(
            instanciate_delayed(self.path),
            name=self.name,
            **(instanciate_delayed(self.load_dataset_opts) or {}),
            **(instanciate_delayed(self.non_hashed_load_dataset_opts) or {}),
        )
        assert isinstance(ds, (Dataset, DatasetDict))

        if self.transform:
            if callable(self.transform):
                ds = self.transform(ds)
                assert isinstance(ds, (Dataset, DatasetDict)), f"After {self.transform} got {type(ds)}"
            else:
                for func in self.transform:
                    ds = func(ds)
                    assert isinstance(ds, (Dataset, DatasetDict)), f"After {func} got {type(ds)}"

        out_d = self.out_dir.get_path()
        os.makedirs(out_d, exist_ok=True)
        map_opts = self.map_opts
        if callable(map_opts):
            map_opts = map_opts(ds)
        ds.map(
            self.map_func,
            **(map_opts or {}),
            **(self.non_hashed_map_opts or {}),
            **({"cache_file_name": f"{out_d}/data.arrow"} if isinstance(ds, Dataset) else {}),
            **(
                {"cache_file_names": {k: f"{out_d}/data-{k}.arrow" for k in ds.keys()}}
                if isinstance(ds, DatasetDict)
                else {}
            ),
        )
