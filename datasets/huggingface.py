"""
https://huggingface.co/docs/datasets/
"""

from typing import Optional, Any, Union
from sisyphus import Job, Task, gs
from sisyphus.delayed_ops import DelayedBase

from i6_core.util import instanciate_delayed


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
