"""
https://huggingface.co/docs/datasets/
"""

from typing import Optional
from sisyphus import *


class DownloadAndPrepareHuggingFaceDatasetJob(Job):
    """
    https://huggingface.co/docs/datasets/
    https://huggingface.co/datasets

    pip install datasets

    Example for Librispeech:

    DownloadAndPrepareHuggingFaceDatasetJob("librispeech_asr", "clean")
    https://github.com/huggingface/datasets/issues/4179
    """

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
    ):
        """
        :param dataset_name:
            https://huggingface.co/datasets
        """
        super().__init__()
        self.path = path
        self.name = name

        self.out_cache_dir = self.output_path("cache_dir")
        self.out_data_dir = self.output_path("data_dir")

    @classmethod
    def hash(cls, kwargs):
        # All other options are ignored for the hash, as they should not have an influence on the result.
        d = {
            "path": kwargs["path"],
            "name": kwargs["name"],
        }
        return super().hash(d)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        import datasets

        ds = datasets.load_dataset(
            self.path,
            self.name,
            data_dir=self.out_data_dir.get(),
            cache_dir=self.out_cache_dir.get(),
        )
        print("Dataset:")
        print(ds)
