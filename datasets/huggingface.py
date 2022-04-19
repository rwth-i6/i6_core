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

        self.out_dir = self.output_path("dataset")

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
        import tempfile
        import datasets

        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:

            ds = datasets.load_dataset(
                self.path,
                self.name,
                cache_dir=tmp_dir,
            )
            print("Dataset:")
            print(ds)

            print("Saving...")
            ds.save_to_disk(self.out_dir.get())

            print("Done.")
