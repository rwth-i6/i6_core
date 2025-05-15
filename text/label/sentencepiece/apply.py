import logging
import shutil
import subprocess as sp
import tempfile
import os

from sisyphus import Job, Task, tk, gs
from typing import Any, Dict, Optional

import i6_core.util as util

try:
    import sentencepiece
except ImportError:
    if not hasattr(gs, "WARNING_NO_SENTENCEPIECE") or gs.WARNING_NO_SENTENCEPIECE is True:
        logging.warning(
            "The package 'sentencepiece' is not installed in the manager python env. Please make sure it is installed "
            "in the python environment running the Sisyphus worker. To suppress this warning set "
            "'WARNING_NO_SENTENCEPIECE=False' in the settings.py"
        )


class ApplySentencepieceToTextJob(Job):
    """
    Apply sentencepiece model on a text file, basically a wrapper for spm.encode
    """

    def __init__(
        self,
        *,
        text_file: tk.Path,
        sentencepiece_model: tk.Path,
        enable_unk: bool = True,
        gzip_output: bool = True,
    ):
        """
        :param text_file: words text file to convert to sentencepiece
        :param sentencepiece_model: path to the trained sentencepiece model
        :param enable_unk: whether enable unk to map OOV symbol to the unknown symbol set in training or keep it as is
        :param gzip_output: use gzip on the output text
        """
        self.text_file = text_file
        self.sentencepiece_model = sentencepiece_model
        self.enable_unk = enable_unk
        self.rqmt: Optional[Dict[str, Any]] = {"cpu": 1, "mem": 2, "time": 2}

        self.out_sentencepiece_text = self.output_path(
            "words_to_sentencepiece.txt.gz" if gzip_output else "words_to_sentencepiece.txt"
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, mini_task=self.rqmt is None)

    def run(self):
        import sentencepiece

        spm = sentencepiece.SentencePieceProcessor(model_file=self.sentencepiece_model.get_path())
        if self.enable_unk:
            spm.SetEncodeExtraOptions("unk")

        with util.uopen(self.text_file, "rt") as fin, util.uopen(self.out_sentencepiece_text, "wt") as fout:
            for line in fin:
                pieces = spm.encode(line.rstrip("\n"), out_type=str)
                fout.write(" ".join(pieces) + "\n")

