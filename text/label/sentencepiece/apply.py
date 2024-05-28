__all__ = ["ApplySentencePieceJob"]


from enum import Enum
import logging
import shutil
import subprocess

from sisyphus import *

from i6_core.util import uopen

try:
    import sentencepiece
except ImportError:
    if not hasattr(gs, "WARNING_NO_SENTENCEPIECE") or gs.WARNING_NO_SENTENCEPIECE is True:
        logging.warning(
            "The package 'sentencepiece' is not installed in the manager python env. Please make sure it is installed "
            "in the python environment running the Sisyphus worker. To suppress this warning set "
            "'WARNING_NO_SENTENCEPIECE=False' in the settings.py"
        )


class SentencePieceType(Enum):
    UNIGRAM = "unigram"
    BPE = "bpe"
    CHAR = "char"
    WORD = "word"


class ApplySentencePieceJob(Job):
    """
    Train a sentence-piece model to be used with RETURNN

    See also `https://returnn.readthedocs.io/en/latest/api/datasets.util.vocabulary.html#returnn.datasets.util.vocabulary.SentencePieces`_
    """

    def __init__(
        self,
        text_file,
        model,
        output_format=None,
        nbest_size=None,
        alpha=None,
        extra_options=None,
    ):
        """

        :param tk.Path training_text: raw text or gzipped text
        :param int vocab_size: target vocabulary size for the created model
        :param SentencePieceType model_type: which sentence model to use, use "UNIGRAM" for "typical" SPM
        :param float character_coverage: official default is 0.9995, but this caused the least used character to be dropped entirely
        :param dict|None additional_options: additional trainer options, see `https://github.com/google/sentencepiece/blob/master/doc/options.md`_
        """

        self.text_file = text_file
        self.model = model
        self.output_format = output_format
        self.nbest_size = nbest_size
        self.alpha = alpha
        self.extra_options = extra_options or {}

        self.out_text = self.output_path("processed.text.gz")

        self.rqmt = {"cpu": 1, "mem": 2, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sentencepiece

        text_path = self.text_file.get_path()
        if text_path.endswith(".gz"):
            local_text_path = "unzipped_text.txt"
            outfile = open(local_text_path, "wt")
            subprocess.check_call(["gzip", "-dc", text_path], stdout=outfile)
            text_path = local_text_path

        sp_ctrl = sentencepiece.SentencePieceProcessor()
        sp_ctrl.load(self.model.get_path())

        with uopen(text_path, "rt") as in_text, uopen(self.out_text, "wt") as out_text:
            for sentence in in_text:
                out_text.write(" ".join(sp_ctrl.encode_as_pieces(sentence)) + "\n")

        """

        sentencepiece.SentencePieceTrainer.Train(
            input=training_text_path,
            model_prefix="spm_out",
            model_type=self.model_type.value,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            **self.additional_options,
        )
        shutil.move("spm_out.model", self.out_model.get_path())
        """
