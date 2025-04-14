from enum import Enum
import logging
import shutil
import subprocess
from typing import Any, Dict, Optional

from sisyphus import *

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


class TrainSentencePieceJob(Job):
    """
    Train a sentence-piece model to be used with RETURNN

    See also `https://returnn.readthedocs.io/en/latest/api/datasets.util.vocabulary.html#returnn.datasets.util.vocabulary.SentencePieces`_
    """

    __sis_hash_exclude__ = {"normalization_rule_name": "nmt_nfkc"}

    def __init__(
        self,
        training_text: Path,
        *,
        vocab_size: int,
        model_type: SentencePieceType,
        character_coverage: float = 1.0,
        additional_options: Optional[Dict[str, Any]] = None,
        normalization_rule_name: str = "nmt_nfkc",
    ):
        """

        :param training_text: raw text or gzipped text
        :param vocab_size: target vocabulary size for the created model
        :param model_type: which sentence model to use, use "UNIGRAM" for "typical" SPM
        :param character_coverage: official default is 0.9995, but this caused the least used character to be dropped entirely
        :param additional_options: additional trainer options, see `https://github.com/google/sentencepiece/blob/master/doc/options.md`_
        :param normalization_rule_name: normalization rule name, see `https://github.com/google/sentencepiece/blob/master/doc/normalization.md`_
            "nmt_nfkc" is the sentencepiece default. Set to "nmt_nfkc_cf" to additionally perform case-folding.
        """

        self.training_text = training_text
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.additional_options = additional_options or {}
        self.normalization_rule_name = normalization_rule_name

        self.out_model = self.output_path("spm_out.model")

        self.rqmt = {"cpu": 1, "mem": 2, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sentencepiece

        training_text_path = self.training_text.get_path()
        if training_text_path.endswith(".gz"):
            local_training_text_path = "unzipped_training_text.txt"
            outfile = open(local_training_text_path, "wt")
            subprocess.check_call(["gzip", "-dc", training_text_path], stdout=outfile)
            training_text_path = local_training_text_path

        sentencepiece.SentencePieceTrainer.Train(
            input=training_text_path,
            model_prefix="spm_out",
            model_type=self.model_type.value,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            normalization_rule_name=self.normalization_rule_name,
            **self.additional_options,
        )

        shutil.move("spm_out.model", self.out_model.get_path())
