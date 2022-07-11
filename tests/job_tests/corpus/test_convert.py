import filecmp
import os
import tempfile
from sisyphus import setup_path

from i6_core.corpus.convert import CorpusToStmJob

Path = setup_path(__package__)


def test_corpus_to_stm():
    with tempfile.TemporaryDirectory() as tmpdir:
        bliss_corpus = Path("files/test_job.corpus.xml")
        stm_ref = Path("files/test_job.corpus.stm")

        bliss_to_stm_job = CorpusToStmJob(
            bliss_corpus=bliss_corpus,
            exclude_non_speech=True,
            non_speech_tokens=["[non-speech]", "[noise]"],
            remove_punctuation=True,
            punctuation_tokens="!,",
            fix_whitespace=True,
        )
        bliss_to_stm_job.out_stm_path = Path(os.path.join(tmpdir, "corpus.stm"))
        bliss_to_stm_job.run()

        assert filecmp.cmp(
            bliss_to_stm_job.out_stm_path.get_path(), stm_ref.get_path(), shallow=False
        )
