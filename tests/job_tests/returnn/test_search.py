import tempfile
from sisyphus import setup_path

from i6_core.returnn.search import SearchBPEtoWordsJob
from i6_core.util import check_file_sha256_checksum, compute_file_sha256_checksum

Path = setup_path(__package__)


def test_search_bpe_to_words_single():
    search_out = Path("files/search_out_single")
    reference_word_search_results = Path("files/word_search_results_single.py")
    bpe_to_words_job = SearchBPEtoWordsJob(search_out)
    bpe_to_words_job.out_word_search_results = Path(tempfile.mktemp(".py"))
    bpe_to_words_job.run()

    reference_checksum = compute_file_sha256_checksum(
        reference_word_search_results.get_path()
    )
    check_file_sha256_checksum(
        bpe_to_words_job.out_word_search_results.get_path(), reference_checksum
    )


def test_search_bpe_to_words_nbest():
    search_out = Path("files/search_out_nbest")
    reference_word_search_results = Path("files/word_search_results_nbest.py")
    bpe_to_words_job = SearchBPEtoWordsJob(search_out)
    bpe_to_words_job.out_word_search_results = Path(tempfile.mktemp(".py"))
    bpe_to_words_job.run()

    reference_checksum = compute_file_sha256_checksum(
        reference_word_search_results.get_path()
    )
    check_file_sha256_checksum(
        bpe_to_words_job.out_word_search_results.get_path(), reference_checksum
    )
