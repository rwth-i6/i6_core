import os
import tempfile
from typing import Optional

from sisyphus import tk, setup_path

from i6_core.tools.compile import MakeJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.recognition import ScliteJob

rel_path = setup_path(__package__)


def compile_sctk(
    branch: Optional[str] = None,
    commit: Optional[str] = None,
    sctk_git_repository: str = "https://github.com/usnistgov/SCTK.git",
) -> tk.Path:
    """
    :param branch: specify a specific branch
    :param commit: specify a specific commit
    :param sctk_git_repository: where to clone SCTK from, usually does not need to be altered
    :return: SCTK binary folder
    """
    sctk_job = CloneGitRepositoryJob(url=sctk_git_repository, branch=branch, commit=commit)
    sctk_job.run()

    sctk_make = MakeJob(
        folder=sctk_job.out_repository,
        make_sequence=["config", "all", "check", "install", "doc"],
        link_outputs={"bin": "bin/"},
    )
    sctk_make.run()
    # This is needed for the compilation to work in the i6 environment, otherwise still untested
    return sctk_make.out_links["bin"]


def test_sclite_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        from sisyphus import gs

        gs.WORK_DIR = tmpdir
        sctk_binary = compile_sctk(branch="v2.4.12")
        hyp = rel_path("files/hyp.ctm")
        ref = rel_path("files/ref.stm")

        sclite_job = ScliteJob(ref=ref, hyp=hyp, sctk_binary_path=sctk_binary)
        sclite_job._sis_setup_directory()
        sclite_job.run()

        assert sclite_job.out_wer.get() == 58.8, "Wrong WER, %s instead of 58.8" % str(sclite_job.out_wer.get())
        assert sclite_job.out_num_errors.get() == 10, "Wrong num errors, %s instead of 10" % str(
            sclite_job.out_num_errors.get()
        )
        assert sclite_job.out_percent_correct.get() == 47.1, "Wrong percent correct, %s instead of 47.1" % str(
            sclite_job.out_percent_correct.get()
        )
        assert sclite_job.out_num_correct.get() == 8, "Wrong num correct, %s instead of 8" % str(
            sclite_job.out_num_correct.get()
        )
        assert (
            sclite_job.out_percent_substitution.get() == 41.2
        ), "Wrong percent substitution, %s instead of 41.2" % str(sclite_job.out_percent_substitution.get())
        assert sclite_job.out_num_substitution.get() == 7, "Wrong num substitution, %s instead of 7" % str(
            sclite_job.out_num_substitution.get()
        )
        assert sclite_job.out_percent_deletions.get() == 11.8, "Wrong percent deletions, %s instead of 11.8" % str(
            sclite_job.out_percent_deletions.get()
        )
        assert sclite_job.out_num_deletions.get() == 2, "Wrong num deletions, %s instead of 2" % str(
            sclite_job.out_num_deletions.get()
        )
        assert sclite_job.out_percent_insertions.get() == 5.9, "Wrong percent insertions, %s instead of 4.5" % str(
            sclite_job.out_percent_insertions.get()
        )
        assert sclite_job.out_num_insertions.get() == 1, "Wrong num insertions, %s instead of 1" % str(
            sclite_job.out_num_insertions.get()
        )
        assert (
            sclite_job.out_percent_word_accuracy.get() == 41.2
        ), "Wrong percent word accuracy, %s instead of 41.2" % str(sclite_job.out_percent_word_accuracy.get())
        assert sclite_job.out_ref_words.get() == 17, "Wrong num ref words, %s instead of 17" % str(
            sclite_job.out_ref_words.get()
        )
        assert sclite_job.out_hyp_words.get() == 16, "Wrong num hyp words, %s instead of 16" % str(
            sclite_job.out_hyp_words.get()
        )
        assert sclite_job.out_aligned_words.get() == 18, "Wrong num aligned words, %s instead of 18" % str(
            sclite_job.out_aligned_words.get()
        )

        # Now test custom precision.

        sclite_job = ScliteJob(ref=ref, hyp=hyp, sctk_binary_path=sctk_binary, precision_ndigit=2)
        sclite_job._sis_setup_directory()
        sclite_job.run()

        assert sclite_job.out_wer.get() == 58.82, "Wrong WER, %s instead of 58.82" % str(sclite_job.out_wer.get())

        assert sclite_job.out_percent_correct.get() == 47.06, "Wrong percent correct, %s instead of 47.1" % str(
            sclite_job.out_percent_correct.get()
        )
        assert (
            sclite_job.out_percent_substitution.get() == 41.18
        ), "Wrong percent substitution, %s instead of 41.18" % str(sclite_job.out_percent_substitution.get())

        assert sclite_job.out_percent_deletions.get() == 11.76, "Wrong percent deletions, %s instead of 11.76" % str(
            sclite_job.out_percent_deletions.get()
        )
        assert sclite_job.out_percent_insertions.get() == 5.88, "Wrong percent insertions, %s instead of 5.88" % str(
            sclite_job.out_percent_insertions.get()
        )
        assert (
            sclite_job.out_percent_word_accuracy.get() == 41.18
        ), "Wrong percent word accuracy, %s instead of 41.18" % str(sclite_job.out_percent_word_accuracy.get())
