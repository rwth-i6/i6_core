import os
import tempfile
from typing import Optional

from sisyphus import tk, setup_path

from i6_core.tools.compile import MakeJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.recognition import ScliteJob

Path = setup_path(__package__)


def compile_sctk(
    branch: Optional[str] = None,
    commit: Optional[str] = None,
    sctk_git_repository: str = "https://github.com/usnistgov/SCTK.git",
    tmpdir=None,
) -> tk.Path:
    """
    :param branch: specify a specific branch
    :param commit: specify a specific commit
    :param sctk_git_repository: where to clone SCTK from, usually does not need to be altered
    :return: SCTK binary folder
    """
    sctk_job = CloneGitRepositoryJob(url=sctk_git_repository, branch=branch, commit=commit)
    sctk_job.out_repository = Path(os.path.join(tmpdir, "sctk_git"))
    sctk_job.run()

    sctk_make = MakeJob(
        folder=sctk_job.out_repository,
        make_sequence=["config", "all", "check", "install", "doc"],
        link_outputs={"bin": "bin/"},
    )
    sctk_make.out_repository = Path(os.path.join(tmpdir, "sctk_compiled"))
    sctk_make.out_links["bin"] = Path(os.path.join(tmpdir, "sctk_bin"))
    sctk_make.run()
    # This is needed for the compilation to work in the i6 environment, otherwise still untested
    sctk_make._sis_environment.set("CPPFLAGS", "-std=c++11")
    return sctk_make.out_links["bin"]


def test_sclite_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        sctk_binary = compile_sctk(tmpdir=tmpdir, branch="v2.4.12")
        hyp = Path("files/hyp.ctm")
        ref = Path("files/ref.stm")

        sclite_job = ScliteJob(ref=ref, hyp=hyp, sctk_binary_path=sctk_binary)

        sclite_job.out_report_dir = Path(os.path.join(tmpdir, "reports"))
        os.makedirs(os.path.join(tmpdir, "reports"))
        sclite_job._sis_output_dirs.add(sclite_job.out_report_dir)
        sclite_job.out_wer = tk.Variable(os.path.join(tmpdir, "wer"))
        sclite_job.out_num_errors = tk.Variable(os.path.join(tmpdir, "num_errors"))
        sclite_job.out_percent_correct = tk.Variable(os.path.join(tmpdir, "percent_correct"))
        sclite_job.out_num_correct = tk.Variable(os.path.join(tmpdir, "num_correct"))
        sclite_job.out_percent_substitution = tk.Variable(os.path.join(tmpdir, "percent_substitution"))
        sclite_job.out_num_substitution = tk.Variable(os.path.join(tmpdir, "num_substitution"))
        sclite_job.out_percent_deletions = tk.Variable(os.path.join(tmpdir, "percent_deletions"))
        sclite_job.out_num_deletions = tk.Variable(os.path.join(tmpdir, "num_deletions"))
        sclite_job.out_percent_insertions = tk.Variable(os.path.join(tmpdir, "percent_insertions"))
        sclite_job.out_num_insertions = tk.Variable(os.path.join(tmpdir, "num_insertions"))
        sclite_job.out_percent_word_accuracy = tk.Variable(os.path.join(tmpdir, "percent_word_accuracy"))
        sclite_job.out_ref_words = tk.Variable(os.path.join(tmpdir, "ref_words"))
        sclite_job.out_hyp_words = tk.Variable(os.path.join(tmpdir, "hyp_words"))
        sclite_job.out_aligned_words = tk.Variable(os.path.join(tmpdir, "aligned_words"))
        sclite_job.run()

        assert sclite_job.out_wer.get() == 68.2, "Wrong WER, %s instead of 68.2" % str(sclite_job.out_wer.get())
        assert sclite_job.out_num_errors.get() == 15, "Wrong num errors, %s instead of 15" % str(
            sclite_job.out_num_errors.get()
        )
        assert sclite_job.out_percent_correct.get() == 36.4, "Wrong percent correct, %s instead of 36.4" % str(
            sclite_job.out_percent_correct.get()
        )
        assert sclite_job.out_num_correct.get() == 8, "Wrong num correct, %s instead of 8" % str(
            sclite_job.out_num_correct.get()
        )
        assert (
            sclite_job.out_percent_substitution.get() == 40.9
        ), "Wrong percent substitution, %s instead of 40.9" % str(sclite_job.out_percent_substitution.get())
        assert sclite_job.out_num_substitution.get() == 9, "Wrong num substitution, %s instead of 9" % str(
            sclite_job.out_num_substitution.get()
        )
        assert sclite_job.out_percent_deletions.get() == 22.7, "Wrong percent deletions, %s instead of 22.7" % str(
            sclite_job.out_percent_deletions.get()
        )
        assert sclite_job.out_num_deletions.get() == 5, "Wrong num deletions, %s instead of 5" % str(
            sclite_job.out_num_deletions.get()
        )
        assert sclite_job.out_percent_insertions.get() == 4.5, "Wrong percent insertions, %s instead of 4.5" % str(
            sclite_job.out_percent_insertions.get()
        )
        assert sclite_job.out_num_insertions.get() == 1, "Wrong num insertions, %s instead of 1" % str(
            sclite_job.out_num_insertions.get()
        )
        assert (
            sclite_job.out_percent_word_accuracy.get() == 31.8
        ), "Wrong percent word accuracy, %s instead of 31.8" % str(sclite_job.out_percent_word_accuracy.get())
        assert sclite_job.out_ref_words.get() == 22, "Wrong num ref words, %s instead of 22" % str(
            sclite_job.out_ref_words.get()
        )
        assert sclite_job.out_hyp_words.get() == 18, "Wrong num hyp words, %s instead of 18" % str(
            sclite_job.out_hyp_words.get()
        )
        assert sclite_job.out_aligned_words.get() == 23, "Wrong num aligned words, %s instead of 23" % str(
            sclite_job.out_aligned_words.get()
        )
