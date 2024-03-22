import filecmp
import os
import tempfile
from sisyphus import setup_path

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lexicon.bpe import CreateBPELexiconJob

Path = setup_path(__package__)


def test_create_bpe_lexicon():
    with tempfile.TemporaryDirectory() as tmpdir:
        lexicon = Path("files/lexicon.xml")

        bpe_codes = Path("files/bpe.codes")
        bpe_vocab = Path("files/bpe.vocab")

        subword_nmt_job = CloneGitRepositoryJob(
            url="https://github.com/rwth-i6/subword-nmt",
            checkout_folder_name="subword-nmt",
        )
        subword_nmt_job.out_repository = Path(os.path.join(tmpdir, "subword_nmt_repo"))
        subword_nmt_job.run()

        create_bpe_lexicon_job = CreateBPELexiconJob(
            base_lexicon_path=lexicon,
            bpe_codes=bpe_codes,
            bpe_vocab=bpe_vocab,
            subword_nmt_repo=subword_nmt_job.out_repository,
            unk_label="<unk>",
            vocab_blacklist={"<s>", "</s>"},
            keep_special_lemmas=True,
        )
        create_bpe_lexicon_job.out_lexicon = Path(os.path.join(tmpdir, "out_lexicon.xml"))
        create_bpe_lexicon_job.run()
        assert filecmp.cmp(
            create_bpe_lexicon_job.out_lexicon.get_path(), Path("files/out_lexicon_with_special.xml"), shallow=False
        )

        create_bpe_lexicon_job = CreateBPELexiconJob(
            base_lexicon_path=lexicon,
            bpe_codes=bpe_codes,
            bpe_vocab=bpe_vocab,
            subword_nmt_repo=subword_nmt_job.out_repository,
            unk_label="<unk>",
            vocab_blacklist={"<s>", "</s>"},
            keep_special_lemmas=False,
        )
        create_bpe_lexicon_job.out_lexicon = Path(os.path.join(tmpdir, "out_lexicon.xml"))
        create_bpe_lexicon_job.run()
        assert filecmp.cmp(
            create_bpe_lexicon_job.out_lexicon.get_path(), Path("files/out_lexicon_without_special.xml"), shallow=False
        )
