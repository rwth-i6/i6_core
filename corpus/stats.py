__all__ = ["ExtractOovWordsFromCorpusJob", "CountCorpusWordFrequenciesJob", "DumpRecordingAudiosJob"]

from collections import Counter
from contextlib import nullcontext
import logging
from typing import List, Union
import xml.etree.cElementTree as ET

from sisyphus import Job, Task, setup_path, tk

from i6_core.lib.audio import compute_rec_duration
import i6_core.lib.corpus as libcorpus
from i6_core.util import uopen


Path = setup_path(__package__)


class ExtractOovWordsFromCorpusJob(Job):
    """
    Extracts the out of vocabulary words based on a given corpus and lexicon
    """

    __sis_hash_exclude__ = {
        "casing": "upper",
    }

    def __init__(self, bliss_corpus, bliss_lexicon, casing="upper"):
        """
        :param Union[Path, str] bliss_corpus: path to corpus file
        :param Union[Path, str] bliss_lexicon: path to lexicon
        :param str casing: changes the casing of the orthography (options: upper, lower, none)
                                str.upper() is problematic for german since ß -> SS
                                https://bugs.python.org/issue34928
        """
        self.bliss_corpus = bliss_corpus
        self.bliss_lexicon = bliss_lexicon
        self.casing = casing

        self.out_oov_words = self.output_path("oov_words")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        if self.casing != "none":
            logging.warning(
                "The orthography/lemma casing is changed. Is this what you want? Normally you should set this to 'none'. For legacy reasons this is set to 'upper'."
            )

        def change_casing(text_str):
            if self.casing == "upper":
                return text_str.upper()
            elif self.casing == "lower":
                return text_str.lower()
            elif self.casing == "none":
                return text_str
            else:
                raise NotImplementedError

        with uopen(self.bliss_lexicon, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
            iv_words = {change_casing(orth.text) for orth in tree.findall(".//lemma/orth") if orth.text}

        with uopen(self.bliss_corpus, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
            oov_words = {
                w
                for kw in tree.findall(".//recording/segment/orth")
                for w in kw.text.strip().split()
                if change_casing(w) not in iv_words
            }

        with uopen(self.out_oov_words, "wt") as f:
            for w in sorted(oov_words):
                f.write("%s\n" % w)


class CountCorpusWordFrequenciesJob(Job):
    """
    Extracts a list of words and their counts in the provided bliss corpus
    """

    def __init__(self, bliss_corpus):
        """
        :param Path bliss_corpus: path to corpus file
        """
        self.bliss_corpus = bliss_corpus

        self.out_word_counts = self.output_path("counts")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        c = libcorpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        words = Counter()
        for s in c.segments():
            words.update(s.orth.strip().split())

        counts = [(v, k) for k, v in words.items()]
        with uopen(self.out_word_counts, "wt") as f:
            f.write("\n".join("%d\t%s" % t for t in sorted(counts, key=lambda t: (-t[0], t[1]))))


class DumpRecordingAudiosJob(Job):
    """
    Dump all recordings of a given corpus file, one audio per line.
    """

    def __init__(
        self, corpus_files: Union[tk.Path, List[tk.Path]], dump_durations: bool = False, zip_output: bool = False
    ):
        r"""
        :param corpus_file: Corpus file from which to obtain the audio list.
        :param dump_durations: Whether to dump the durations of the audios along with the audio list.
            The `out_audio_durations` output file will contain an audio/duration pair, separated by `\t`.
        :param zip_output: Whether the output should be zipped.
        """
        self.corpus_files = [corpus_files] if isinstance(corpus_files, tk.Path) else corpus_files
        self.dump_durations = dump_durations

        suffix = ".gz" if zip_output else ""
        self.out_audio_list = self.output_path(f"out.txt{suffix}")
        if dump_durations:
            self.out_audio_durations = self.output_path(f"out_durations.txt{suffix}")

        self.rqmt = {"cpu": 1, "mem": 1.0, "time": 1.0}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        with uopen(self.out_audio_list.get_path(), "wt") as f, uopen(
            self.out_audio_durations.get_path(), "wt"
        ) if self.dump_durations else nullcontext() as f_dur:  # fmt: off
            for corpus_file in self.corpus_files:
                c = libcorpus.Corpus()
                c.load(corpus_file.get_path())

                for r in c.all_recordings():
                    f.write(f"{r.audio}\n")
                    if self.dump_durations:
                        f_dur.write(f"{r.audio}\t{compute_rec_duration(r.audio)}\n")
