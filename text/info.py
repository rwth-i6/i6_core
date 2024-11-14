__all__ = ["SentenceLengthHistogramJob"]

from collections import Counter
from typing import Optional

from i6_core.util import uopen
from sisyphus import Job, Task, tk


class SentenceLengthHistogramJob(Job):
    def __init__(self, corpus_path: tk.Path, max_chars: Optional[int] = None, max_words: Optional[int] = None):
        """
        Job computes a histogram of the sentence lengths. `max_chars`/`max_words` can be set to count the number of
        removed lines.

        :param corpus_path: text file path.
        :param max_chars: maximum number of characters per line.
        :param max_words: maximum number of words per line.
        """
        self.corpus_path = corpus_path
        self.max_chars = max_chars
        self.max_words = max_words

        self.out_num_sentences_full_corpus = self.output_var("num_sentences_full_corpus.txt")
        self.out_num_sentences_removed = self.output_var("num_sentences_removed.txt")
        self.out_word_plot = self.output_path("sentence_length_word_histogram.png")
        self.out_char_plot = self.output_path("sentence_length_char_histogram.png")
        self.out_sentences = self.output_path("sentences.txt.gz")

        self.rqmt = {"cpu": 1, "mem": 4, "time": 6}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        counter_words = Counter()
        counter_chars = Counter()
        num_sentences_full_corpus = 0
        num_sentences_removed = 0
        with uopen(self.corpus_path.get_path(), "rt") as f_in, uopen(self.out_sentences.get_path(), "wt") as f_out:
            for line in f_in:
                num_sentences_full_corpus += 1
                line = line.strip()

                num_words = len(line.split())
                num_chars = len(line)

                counter_words[num_words] += 1
                counter_chars[num_chars] += 1

                if (self.max_words is not None or self.max_chars is not None) and (
                    num_words > self.max_words or num_chars > self.max_chars
                ):
                    f_out.write(f"{line}\n")
                    num_sentences_removed += 1

        self.out_num_sentences_full_corpus.set(num_sentences_full_corpus)
        self.out_num_sentences_removed.set(num_sentences_removed)

        self._plot(counter_words, self.out_word_plot.get_path())
        self._plot(counter_chars, self.out_char_plot.get_path())

    @staticmethod
    def _plot(counter: Counter, fig_path: str):
        import matplotlib.pyplot as plt

        x = list(counter.keys())
        y = list(counter.values())
        fig, ax = plt.subplots(layout="constrained")
        ax.set_xlabel("Sentence length")
        ax.set_ylabel("Number of sentences")
        ax.plot(x, y, "s")

        fig.savefig(fig_path, bbox_inches="tight")
