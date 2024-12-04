__all__ = ["SentenceLengthHistogramJob", "CountLinesJob"]

from collections import Counter
import subprocess
from typing import Optional

from i6_core.util import uopen
from sisyphus import Job, Task, tk


class SentenceLengthHistogramJob(Job):
    def __init__(
        self,
        input_text: tk.Path,
        max_chars: Optional[int] = None,
        max_words: Optional[int] = None,
        full_histogram: bool = True,
    ):
        """
        Job computes a histogram of the sentence lengths. `max_chars`/`max_words` can be set to count the number of
        removed lines.

        :param input_text: text file path.
        :param max_chars: maximum number of characters per line.
            Any line with more characters than this amount won't be added to the final variables.
        :param max_words: maximum number of words per line.
            Any line with more words than this amount won't be added to the final variables.
        :param full_histogram: the removed lines will be shown in the histogram.
        """
        self.input_text = input_text
        self.max_chars = max_chars
        self.max_words = max_words
        self.full_histogram = full_histogram

        self.out_num_sentences_full_corpus = self.output_var("num_sentences_full_corpus.txt")
        self.out_num_sentences_removed = self.output_var("num_sentences_removed.txt")
        self.out_word_plot = self.output_path("sentence_length_word_histogram.png")
        self.out_char_plot = self.output_path("sentence_length_char_histogram.png")
        self.out_removed_sentences = self.output_path("removed_sentences.txt.gz")

        self.rqmt = {"cpu": 1, "mem": 4, "time": 6}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        counter_words = Counter()
        counter_chars = Counter()
        num_sentences_full_corpus = 0
        num_sentences_removed = 0
        with uopen(self.input_text.get_path(), "rt") as f_in, uopen(
            self.out_removed_sentences.get_path(), "wt"
        ) as f_out:
            for line in f_in:
                num_sentences_full_corpus += 1
                line = line.strip()

                num_words = len(line.split())
                num_chars = len(line)

                counter_words[num_words] += 1
                counter_chars[num_chars] += 1

                if (self.max_words is not None and num_words > self.max_words) or (
                    self.max_chars is not None and num_chars > self.max_chars
                ):
                    f_out.write(f"{line}\n")
                    num_sentences_removed += 1

        self.out_num_sentences_full_corpus.set(num_sentences_full_corpus)
        self.out_num_sentences_removed.set(num_sentences_removed)

        if not self.full_histogram:
            counter_words = Counter({k: v for k, v in counter_words.items() if k <= self.max_words})
            counter_chars = Counter({k: v for k, v in counter_chars.items() if k <= self.max_chars})

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


class CountLinesJob(Job):
    def __init__(self, input_text: tk.Path):
        self.input_text = input_text

        self.out_num_lines = self.output_var("num_lines")

        self.rqmt = {"cpu": 1, "mem": 1, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        zcat_cmd = ["zcat", "-f", self.input_text.get_path()]
        zcat_res = subprocess.run(zcat_cmd, check=True, capture_output=True)

        wc_cmd = ["wc", "-l"]
        wc_res = subprocess.run(wc_cmd, input=zcat_res.stdout, check=True, capture_output=True)

        self.out_num_sentences.set(int(wc_res.stdout.decode("utf-8").strip()))
