import gzip

from sisyphus import *

import i6_core.util as util


class Lm:
    """
    Interface to access the ngrams of an LM.
    """

    def __init__(self, lm_path):
        self.lm_path = lm_path
        self.ngram_counts = []
        self.ngrams = []
        self.ngrams_start = []
        self.ngrams_end = []
        self.sentprob = 0.0

    @classmethod
    def from_arpa(cls, lm_path):
        """
        Create an Lm object from an arpa file
        :param Path lm_path: path to the arpa file
        """
        # read language model in ARPA format
        lm_path = tk.uncached_path(lm_path)
        with util.uopen(lm_path, "rt", encoding="utf-8") as infile:
            reader = {
                "infile": infile,
                "lineno": 0,
            }

            def read_increase_line():
                reader["lineno"] += 1
                return reader["infile"].readline()

            text = read_increase_line()
            while text and text[:6] != "\\data\\":
                text = read_increase_line()
            assert text, "Invalid ARPA file"

            while text and text[:5] != "ngram":
                text = read_increase_line()

            # get ngram counts
            lm = cls(lm_path)
            n = 0
            while text and text[:5] == "ngram":
                ind = text.split("=")
                counts = int(ind[1].strip())
                r = ind[0].split()
                read_n = int(r[1].strip())
                assert read_n == n + 1, "invalid ARPA file: %s %d %d" % (
                    text.strip(),
                    read_n,
                    n + 1,
                )
                n = read_n
                lm.ngram_counts.append(counts)
                text = read_increase_line()

            # read through the file and find start and end lines for each ngrams order
            for n in range(1, len(lm.ngram_counts) + 1):  # unigrams, bigrams, trigrams
                while text and "-grams:" not in text:
                    text = read_increase_line()
                assert n == int(text[1]), "invalid ARPA file: %s" % text

                lm.ngrams_start.append(reader["lineno"] + 1)
                for ng in range(lm.ngram_counts[n - 1]):
                    while text and len(text.split()) < 2:
                        text = read_increase_line()
                        if not_ngrams(text):
                            break
                    if not_ngrams(text):
                        break  # to deal with incorrect ARPA files
                    text = read_increase_line()
                    if not_ngrams(text):
                        break
                lm.ngrams_end.append(reader["lineno"])

            while text and text[:5] != "\\end\\":
                text = read_increase_line()
            assert text, "invalid ARPA file"

        assert len(lm.ngrams) == len(lm.ngrams_start) == len(lm.ngrams_end)
        for i in range(len(lm.ngrams)):
            assert lm.ngram_counts[i] == (
                lm.ngrams_end[i] - lm.ngrams_start[i] + 1
            ), "Stated %d-gram count is wrong %d != %d" % (
                i + 1,
                lm.ngram_counts[i],
                (lm.ngrams_end[i] - lm.ngrams_start[i] + 1),
            )
        return lm

    def get_ngrams(self, n):
        """
        returns all the ngrams of order n
        """
        return self._read_ngrams(n - 1)

    def _read_ngrams(self, n):
        """
        Read the ngrams knowing start and end lines
        """
        ngrams = dict()
        with util.uopen(self.lm_path, "rt", encoding="utf-8") as infile:
            skip_n_lines(infile, self.ngrams_start[n] - 1)
            i = self.ngrams_start[n] - 1
            while i < self.ngrams_end[n]:
                i += 1
                text = infile.readline()
                entry = text.split()
                prob = float(entry[0])
                if len(entry) > n + 1:
                    back = float(entry[-1])
                    words = entry[1 : n + 1]
                else:
                    back = 0.0
                    words = entry[1:]
                ngram = " ".join(words)
                if (n == 1) and words[0] == "<s>":
                    self.sentprob = prob
                    prob = 0.0
                ngrams[ngram] = (prob, back)

        return ngrams


def skip_n_lines(f_handle, n):
    assert n > 0
    i = 0
    while i < n:
        text = f_handle.readline()
    return text


def not_ngrams(text: str):
    return (not text) or (
        (len(text.split()) == 1) and (("-grams:" in text) or (text[:5] == "\\end\\"))
    )
