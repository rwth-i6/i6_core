import gzip

from sisyphus import *


class Lm:
    """
    Interface to access the ngrams of an LM.
    """

    def __init__(self):
        self.ngram_counts = []
        self.ngrams = []
        self.sentprob = 0.0

    @classmethod
    def from_arpa(cls, lm_path):
        """
        Create an Lm object from an arpa file
        :param Path lm_path: path to the arpa file
        """
        # read language model in ARPA format
        lm_path = tk.uncached_path(lm_path)
        open_fun = gzip.open if lm_path.endswith(".gz") else open

        with open_fun(lm_path, "rt", encoding="utf-8") as infile:
            text = infile.readline()
            while text and text[:6] != "\\data\\":
                text = infile.readline()
            assert text, "Invalid ARPA file"

            while text and text[:5] != "ngram":
                text = infile.readline()

            # get ngram counts
            lm = cls()
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
                text = infile.readline()

            # read all n-grams order by order
            inf = float("inf")
            for n in range(1, len(lm.ngram_counts) + 1):  # unigrams, bigrams, trigrams
                while text and "-grams:" not in text:
                    text = infile.readline()
                assert n == int(text[1]), "invalid ARPA file: %s" % text

                this_ngrams = {}  # stores all read ngrams
                for ng in range(lm.ngram_counts[n - 1]):
                    while text and len(text.split()) < 2:
                        text = infile.readline()
                        if (not text) or (
                            (len(text.split()) == 1)
                            and (("-grams:" in text) or (text[:5] == "\\end\\"))
                        ):
                            break
                    if (not text) or (
                        (len(text.split()) == 1)
                        and (("-grams:" in text) or (text[:5] == "\\end\\"))
                    ):
                        break  # to deal with incorrect ARPA files
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
                        lm.sentprob = prob
                        prob = 0.0
                    this_ngrams[ngram] = (prob, back)

                    for x in range(n - 1, 0, -1):
                        # add all missing backoff ngrams for reversed lm
                        l_ngram = " ".join(words[:x])  # shortened ngram
                        r_ngram = " ".join(
                            words[1 : 1 + x]
                        )  # shortened ngram with offset one
                        if l_ngram not in lm.ngrams[x - 1]:  # create missing ngram
                            lm.ngrams[x - 1][l_ngram] = (0.0, inf)
                        if r_ngram not in lm.ngrams[x - 1]:  # create missing ngram
                            lm.ngrams[x - 1][r_ngram] = (0.0, inf)

                        # add all missing backoff ngrams for forward lm
                        h_ngram = " ".join(words[n - x :])  # shortened history
                        if h_ngram not in lm.ngrams[x - 1]:  # create missing ngram
                            lm.ngrams[x - 1][h_ngram] = (0.0, inf)
                    text = infile.readline()
                    if (not text) or (
                        (len(text.split()) == 1)
                        and (("-grams:" in text) or (text[:5] == "\\end\\"))
                    ):
                        break
                lm.ngrams.append(this_ngrams)

            while text and text[:5] != "\\end\\":
                text = infile.readline()
            assert text, "invalid ARPA file"

        assert len(lm.ngrams) == len(lm.ngram_counts)
        for i in range(len(lm.ngrams)):
            assert len(lm.ngrams[i]) == lm.ngram_counts[i], "Stated n-gram count is wrong"
        return lm
