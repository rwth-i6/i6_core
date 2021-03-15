__all__ = ["ReverseARPALmJob"]

from sisyphus import *

Path = setup_path(__package__)

import gzip


class ReverseARPALmJob(Job):
    def __init__(self, lm_path):
        self.lm_path = lm_path

        self.reverse_lm_path = self.output_path("reverse.lm.gz", cached=True)

        self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        # Copyright 2012 Mirko Hannemann BUT, mirko.hannemann@gmail.com
        # Copied from Kaldi: http://sourceforge.net/p/kaldi/code/1640/tree//trunk/egs/wsj/s5/utils/reverse_arpa.py
        # Adapted for python 3

        # read language model in ARPA format
        lm_path = tk.uncached_path(self.lm_path)
        open_fun = gzip.open if lm_path.endswith(".gz") else open

        infile = open_fun(lm_path, "rt", encoding="utf-8")
        outfile = gzip.open(self.reverse_lm_path.get_path(), "wt", encoding="utf-8")

        text = infile.readline()
        while text and text[:6] != "\\data\\":
            text = infile.readline()
        assert text, "Invalid ARPA file"

        while text and text[:5] != "ngram":
            text = infile.readline()

        # get ngram counts
        cngrams = []
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
            cngrams.append(counts)
            text = infile.readline()

        # read all n-grams order by order
        sentprob = 0.0  # sentence begin unigram
        ngrams = []
        inf = float("inf")
        for n in range(1, len(cngrams) + 1):  # unigrams, bigrams, trigrams
            while text and "-grams:" not in text:
                text = infile.readline()
            assert n == int(text[1]), "invalid ARPA file: %s" % text

            this_ngrams = {}  # stores all read ngrams
            for ng in range(cngrams[n - 1]):
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
                    sentprob = prob
                    prob = 0.0
                this_ngrams[ngram] = (prob, back)

                for x in range(n - 1, 0, -1):
                    # add all missing backoff ngrams for reversed lm
                    l_ngram = " ".join(words[:x])  # shortened ngram
                    r_ngram = " ".join(
                        words[1 : 1 + x]
                    )  # shortened ngram with offset one
                    if l_ngram not in ngrams[x - 1]:  # create missing ngram
                        ngrams[x - 1][l_ngram] = (0.0, inf)
                    if r_ngram not in ngrams[x - 1]:  # create missing ngram
                        ngrams[x - 1][r_ngram] = (0.0, inf)

                    # add all missing backoff ngrams for forward lm
                    h_ngram = " ".join(words[n - x :])  # shortened history
                    if h_ngram not in ngrams[x - 1]:  # create missing ngram
                        ngrams[x - 1][h_ngram] = (0.0, inf)
                text = infile.readline()
                if (not text) or (
                    (len(text.split()) == 1)
                    and (("-grams:" in text) or (text[:5] == "\\end\\"))
                ):
                    break
            ngrams.append(this_ngrams)

        while text and text[:5] != "\\end\\":
            text = infile.readline()
        assert text, "invalid ARPA file"
        infile.close()

        # compute new reversed ARPA model
        outfile.write("\\data\\\n")
        for n in range(1, len(cngrams) + 1):  # unigrams, bigrams, trigrams
            outfile.write("ngram %d=%d\n" % (n, len(ngrams[n - 1].keys())))
        offset = 0.0
        for n in range(1, len(cngrams) + 1):  # unigrams, bigrams, trigrams
            outfile.write("\\%d-grams:\n" % n)
            for ngram in sorted(ngrams[n - 1]):
                prob = ngrams[n - 1][ngram]
                # reverse word order
                words = ngram.split()
                rstr = " ".join(reversed(words))
                # swap <s> and </s>
                rev_ngram = (
                    rstr.replace("<s>", "<temp>")
                    .replace("</s>", "<s>")
                    .replace("<temp>", "</s>")
                )

                revprob = prob[0]
                if prob[1] != inf:  # only backoff weights from not newly created ngrams
                    revprob = revprob + prob[1]
                # sum all missing terms in decreasing ngram order
                for x in range(n - 1, 0, -1):
                    l_ngram = " ".join(words[:x])  # shortened ngram
                    if l_ngram not in ngrams[x - 1]:
                        sys.stderr.write("%s: not found %s\n" % (rev_ngram, l_ngram))
                    p_l = ngrams[x - 1][l_ngram][0]
                    revprob = revprob + p_l

                    r_ngram = " ".join(
                        words[1 : 1 + x]
                    )  # shortened ngram with offset one
                    if r_ngram not in ngrams[x - 1]:
                        sys.stderr.write("%s: not found %s\n" % (rev_ngram, r_ngram))
                    p_r = ngrams[x - 1][r_ngram][0]
                    revprob = revprob - p_r

                if n != len(cngrams):  # not highest order
                    back = 0.0
                    if (
                        rev_ngram[:3] == "<s>"
                    ):  # special handling since arpa2fst ignores <s> weight
                        if n == 1:
                            offset = revprob  # remember <s> weight
                            revprob = sentprob  # apply <s> weight from forward model
                            back = offset
                        elif n == 2:
                            revprob = (
                                revprob + offset
                            )  # add <s> weight to bigrams starting with <s>
                    if (
                        prob[1] != inf
                    ):  # only backoff weights from not newly created ngrams
                        outfile.write("%g %s %g\n" % (revprob, rev_ngram, back))
                    else:
                        outfile.write("%g %s -100000.0\n" % (revprob, rev_ngram))
                else:  # highest order - no backoff weights
                    if (n == 2) and (rev_ngram[:3] == "<s>"):
                        revprob = revprob + offset
                    outfile.write("%g %s\n" % (revprob, rev_ngram))
        outfile.write("\\end\\\n")
