from sisyphus import Job, Task, tk

from i6_core.util import uopen


class LmVocabularyJob(Job):
    def __init__(self, lm_file):
        """
        :param str|Path lm_file: path to the lm arpa file
        """
        self.lm_path = lm_file
        self.out_vocabulary = self.output_path("vocabulary.txt")
        self.out_vocabulary_size = self.output_var("vocabulary_size")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # read language model in ARPA format, adapted from the code in reverse_arpa.py
        # Copyright 2012 Mirko Hannemann BUT, mirko.hannemann@gmail.com
        # Copied from Kaldi: http://sourceforge.net/p/kaldi/code/1640/tree//trunk/egs/wsj/s5/utils/reverse_arpa.py

        lm_path = tk.uncached_path(self.lm_path)
        with uopen(lm_path, "rt", encoding="utf-8") as infile:

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
            self.out_vocabulary_size.set(cngrams[0])

            # read all 1-grams
            n = 1
            while text and "-grams:" not in text:
                text = infile.readline()
            assert n == int(text[1]), "invalid ARPA file: %s" % text

            ngrams = []
            for _ in range(cngrams[0]):
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
                if len(entry) > n + 1:
                    words = entry[1 : n + 1]
                else:
                    words = entry[1:]
                ngram = " ".join(words)
                ngrams.append(ngram)
                text = infile.readline()

        with open(self.out_vocabulary.get_path(), "w") as fout:
            for word in ngrams:
                fout.write("%s\n" % word)
