from sisyphus import Job, Task

from i6_core.lib.lm import Lm


class VocabularyFromLmJob(Job):
    """
    Extract the vocabulary from an existing LM. Currently supports only arpa files for input.
    """
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
        lm = Lm.from_arpa(self.lm_path)
        self.out_vocabulary_size.set(lm.ngram_counts[0])

        with open(self.out_vocabulary.get_path(), "w") as fout:
            for word in lm.ngrams[0]:
                fout.write("%s\n" % word)
