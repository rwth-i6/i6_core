__all__ = ["WriteCorpusJob"]

from sisyphus import Job, Task

import i6_core.lib.corpus as libcorpus


class WriteCorpusJob(Job):
    """
    Writes the Bliss corpus received as parameter into an output file.
    """

    def __init__(self, corpus: libcorpus.Corpus):
        self.corpus = corpus

        self.out_corpus_file = self.output_path("out.xml.gz")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        self.corpus.dump(self.out_corpus_file.get_path())
