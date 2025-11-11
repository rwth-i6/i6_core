__all__ = ["CreateEmptyCorpusJob"]

from sisyphus import Job, Task

import i6_core.lib.corpus as libcorpus


class CreateEmptyCorpusJob(Job):
    """
    Creates an empty Bliss corpus.
    """

    def __init__(self):
        self.out_empty_corpus = self.output_path("out.xml.gz")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        empty_corpus = libcorpus.Corpus()
        empty_corpus.dump(self.out_empty_corpus.get_path())
