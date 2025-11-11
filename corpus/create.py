__all__ = ["CreateEmptyCorpusJob"]

from sisyphus import Job, Task

import i6_core.lib.corpus as libcorpus


class CreateEmptyCorpusJob(Job):
    """
    Creates an empty Bliss corpus.
    """

    def __init__(self):
        self.out_empty_corpus = self.output_path("out.xml.gz")

        self.rqmt = None

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=self.rqmt is None)

    def run(self):
        empty_corpus = libcorpus.Corpus()
        empty_corpus.dump(self.out_empty_corpus.get_path())
