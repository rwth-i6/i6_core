__all__ = [
    "CorpusAddSpeakerTagsFromMappingJob",
    "CorpusRemoveSpeakerTagsJob",
]

import pickle
import collections
from typing import Dict

from sisyphus import *

from i6_core.lib import corpus

Path = setup_path(__package__)


class CorpusAddSpeakerTagsFromMappingJob(Job):
    """
    Adds speaker tags from given mapping defined by dictonary to corpus
    """

    def __init__(self, corpus: tk.Path, mapping: tk.Path):
        """

        :param corpus: Corpus to add tags to
        :param mapping: pickled dictionary that defines a mapping corpus fullname -> speaker id
        """
        self.corpus = corpus
        self.mapping = mapping

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        bliss = corpus.Corpus()
        bliss.load(self.corpus.get_path())

        with open(self.mapping.get_path(), "rb") as f:
            mapping = pickle.load(f)  # type: Dict

        bliss.speakers = collections.OrderedDict()
        for id in set(mapping.values()):
            speaker = corpus.Speaker()
            speaker.name = str(id)
            bliss.add_speaker(speaker)
        for recording in bliss.all_recordings():
            for segment in recording.segments:
                segment.speaker_name = mapping[segment.fullname()]

        bliss.dump(self.out_corpus.get_path())


class CorpusRemoveSpeakerTagsJob(Job):
    """
    Remove speaker tags from given corpus
    """

    def __init__(self, corpus: tk.Path):
        """

        :param corpus: Corpus to remove the tags from
        """

        self.corpus = corpus
        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        bliss = corpus.Corpus()
        bliss.load(self.corpus.get_path())

        bliss.speakers = {}
        for recording in bliss.all_recordings():
            for segment in recording.segments:
                segment.speaker_name = None
                recording.speaker_name = None

        bliss.dump(self.out_corpus.get_path())
