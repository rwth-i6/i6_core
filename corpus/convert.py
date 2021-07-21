__all__ = ["CorpusToStmJob", "CorpusToTxtJob", "CorpusReplaceOrthFromTxtJob"]

import gzip
import itertools
import pprint

from sisyphus import *

from i6_core.lib import corpus
from i6_core.util import uopen

Path = setup_path(__package__)


class CorpusToStmJob(Job):
    """
    Convert a Bliss corpus into a .stm file
    """

    def __init__(
        self,
        bliss_corpus,
        exclude_non_speech=True,
        remove_punctuation=True,
        fix_whitespace=True,
        name="",
        tag_mapping=(),
    ):
        """

        :param Path bliss_corpus: Bliss corpus
        :param bool exclude_non_speech:
        :param bool remove_punctuation:
        :param bool fix_whitespace:
        :param str name:
        :param tuple[str, dict[str, str]] tag_mapping:
        """
        self.set_vis_name("Extract STM from Corpus")

        self.bliss_corpus = bliss_corpus
        self.exclude_non_speech = exclude_non_speech
        self.remove_punctuation = remove_punctuation
        self.fix_whitespace = fix_whitespace
        self.tag_mapping = tag_mapping
        self.name = name

        self.out_stm_path = self.output_path("%scorpus.stm" % name)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        tag_map = {}

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        all_tags = [
            ("d%d" % i, "default%d" % i, "all other segments of category %d" % i)
            for i in range(len(self.tag_mapping) + 1)
        ]

        for segment in c.segments():
            tag_map[segment.fullname()] = [
                "d%d" % i for i in range(len(self.tag_mapping) + 1)
            ]

        for i, (tag, segments) in enumerate(self.tag_mapping):
            all_tags.append(tag)
            for file in segments.values():
                for segment in uopen(file):
                    if segment.rstrip() in tag_map:
                        tag_map[segment.rstrip()][i] = tag[0]

        with uopen(self.out_stm_path, "wt") as out:
            for segment in c.segments():
                speaker_name = (
                    segment.speaker().name
                    if segment.speaker() is not None
                    else segment.recording.name
                )
                segment_track = segment.track + 1 if segment.track else 1
                out.write(
                    "%s %d %s %5.2f %5.2f <%s> %s\n"
                    % (
                        segment.recording.name,
                        segment_track,
                        speaker_name,
                        segment.start,
                        segment.end,
                        ",".join(tag_map[segment.fullname()]),
                        segment.orth,
                    )
                )
            for tag in all_tags:
                out.write(';; LABEL "%s" "%s" "%s"\n' % tag)


class CorpusToTxtJob(Job):
    """
    Extract orth from a Bliss corpus and store as raw txt or gzipped txt
    """

    def __init__(self, bliss_corpus, segment_file=None, gzip=False):
        """

        :param Path bliss_corpus: Bliss corpus
        :param Path segment_file: segment file
        :param bool gzip: gzip the output text file
        """
        self.set_vis_name("Extract TXT from Corpus")

        self.bliss_corpus = bliss_corpus
        self.gzip = gzip
        self.segment_file = segment_file

        self.out_txt = self.output_path("corpus.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        if self.segment_file:
            with uopen(self.segment_file, "rt") as f:
                segments_whitelist = set(
                    l.strip() for l in f.readlines() if len(l.strip()) > 0
                )
        else:
            segments_whitelist = None

        with uopen(self.out_txt.get_path(), "wt") as f:
            for segment in c.segments():
                if not segments_whitelist or segment.fullname in segments_whitelist:
                    f.write(segment.orth + "\n")


class CorpusReplaceOrthFromTxtJob(Job):
    """
    Merge raw text back into a bliss corpus
    """

    def __init__(self, bliss_corpus, text_file, segment_file=None):
        """
        :param Path bliss_corpus: Bliss corpus
        :param Path text_file: a raw or gzipped text file
        :param Path|None: only replace the segments as specified in the segment file
        """
        self.bliss_corpus = bliss_corpus
        self.text_file = text_file
        self.segment_file = segment_file

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        if self.segment_file:
            with uopen(tk.uncached_path(self.segment_file), "rt") as f:
                segments_whitelist = set(
                    l.strip() for l in f.readlines() if len(l.strip()) > 0
                )
            segment_iterator = filter(lambda s: s in segments_whitelist, c.segments())
        else:
            segment_iterator = c.segments()

        with uopen(self.text_file, "rt") as f:
            for segment, line in itertools.zip_longest(segment_iterator, f):
                assert (
                    segment is not None
                ), "there were more text file lines than segments"
                assert line is not None, "there were less text file lines than segments"
                assert len(line) > 0
                segment.orth = line.strip()

        c.dump(self.out_corpus.get_path())


class CorpusToTextDictJob(Job):
    """
    Extract the Text from a Bliss corpus to fit a "{key: text}" structure (e.g. for RETURNN)
    """

    def __init__(self, bliss_corpus, segment_file=None, invert_match=False):
        """
        :param Path bliss_corpus: bliss corpus file
        :param Path|None segment_file: a segment file as optional whitelist
        :param bool invert_match: use segment file as blacklist (needs to contain full segment names then)
        """
        self.bliss_corpus = bliss_corpus
        self.segment_file = segment_file
        self.invert_match = invert_match

        self.out_dictionary = self.output_path("text_dictionary.py")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        dictionary = {}

        segments = None
        if self.segment_file:
            with uopen(self.segment_file) as f:
                segments = set(line.decode().strip() for line in f)

        for segment in c.segments():
            orth = segment.orth.strip()
            key = segment.fullname()
            if segments:
                if (
                    not self.invert_match
                    and key not in segments
                    and segment.name not in segments
                ):
                    continue
                if self.invert_match and key in segments:
                    continue
            dictionary[key] = orth

        dictionary_string = pprint.pformat(dictionary, width=1000)
        with uopen(self.out_dictionary, "wt") as f:
            f.write(dictionary_string)
