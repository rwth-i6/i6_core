__all__ = ["CorpusToStmJob", "CorpusToTxtJob", "CorpusReplaceOrthFromTxtJob"]

import gzip
import itertools
import pprint

from sisyphus import *

from recipe.i6_core.lib import corpus
from recipe.i6_core.util import uopen

Path = setup_path(__package__)


class CorpusToStmJob(Job):
    """
    Convert a Bliss corpus into a .stm file
    """

    def __init__(
        self,
        corpus_path,
        exclude_non_speech=True,
        remove_punctuation=True,
        fix_whitespace=True,
        name="",
        tag_mapping=(),
    ):
        """

        :param Path corpus_path: Bliss corpus
        :param bool exclude_non_speech:
        :param bool remove_punctuation:
        :param bool fix_whitespace:
        :param str name:
        :param tuple[str, dict[str, str]] tag_mapping:
        """
        self.set_vis_name("Extract STM from Corpus")

        self.corpus_path = corpus_path
        self.exclude_non_speech = exclude_non_speech
        self.remove_punctuation = remove_punctuation
        self.fix_whitespace = fix_whitespace
        self.tag_mapping = tag_mapping
        self.name = name

        self.out_stm_path = self.output_path("%scorpus.stm" % name)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        corpus_path = tk.uncached_path(self.corpus_path)
        tag_map = {}

        c = corpus.Corpus()
        c.load(corpus_path)

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
                for segment in open(tk.uncached_path(file)):
                    if segment.rstrip() in tag_map:
                        tag_map[segment.rstrip()][i] = tag[0]

        with open(self.out_stm_path.get_path(), "wt") as out:
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

    def __init__(self, corpus_path, segment_file=None, gzip=False):
        """

        :param Path corpus_path: Bliss corpus
        :param Path segment_file: segment file
        :param bool gzip: gzip the output text file
        """
        self.set_vis_name("Extract TXT from Corpus")

        self.corpus_path = corpus_path
        self.gzip = gzip
        self.segment_file = segment_file

        self.out_txt = self.output_path("corpus.txt" + ".gz" if gzip else "")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        corpus_path = tk.uncached_path(self.corpus_path)
        c = corpus.Corpus()
        c.load(corpus_path)

        if self.segment_file:
            with uopen(tk.uncached_path(self.segment_file), "rt") as f:
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

    def __init__(self, corpus, text_file, segment_file=None):
        """

        :param Path corpus: Bliss corpus
        :param Path text_file: a raw or gzipped text file
        :param Path|None: only replace the segments as specified in the segment file
        """
        self.corpus_path = corpus
        self.text_file = text_file
        self.segment_file = segment_file

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(tk.uncached_path(self.corpus_path))

        if self.segment_file:
            with uopen(tk.uncached_path(self.segment_file), "rt") as f:
                segments_whitelist = set(
                    l.strip() for l in f.readlines() if len(l.strip()) > 0
                )
            segment_iterator = filter(lambda s: s in segments_whitelist, c.segments())
        else:
            segment_iterator = c.segments()

        with uopen(tk.uncached_path(self.text_file), "rt") as f:
            for segment, line in itertools.zip_longest(segment_iterator, f):
                assert (
                    segment is not None
                ), "there were more text file lines than segments"
                assert line is not None, "there were less text file lines than segments"
                assert len(line) > 0
                segment.orth = line.strip()

        c.dump(tk.uncached_path(self.out_corpus))


class CorpusToTextDictJob(Job):
    """
    Extract the Text from a Bliss corpus to fit a "{key: text}" structure (e.g. for RETURNN)
    """

    def __init__(self, corpus, segments=None):
        """
        :param Path corpus: bliss corpus file
        :param Path|None segments: a segment file as optional whitelist
        """
        self.corpus_path = corpus
        self.segments_file_path = segments

        self.out_dictionary = self.output_path("text_dictionary.py")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(tk.uncached_path(self.corpus_path))

        dictionary = {}

        segments = None
        if self.segments_file_path:
            with uopen(self.segments_file_path) as f:
                segments = set(line.decode().strip() for line in f)

        for segment in c.segments():
            orth = segment.orth.strip()
            key = segment.fullname()
            if segments and key not in segments:
                continue
            dictionary[key] = orth

        dictionary_string = pprint.pformat(dictionary, width=1000)
        with uopen(self.out_dictionary.get_path(), "wt") as f:
            f.write(dictionary_string)
