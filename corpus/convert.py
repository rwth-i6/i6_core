__all__ = ["CorpusToStmJob", "CorpusToTxtJob"]

from recipe.i6_core.lib import corpus

from sisyphus import *

Path = setup_path(__package__)


class CorpusToStmJob(Job):
    def __init__(
        self,
        corpus_path,
        exclude_non_speech=True,
        remove_punctuation=True,
        fix_whitespace=True,
        name="",
        tag_mapping=(),
    ):
        self.set_vis_name("Extract STM from Corpus")

        self.corpus_path = corpus_path
        self.exclude_non_speech = exclude_non_speech
        self.remove_punctuation = remove_punctuation
        self.fix_whitespace = fix_whitespace
        self.tag_mapping = tag_mapping
        self.name = name
        self.stm_path = self.output_path("%scorpus.stm" % name)

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

        with open(self.stm_path.get_path(), "wt") as out:
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
    def __init__(self, corpus_path):
        self.set_vis_name("Extract TXT from Corpus")

        self.corpus_path = corpus_path
        self.txt_path = self.output_path("corpus.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        corpus_path = tk.uncached_path(self.corpus_path)
        c = corpus.Corpus()
        c.load(corpus_path)

        with open(self.txt_path.get_path(), "wt") as out:
            for segment in c.segments():
                out.write(segment.orth + "\n")
