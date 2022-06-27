from sisyphus import Job, Path, Task

from i6_core.lib import corpus
from i6_core.util import uopen


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
                        "_".join(speaker_name.split()),
                        segment.start,
                        segment.end,
                        ",".join(tag_map[segment.fullname()]),
                        segment.orth,
                    )
                )
            for tag in all_tags:
                out.write(';; LABEL "%s" "%s" "%s"\n' % tag)
