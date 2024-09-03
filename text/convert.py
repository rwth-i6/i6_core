__all__ = [
    "TextDictToTextLinesJob",
    "TextDictToStmJob",
]

from typing import Union, Sequence, Dict, Tuple
import re
from sisyphus import Job, Path, Task
from i6_core.util import parse_text_dict, uopen


class TextDictToTextLinesJob(Job):
    """
    Operates on RETURNN search output (see :mod:`i6_core.returnn.search`)
    or :class:`CorpusToTextDictJob` output and prints the values line-by-line.
    The ordering from the dict is preserved.
    """

    def __init__(self, text_dict: Path, *, gzip: bool = False):
        """
        :param text_dict: a text file with a dict in python format, {seq_tag: text}
        :param gzip: if True, gzip the output
        """
        self.text_dict = text_dict
        self.out_text_lines = self.output_path("text_lines.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # nan/inf should not be needed, but avoids errors at this point and will print an error below,
        # that we don't expect an N-best list here.
        d = parse_text_dict(self.text_dict)

        with uopen(self.out_text_lines, "wt") as out:
            for seq_tag, entry in d.items():
                assert isinstance(entry, str), f"expected str, got {entry!r} (type {type(entry).__name__})"
                out.write(entry + "\n")


class TextDictToStmJob(Job):
    """
    Similar as :class:`CorpusToStmJob`, but does not use the Bliss XML, but instead the text dict as input
    (e.g. via :class:`CorpusToTextDictJob`).
    Also see :class:`SearchWordsDummyTimesToCTMJob`.
    """

    def __init__(
        self,
        text_dict: Path,
        *,
        remove_non_speech_tokens: Sequence[str] = (),
        remove_punctuation_tokens: Union[str, Sequence[str]] = (),
        fix_whitespace: bool = True,
        tag_mapping: Sequence[Tuple[Tuple[str, str, str], Dict[int, Path]]] = (),
        seg_length_time: float = 1.0,
    ):
        """
        :param text_dict: e.g. via :class:`CorpusToTextDictJob`
        :param remove_non_speech_tokens: defines the list of non speech tokens to remove
        :param remove_punctuation_tokens: defines list/string of punctuation tokens to remove
        :param fix_whitespace: should white space be fixed.
        :param tag_mapping: 3-string tuple contains ("short name", "long name", "description") of each tag.
            and the Dict[int, Path] is e.g. the out_single_segment_files of a FilterSegments*Jobs
        :param seg_length_time: length of each segment in seconds.
            should be consistent to :class:`SearchWordsDummyTimesToCTMJob`
        """
        self.set_vis_name("Extract STM from text-dict file")

        self.text_dict = text_dict
        self.remove_non_speech_tokens = remove_non_speech_tokens
        self.remove_punctuation_tokens = remove_punctuation_tokens
        self.fix_whitespace = fix_whitespace
        self.tag_mapping = tag_mapping
        self.seg_length_time = seg_length_time

        self.out_stm_path = self.output_path("corpus.stm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # nan/inf should not be needed, but avoids errors at this point and will print an error below,
        # that we don't expect an N-best list here.
        c = parse_text_dict(self.text_dict)

        all_tags = [
            ("d%d" % i, "default%d" % i, "all other segments of category %d" % i)
            for i in range(len(self.tag_mapping) + 1)
        ]

        tag_map = {}
        for seg_name in c.keys():
            tag_map[seg_name] = ["d%d" % i for i in range(len(self.tag_mapping) + 1)]

        for i, (tag, segments) in enumerate(self.tag_mapping):
            all_tags.append(tag)
            for file in segments.values():
                for seg_name in uopen(file):
                    if seg_name.rstrip() in tag_map:
                        tag_map[seg_name.rstrip()][i] = tag[0]

        with uopen(self.out_stm_path, "wt") as out:
            for seg_name, orth in c.items():
                assert isinstance(orth, str)
                recording_name = seg_name  # simplification
                speaker_name = recording_name  # same as in CorpusToStmJob when no speaker information is available
                segment_track = 1  # same as in CorpusToStmJob when no track information is available
                seg_start = 0.0
                seg_end = self.seg_length_time

                orth = f" {orth.strip()} "

                for nst in self.remove_non_speech_tokens:
                    orth = self.replace_recursive(orth, nst)

                for pt in self.remove_punctuation_tokens:
                    orth = orth.replace(pt, "")

                if self.fix_whitespace:
                    orth = re.sub(" +", " ", orth)

                orth = orth.strip()

                out.write(
                    "%s %d %s %5.2f %5.2f <%s> %s\n"
                    % (
                        recording_name,
                        segment_track,
                        "_".join(speaker_name.split()),
                        seg_start,
                        seg_end,
                        ",".join(tag_map[seg_name]),
                        orth,
                    )
                )
            for tag in all_tags:
                out.write(';; LABEL "%s" "%s" "%s"\n' % tag)

    @classmethod
    def replace_recursive(cls, orthography: str, token: str) -> str:
        """
        recursion is required to find repeated tokens
        string.replace is not sufficient
        some other solution might also work
        """
        while True:
            pos = orthography.find(f" {token} ")
            if pos == -1:
                return orthography
            else:
                orthography = orthography.replace(f" {token} ", " ")
