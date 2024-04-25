__all__ = [
    "TextDictToTextLinesJob",
    "TextDictToStmJob",
]

from typing import Optional, Union, Sequence, Dict, List, Tuple
import re
from sisyphus import Job, Path, Task
from i6_core.util import uopen


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
        d = eval(uopen(self.text_dict, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> text

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
        exclude_non_speech: bool = True,
        non_speech_tokens: Optional[List[str]] = None,
        remove_punctuation: bool = True,
        punctuation_tokens: Optional[Union[str, List[str]]] = None,
        fix_whitespace: bool = True,
        tag_mapping: Sequence[Tuple[Tuple[str, str, str], Dict[int, Path]]] = (),
        seg_length_time: float = 1.0,
    ):
        """
        :param text_dict: e.g. via :class:`CorpusToTextDictJob`
        :param exclude_non_speech: non speech tokens should be removed
        :param non_speech_tokens: defines the list of non speech tokens
        :param remove_punctuation: should punctuation be removed
        :param punctuation_tokens: defines list/string of punctuation tokens
        :param fix_whitespace: should white space be fixed.
            !!!be aware that the corpus loading already fixes white space!!!
        :param tag_mapping: 3-string tuple contains ("short name", "long name", "description") of each tag.
            and the Dict[int, Path] is e.g. the out_single_segment_files of a FilterSegments*Jobs
        :param seg_length_time: length of each segment in seconds.
            should be consistent to :class:`SearchWordsDummyTimesToCTMJob`
        """
        self.set_vis_name("Extract STM from text-dict file")

        self.text_dict = text_dict
        self.exclude_non_speech = exclude_non_speech
        self.non_speech_tokens = non_speech_tokens if non_speech_tokens is not None else []
        self.remove_punctuation = remove_punctuation
        self.punctuation_tokens = punctuation_tokens if punctuation_tokens is not None else []
        self.fix_whitespace = fix_whitespace
        self.tag_mapping = tag_mapping
        self.seg_length_time = seg_length_time

        self.out_stm_path = self.output_path("corpus.stm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # nan/inf should not be needed, but avoids errors at this point and will print an error below,
        # that we don't expect an N-best list here.
        c = eval(uopen(self.text_dict, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(c, dict)

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

                if self.exclude_non_speech:
                    for nst in self.non_speech_tokens:
                        orth = self.replace_recursive(orth, nst)

                if self.remove_punctuation:
                    for pt in self.punctuation_tokens:
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
    def replace_recursive(cls, orthography, token):
        """
        recursion is required to find repeated tokens
        string.replace is not sufficient
        some other solution might also work
        """
        pos = orthography.find(f" {token} ")
        if pos == -1:
            return orthography
        else:
            orthography = orthography.replace(f" {token} ", " ")
            return cls.replace_recursive(orthography, token)