"""
Helper functions and classes for Bliss xml corpus loading and writing
"""
from __future__ import annotations

__all__ = ["NamedEntity", "CorpusSection", "Corpus", "Recording", "Segment", "Speaker"]

import collections
import gzip
import os
import re
from typing import Callable, Dict, Iterable, List, Optional, TextIO
import xml
import xml.sax as sax
import xml.sax.saxutils as saxutils
import xml.etree.ElementTree as ET


FilterFunction = Callable[["Corpus", "Recording", "Segment"], bool]


class NamedEntity:
    def __init__(self):
        super().__init__()
        self.name: Optional[str] = None


class CorpusSection:
    def __init__(self):
        super().__init__()
        self.speaker_name: Optional[str] = None
        self.default_speaker: Optional[Speaker] = None

        self.speakers = collections.OrderedDict()


class CorpusParser(sax.handler.ContentHandler):
    """
    This classes methods are called by the sax-parser whenever it encounters an event in the xml-file
    (tags/characters/namespaces/...). It uses a stack of elements to remember the part of the corpus that
    is currently beeing read.
    """

    def __init__(self, corpus: Corpus, path: str):
        super().__init__()

        self.elements: List[NamedEntity] = [
            corpus
        ]  # stack of objects to store the element of the corpus that is beeing read
        self.path = path  # path of the parent corpus (needed for include statements)
        self.chars = ""  # buffer for character events, it is reset whenever a new element starts

    def startElement(self, name: str, attrs: Dict[str, str]):
        e = self.elements[-1]
        if name == "corpus":
            assert len(self.elements) == 1, "<corpus> may only occur as the root element"
            e.name = attrs["name"]
        elif name == "subcorpus":
            assert isinstance(e, Corpus), "<subcorpus> may only occur within a <corpus> or <subcorpus> element"
            subcorpus = Corpus()
            subcorpus.name = attrs["name"]
            subcorpus.parent_corpus = e
            e.subcorpora.append(subcorpus)
            self.elements.append(subcorpus)
        elif name == "include":
            assert isinstance(e, Corpus), "<include> may only occur within a <corpus> or <subcorpus> element"
            path = os.path.join(os.path.dirname(self.path), attrs["file"])
            c = Corpus()
            c.load(path)
            if c.name != e.name:
                print(
                    "Warning: included corpus (%s) has a different name than the current corpus (%s)" % (c.name, e.name)
                )
            for sc in c.subcorpora:
                sc.parent_corpus = e.parent_corpus
            for r in c.recordings:
                r.corpus = e
            e.subcorpora.extend(c.subcorpora)
            e.recordings.extend(c.recordings)
            e.speakers.update(c.speakers)
        elif name == "recording":
            assert isinstance(e, Corpus), "<recording> may only occur within a <corpus> or <subcorpus> element"
            rec = Recording()
            rec.name = attrs["name"]
            rec.audio = attrs["audio"]
            rec.corpus = e
            e.recordings.append(rec)
            self.elements.append(rec)
        elif name == "segment":
            assert isinstance(e, Recording), "<segment> may only occur within a <recording> element"
            seg = Segment()
            seg.name = attrs.get("name", str(len(e.segments) + 1))
            seg.start = float(attrs.get("start", "0.0"))
            seg.end = float(attrs.get("end", "0.0"))
            seg.track = int(attrs["track"]) if "track" in attrs else None
            seg.recording = e
            e.segments.append(seg)
            self.elements.append(seg)
        elif name == "speaker-description":
            assert isinstance(
                e, CorpusSection
            ), "<speaker-description> may only occur within a <corpus>, <subcorpus> or <recording>"
            speaker = Speaker()
            speaker.name = attrs.get("name", None)
            if speaker.name is not None:
                e.speakers[speaker.name] = speaker
            else:
                e.default_speaker = speaker
            self.elements.append(speaker)
        elif name == "speaker":
            assert isinstance(
                e, (CorpusSection, Segment)
            ), "<speaker> may only occur within a <corpus>, <subcorpus>, <recording> or <segment>"
            e.speaker_name = attrs["name"]
        self.chars = ""

    def endElement(self, name: str):
        e = self.elements[-1]

        if name == "orth":
            assert isinstance(e, Segment)
            # we do some processing of the text that goes into the orth tag to get a nicer formating, some corpora may have
            # multiline content in the orth tag, but to keep it that way might not be consistent with the indentation during
            # writing, thus we remove multiple spaces and newlines
            text = self.chars.strip()
            text = re.sub(" +", " ", text)
            text = re.sub("\n", "", text)
            e.orth = text
        elif isinstance(e, Speaker) and name != "speaker-description":
            # we allow all sorts of elements within a speaker description
            e.attribs[name] = self.chars.strip()

        if name in [
            "corpus",
            "subcorpus",
            "recording",
            "segment",
            "speaker-description",
        ]:
            self.elements.pop()

    def characters(self, characters: str):
        self.chars += characters


class Corpus(NamedEntity, CorpusSection):
    """
    This class represents a corpus in the Bliss format. It is also used to represent subcorpora when the parent_corpus
    attribute is set. Corpora with include statements can be read but are written back as a single file.
    """

    def __init__(self):
        super().__init__()

        self.parent_corpus: Optional[Corpus] = None

        self.subcorpora: List[Corpus] = []
        self.recordings: List[Recording] = []

    def segments(self) -> Iterable[Segment]:
        """
        :return: an iterator over all segments within the corpus
        """
        for r in self.recordings:
            yield from r.segments
    for sc in self.subcorpora:
            yield from sc.segments()

    def get_segment_by_name(self, name: str) -> Segment:
        """
        :return: the segment specified by its name
        """
        for seg in self.segments():
            if seg.fullname() == name:
                return seg
        assert False, f"Segment '{name}' was not found in corpus"

    def all_recordings(self) -> Iterable[Recording]:
        yield from self.recordings
        for sc in self.subcorpora:
            yield from sc.all_recordings()

    def all_speakers(self) -> Iterable[Speaker]:
        yield from self.speakers.values()
        for sc in self.subcorpora:
            yield from sc.all_speakers()

    def top_level_recordings(self) -> Iterable[Recording]:
        yield from self.recordings

    def top_level_subcorpora(self) -> Iterable[Corpus]:
        yield from self.subcorpora

    def top_level_speakers(self) -> Iterable[Speaker]:
        yield from self.speakers.values()

    def remove_recording(self, recording: Recording):
        to_delete = []
        for idx, r in enumerate(self.recordings):
            if r is recording or r == recording or r.name == recording:
                to_delete.append(idx)
        for idx in reversed(to_delete):
            del self.recordings[idx]
        for sc in self.subcorpora:
            sc.remove_recording(recording)

    def remove_recordings(self, recordings: List[Recording]):
        recording_fullnames = {recording.fullname() for recording in recordings}
        to_delete = []
        for idx, r in enumerate(self.recordings):
            if r.fullname() in recording_fullnames:
                to_delete.append(idx)
        for idx in reversed(to_delete):
            del self.recordings[idx]
        for sc in self.subcorpora:
            sc.remove_recordings(recordings)

    def add_recording(self, recording: Recording):
        assert isinstance(recording, Recording)
        recording.corpus = self
        self.recordings.append(recording)

    def add_subcorpus(self, corpus: Corpus):
        assert isinstance(corpus, Corpus)
        corpus.parent_corpus = self
        self.subcorpora.append(corpus)

    def add_speaker(self, speaker: Speaker):
        assert isinstance(speaker, Speaker)
        self.speakers[speaker.name] = speaker

    def fullname(self) -> str:
        if self.parent_corpus is not None:
            return self.parent_corpus.fullname() + "/" + self.name
        else:
            return self.name

    def speaker(self, speaker_name: Optional[str], default_speaker: Optional[Speaker]) -> Speaker:
        if speaker_name is None:
            speaker_name = self.speaker_name
        if speaker_name in self.speakers:
            return self.speakers[speaker_name]
        else:
            if default_speaker is None:
                default_speaker = self.default_speaker
            if self.parent_corpus is not None:
                return self.parent_corpus.speaker(speaker_name, default_speaker)
            else:
                return default_speaker

    def filter_segments(self, filter_function: FilterFunction):
        """
        filter all segments (including in subcorpora) using filter_function
        :param filter_function: takes arguments corpus, recording and segment, returns True if segment should be kept
        """
        for r in self.recordings:
            r.segments = [s for s in r.segments if filter_function(self, r, s)]
        for sc in self.subcorpora:
            sc.filter_segments(filter_function)

    def load(self, path: str):
        """
        :param path: corpus .xml or .xml.gz
        """
        open_fun = gzip.open if path.endswith(".gz") else open

        with open_fun(path, "rt") as f:
            handler = CorpusParser(self, path)
            sax.parse(f, handler)

    def dump(self, path: str):
        """
        :param path: target .xml or .xml.gz path
        """
        open_fun = gzip.open if path.endswith(".gz") else open

        with open_fun(path, "wt") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            self._dump_internal(f)

    def _dump_internal(self, out: TextIO, indentation: str = ""):
        if self.parent_corpus is None:
            out.write('<corpus name="%s">\n' % self.name)
        else:
            out.write('%s<subcorpus name="%s">\n' % (indentation, self.name))

        for s in self.speakers.values():
            s.dump(out, indentation + "  ")
        if self.speaker_name is not None:
            out.write('%s  <speaker name="%s"/>\n' % (indentation, self.speaker_name))

        for r in self.recordings:
            r.dump(out, indentation + "  ")

        for sc in self.subcorpora:
            sc._dump_internal(out, indentation + "  ")

        if self.parent_corpus is None:
            out.write("</corpus>\n")
        else:
            out.write("%s</subcorpus>\n" % (indentation,))


class CorpusV2(Corpus):
    """
    This class represents a corpus in the Bliss format. It is also used to represent subcorpora when the parent_corpus
    attribute is set. Corpora with include statements can be read but are written back as a single file.

    The difference with respect to :class:`i6_core.lib.corpus.Corpus` is that in this class we can access
    any recording or segment in practically O(1).
    """

    def __init__(self):
        super().__init__()

        self.parent_corpus: Optional[Corpus] = None

        self.subcorpora: Dict[str, Corpus] = {}
        self.recordings: Dict[str, RecordingV2] = {}

    def segments(self) -> Iterable[Segment]:
        """
        :return: an iterator over all segments within the corpus
        """
        for r in self.recordings:
            yield from r.segments.values()
        for sc in self.subcorpora:
            yield from sc.segments()

    def get_segment_by_name(self, name: str) -> Segment:
        """
        :return: the segment specified by its name
        """
        for seg in self.segments():
            if seg.name == name:
                return seg
        assert False, f"Segment '{name}' was not found in corpus"

     def get_segment_by_full_name(self, name: str) -> Optional[Segment]:
        """
        :return: the segment specified by its full name
        """
        if name == "":
            # Found nothing.
            return None

        if name in self.segments:
            return self.segments[name]
        else:
            subcorpus_name = name.split("/")[0]
            segment_name_from_subcorpus = name[len(f"{subcorpus_name}/"):]
            return self.subcorpora[subcorpus_name].get_segment_by_full_name(segment_name_from_subcorpus)

    def get_recording_by_full_name(self, name: str) -> Optional[Segment]:
        """
        :return: the segment specified by its full name
        """
        if name == "":
            # Found nothing.
            return None

        if name in self.segments:
            return self.segments[name]
        else:
            subcorpus_name = name.split("/")[0]
            segment_name_from_subcorpus = name[len(f"{subcorpus_name}/"):]
            return self.subcorpora[subcorpus_name].get_segment_by_full_name(segment_name_from_subcorpus)

    def all_recordings(self) -> Iterable[Recording]:
        yield from self.recordings.values()
        for sc in self.subcorpora.values():
            yield from sc.all_recordings()

    def all_speakers(self) -> Iterable[Speaker]:
        yield from self.speakers.values()
        for sc in self.subcorpora:
            yield from sc.all_speakers()

    def top_level_recordings(self) -> Iterable[Recording]:
        yield from self.recordings.values()

    def top_level_subcorpora(self) -> Iterable[Corpus]:
        yield from self.subcorpora.values()

    def top_level_speakers(self) -> Iterable[Speaker]:
        yield from self.speakers.values()

    def remove_recording(self, recording: Recording):
        del self.recordings[recording.name]
        for sc in self.subcorpora.values():
            sc.remove_recording(recording)

    def remove_recordings(self, recordings: List[Recording]):
        for recording in recordings:
            del self.recordings[recording.name]
        for sc in self.subcorpora:
            sc.remove_recordings(recordings)

    def add_recording(self, recording: Recording):
        assert isinstance(recording, Recording)
        recording.corpus = self
        self.recordings[recording.name] = recording

    def add_subcorpus(self, corpus: Corpus):
        assert isinstance(corpus, Corpus)
        corpus.parent_corpus = self
        self.subcorpora[corpus.name] = corpus

    def add_speaker(self, speaker: Speaker):
        assert isinstance(speaker, Speaker)
        self.speakers[speaker.name] = speaker

    def fullname(self) -> str:
        if self.parent_corpus is not None:
            return self.parent_corpus.fullname() + "/" + self.name
        else:
            return self.name

    def filter_segments(self, filter_function: FilterFunction):
        """
        filter all segments (including in subcorpora) using filter_function
        :param filter_function: takes arguments corpus, recording and segment, returns True if segment should be kept
        """
        for r in self.recordings.values():
            r.segments = [s for s in r.segments.values() if filter_function(self, r, s)]
        for sc in self.subcorpora:
            sc.filter_segments(filter_function)

    def _dump_internal(self, out: TextIO, indentation: str = ""):
        if self.parent_corpus is None:
            out.write('<corpus name="%s">\n' % self.name)
        else:
            out.write('%s<subcorpus name="%s">\n' % (indentation, self.name))

        for s in self.speakers.values():
            s.dump(out, indentation + "  ")
        if self.speaker_name is not None:
            out.write('%s  <speaker name="%s"/>\n' % (indentation, self.speaker_name))

        for r in self.recordings.values():
            r.dump(out, indentation + "  ")

        for sc in self.subcorpora.values():
            sc._dump_internal(out, indentation + "  ")

        if self.parent_corpus is None:
            out.write("</corpus>\n")
        else:
            out.write("%s</subcorpus>\n" % (indentation,))


class Recording(NamedEntity, CorpusSection):
    def __init__(self):
        super().__init__()
        self.audio: Optional[str] = None
        self.corpus: Optional[Corpus] = None
        self.segments: List[Segment] = []

    def fullname(self) -> str:
        return self.corpus.fullname() + "/" + self.name

    def speaker(self, speaker_name: Optional[str] = None) -> Speaker:
        if speaker_name is None:
            speaker_name = self.speaker_name
        if speaker_name in self.speakers:
            return self.speakers[speaker_name]
        else:
            return self.corpus.speaker(speaker_name, self.default_speaker)

    def dump(self, out: TextIO, indentation: str = ""):
        out.write('%s<recording name="%s" audio="%s">\n' % (indentation, self.name, self.audio))

        for s in self.speakers.values():
            s.dump(out, indentation + "  ")
        if self.speaker_name is not None:
            out.write('%s  <speaker name="%s"/>\n' % (indentation, self.speaker_name))

        for s in self.segments:
            s.dump(out, indentation + "  ")

        out.write("%s</recording>\n" % indentation)

    def add_segment(self, segment: Segment):
        assert isinstance(segment, Segment)
        segment.recording = self
        self.segments.append(segment)


class RecordingV2(Recording):
    def __init__(self):
        super().__init__()
        self.audio: Optional[str] = None
        self.corpus: Optional[Corpus] = None
        self.segments: Dict[str, Segment] = {}

    def add_segment(self, segment: Segment):
        assert isinstance(segment, Segment)
        segment.recording = self
        self.segments[segment.name] = segment



class Segment(NamedEntity):
    def __init__(self):
        super().__init__()
        self.start = 0.0
        self.end = 0.0
        self.track: Optional[int] = None
        self.orth: Optional[str] = None
        self.speaker_name: Optional[str] = None

        self.recording: Optional[Recording] = None

    def fullname(self) -> str:
        return self.recording.fullname() + "/" + self.name

    def speaker(self) -> Speaker:
        return self.recording.speaker(self.speaker_name)

    def dump(self, out: TextIO, indentation: str = ""):
        has_child_element = self.orth is not None or self.speaker_name is not None
        t = ' track="%d"' % self.track if self.track is not None else ""
        new_line = "\n" if has_child_element else ""
        out.write(
            '%s<segment name="%s" start="%.4f" end="%.4f"%s>%s'
            % (indentation, self.name, self.start, self.end, t, new_line)
        )
        if self.speaker_name is not None:
            out.write('%s  <speaker name="%s"/>\n' % (indentation, self.speaker_name))
        if self.orth is not None:
            out.write("%s  <orth> %s </orth>\n" % (indentation, saxutils.escape(self.orth)))
        if has_child_element:
            out.write("%s</segment>\n" % indentation)
        else:
            out.write("</segment>\n")


class Speaker(NamedEntity):
    def __init__(self):
        super().__init__()
        self.attribs: Dict[str, str] = {}

    def dump(self, out: TextIO, indentation: str = ""):
        out.write(
            "%s<speaker-description%s>" % (indentation, (' name="%s"' % self.name) if self.name is not None else "")
        )
        if len(self.attribs) > 0:
            out.write("\n")
        for k, v in self.attribs.items():
            out.write("%s  <%s>%s</%s>\n" % (indentation, k, v, k))
        out.write("%s</speaker-description>\n" % (indentation if len(self.attribs) > 0 else ""))


class SegmentMap(object):
    def __init__(self):
        self.map_entries: List[SegmentMapItem] = []

    def load(self, path: str):
        """
        :param  path: segment file path (optionally with .gz)
        """
        open_fun = gzip.open if path.endswith(".gz") else open

        with open_fun(path, "rb") as f:
            for event, elem in ET.iterparse(f, events=("start",)):
                elem: xml.etree.ElementTree = elem
                if elem.tag == "map-item":
                    item = SegmentMapItem()
                    item.key = elem.attrib["key"]
                    item.value = elem.attrib["value"]
                    self.map_entries.append(item)

    def dump(self, path: str):
        """
        :param path: segment file path with no extension or .gz
        """
        open_fun = gzip.open if path.endswith(".gz") else open

        with open_fun(path, "wt") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write("<segment-key-map>\n")

            for s in self.map_entries:
                s.dump(f)

            f.write("</segment-key-map>\n")


class SegmentMapItem(object):
    def __init__(self):
        self.key: Optional[str] = None
        self.value: Optional[str] = None

    def dump(
        self,
        out: TextIO,
    ):
        out.write('<map-item key="%s" value="%s" />\n' % (self.key, self.value))
