"""
Switchboard is conversational telephony speech with 8 Khz audio files. The training data consists of
300h hours.
Reference: https://catalog.ldc.upenn.edu/LDC97S62

number of recordings: 4876
number of segments: 249624
number of speakers: 2260
"""


from sisyphus import Job, Task, tk, setup_path

from collections import defaultdict
import glob
import subprocess
import shutil
import os
import re
from typing import List, DefaultDict

from i6_core.lib import corpus
from i6_core.util import uopen
from i6_core.tools.download import DownloadJob


SPECIAL_TOKENS = {
    "[vocalized-noise]",
    "[noise]",
    "[laughter]",
}


def _map_token(token):
    """
    This function applies some mapping rules for Switchboard transcription words
    Reference: https://github.com/espnet/espnet/blob/master/egs/swbd/asr1/local/swbd1_map_words.pl

    :param str token: string representing token, e.g word
    :rtype str
    """

    mapped_token = re.sub(
        "(|\-)^\[laughter-(.+)\](|\-)$", "\g<1>\g<2>\g<3>", token
    )  # e.g. [laughter-story] -> story;
    # 1 and 3 relate to preserving trailing "-"
    mapped_token = re.sub(
        "^\[(.+)/.+\](|\-)$", "\g<1>\g<2>", mapped_token
    )  # e.g. [it'n/isn't] -> it'n ... note
    # 1st part may include partial-word stuff, which we process further below,
    # e.g. [LEM[GUINI]-/LINGUINI]
    # the (|\_) at the end is to accept and preserve trailing -'s.
    mapped_token = re.sub(
        "^(|\-)\[[^][]+\](.+)$", "-\g<2>", mapped_token
    )  # e.g. -[an]y , note \047 is quote;
    # let the leading - be optional on input, as sometimes omitted.
    mapped_token = re.sub(
        "^(.+)\[[^][]+\](|\-)$", "\g<1>-", mapped_token
    )  # e.g. ab[solute]- -> ab-;
    # let the trailing - be optional on input, as sometimes omitted.
    mapped_token = re.sub(
        "([^][]+)\[.+\]$", "\g<1>", mapped_token
    )  # e.g. ex[specially]-/especially] -> ex-
    # which is a  mistake in the input.
    mapped_token = re.sub(
        "^\{(.+)\}$", "\g<1>", mapped_token
    )  # e.g. {yuppiedom} -> yuppiedom
    mapped_token = re.sub(
        "([a-z])\[([^][])+\]([a-z])", "\g<1>-\g<3>", mapped_token
    )  # e.g. ammu[n]it- -> ammu-it-
    mapped_token = re.sub("_\d$", "", mapped_token)  # e.g. them_1 -> them

    return mapped_token


class DownloadSwitchboardTranscriptionAndDictJob(Job):
    """
    Downloads switchboard training transcriptions and dictionary (or lexicon)
    """

    def __init__(self):
        self.out_raw_dict = self.output_path("swb_trans/sw-ms98-dict.text")
        self.out_trans_dir = self.output_path("swb_trans")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        zipped_filename = "switchboard_word_alignments.tar.gz"
        subprocess.check_call(
            ["wget", "http://www.openslr.org/resources/5/" + zipped_filename]
        )
        subprocess.check_call(
            [
                "tar",
                "-xf",
                zipped_filename,
                "-C",
                ".",
            ]
        )
        shutil.move("swb_ms98_transcriptions", self.out_trans_dir.get_path())
        os.remove(zipped_filename)


class DownloadSwitchboardSpeakersStatsJob(DownloadJob):
    """
    Note that this does not contain the speaker info for all recordings. We assume later that each
    recording has a unique speaker and a unique id is used for those recordings with unknown speakers info
    """

    def __init__(self):
        super(DownloadSwitchboardSpeakersStatsJob, self).__init__(
            url="http://www.isip.piconepress.com/projects/switchboard/doc/statistics/ws97_speaker_stats.text",
            checksum="64f538839073dbbdf46027fff40cec57a11c5de1eed4e8b22b50ed86038d9e90",
        )

    @classmethod
    def hash(cls, parsed_args):
        return Job.hash(parsed_args)


class CreateSwitchboardSpeakersListJob(Job):
    """
    Given some speakers statistics info, this job creates a text file having on each line:
        speaker_id gender recording
    """

    def __init__(self, speakers_stats_file):
        """
        :param tk.Path speakers_stats_file: speakers stats text file
        """
        self.speakers_stats_file = speakers_stats_file
        self.out_speakers_list = self.output_path("speakers_list.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        speaker_id, gender, rec_name = None, None, None
        with uopen(self.speakers_stats_file) as read_f, uopen(
            self.out_speakers_list, "w"
        ) as out_f:
            for line in read_f:
                l = line.strip().split()
                if len(l) < 2:
                    continue
                if l[1] == "F" or l[1] == "M":  # start new speaker
                    speaker_id = l[0]
                    gender = l[1]
                    rec_name = l[2]
                elif l[0].endswith("A") or l[0].endswith("B"):  # recording name
                    rec_name = l[0]
                else:
                    continue

                if speaker_id:
                    out_f.write(
                        speaker_id + " " + gender + " " + rec_name + "\n"
                    )  # speaker_id gender recording


class CreateLDCSwitchboardSpeakerListJob(Job):
    """
    This creates the speaker list according to the conversation and speaker table
    from the LDC documentation: https://catalog.ldc.upenn.edu/docs/LDC97S62

    The resulting file contains 520 speakers in the format of:
        <speaker_id> <gender> <recording>
    """

    def __init__(self, caller_tab_file, conv_tab_file):
        """
        :param caller_tab_file: caller_tab.csv from the Switchboard LDC documentation
        :param conv_tab_file: conv_tab.csv from the Switchboard LDC documentation
        """
        # locally create the download jobs
        self.caller_tab_file = caller_tab_file
        self.conv_tab_file = conv_tab_file

        self.out_speakers_list = self.output_path("speakers_list.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _conv_gender(gender):
        if gender == '"MALE"':
            return "M"
        elif gender == '"FEMALE"':
            return "F"
        else:
            assert False, "invalid gender %s" % gender

    def run(self):
        speakers = {}
        with uopen(self.caller_tab_file, "rt") as f:
            for line in f.readlines():
                split = line.strip().split(",")
                sid = int(split[0])
                gender = split[3].strip()
                speakers[sid] = gender

        with uopen(self.out_speakers_list, "wt") as fout:
            with uopen(self.conv_tab_file, "rt") as f:
                for line in f.readlines():
                    split = line.strip().split(",")
                    seq_id = int(split[0])
                    callerA = int(split[2])
                    callerB = int(split[3])
                    genderA = self._conv_gender(speakers[callerA])
                    genderB = self._conv_gender(speakers[callerB])
                    fout.write("%d %s %dA\n" % (callerA, genderA, seq_id))
                    fout.write("%d %s %dB\n" % (callerB, genderB, seq_id))


class CreateSwitchboardBlissCorpusJob(Job):
    """
    Creates Switchboard bliss corpus xml

    segment name format: sw2001B-ms98-a-<folder-name>
    """

    __sis_hash_exclude__ = {"skip_empty_ldc_file": False, "lowercase": False}

    def __init__(
        self,
        audio_dir,
        trans_dir,
        speakers_list_file,
        skip_empty_ldc_file=True,
        lowercase=True,
    ):
        """
        :param tk.Path audio_dir: path for audio data
        :param tk.Path trans_dir: path for transcription data. see `DownloadSwitchboardTranscriptionAndDictJob`
        :param tk.Path speakers_list_file: path to a speakers list text file with format:
                speaker_id gender recording<channel>, e.g. 1005 F 2452A
            on each line. see `CreateSwitchboardSpeakersListJob` job
        :param bool skip_empty_ldc_file: In the original corpus the sequence 2167B is mostly empty,
            thus exclude it from training (recommended, GMM will fail otherwise)
        :param bool lowercase: lowercase the transcriptions of the corpus (recommended)
        """
        self.audio_dir = audio_dir
        self.trans_dir = trans_dir
        self.speakers_list_file = speakers_list_file
        self.skip_empty_ldc_file = skip_empty_ldc_file
        self.lowercase = lowercase

        self.out_corpus = self.output_path("swb.corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.name = "switchboard-1"

        rec_to_segs = self._get_rec_to_segs_map()

        rec_to_speaker = {}
        with uopen(self.speakers_list_file) as f:
            for line in f:
                l = line.strip().split()
                assert len(l) == 3
                assert (
                    l[2] not in rec_to_speaker
                ), "duplicate recording name: {}?".format(l[2])
                assert l[1] in ["F", "M"]

                # "sw0" prefix is added to match recording names
                rec_to_speaker["sw0" + l[2]] = {
                    "speaker_id": l[0],
                    "gender": {"M": "male", "F": "female"}.get(l[1]),
                }

        # assume unique speaker for each recording with no speaker info
        unk_spk_id = 1
        for rec in sorted(rec_to_segs.keys()):
            if rec not in rec_to_speaker:
                rec_to_speaker[rec] = {"speaker_id": "speaker#" + str(unk_spk_id)}
                unk_spk_id += 1

        if self.skip_empty_ldc_file:
            rec_to_segs.pop("sw02167B")

        for rec_name, segs in sorted(rec_to_segs.items()):
            recording = corpus.Recording()
            recording.name = rec_name
            recording.audio = os.path.join(self.audio_dir.get_path(), rec_name + ".wav")

            assert os.path.exists(
                recording.audio
            ), "recording {} does not exist?".format(recording.audio)

            assert (
                rec_name in rec_to_speaker
            ), "recording {} does not have speaker id?".format(rec_name)
            rec_speaker_id = rec_to_speaker[rec_name]["speaker_id"]

            for seg in segs:
                segment = corpus.Segment()
                segment.name = seg[0]
                segment.start = float(seg[1])
                segment.end = float(seg[2])
                segment.speaker_name = rec_speaker_id
                segment.orth = self._filter_orth(seg[3])
                if len(segment.orth) == 0:
                    continue

                recording.segments.append(segment)
            c.recordings.append(recording)

        # add speakers to corpus
        for speaker_info in rec_to_speaker.values():
            speaker = corpus.Speaker()
            speaker.name = speaker_info["speaker_id"]
            if speaker_info.get("gender", None):
                speaker.attribs["gender"] = speaker_info["gender"]
            c.add_speaker(speaker)

        c.dump(self.out_corpus.get_path())

    def _filter_orth(self, orth):
        """
        Filters orth by handling special cases such as silence tag removal, partial words, etc

        :param str orth: segment orth to be preprocessed
        """
        removed_tokens = {
            "[silence]",
            "<b_aside>",
            "<e_aside>",
        }  # unnecessary tags to be removed
        filtered_orth = []
        tokens = orth.strip().split()
        for token_ in tokens:
            token = token_.strip()
            if self.lowercase:
                token = token.lower()
            if token in removed_tokens:
                continue
            elif token in SPECIAL_TOKENS:
                filtered_orth.append(
                    token.upper()
                )  # make upper case for consistency with older setups
            else:
                filtered_orth.append(_map_token(token))

        # do not add empty transcription segments
        all_special = True
        for token in filtered_orth:
            if token.lower() not in SPECIAL_TOKENS:
                all_special = False
                break
        if all_special:
            return ""

        out = " ".join(filtered_orth)

        # replace &
        # for AT&T's we drop the 's as t's is not in the lexicon
        if self.lowercase:
            out = out.replace("at&t's", "at and t")
        else:
            out = out.replace("AT&T's", "AT and T")
        out = out.replace("&", " and ")

        return out

    def _get_rec_to_segs_map(self):
        """
        Returns recording to list of segments mapping
        """
        rec_to_segs = defaultdict(list)
        for trans_file in glob.glob(
            os.path.join(self.trans_dir.get_path(), "*/*/*-trans.text")
        ):
            with uopen(trans_file, "rt") as f:
                for line in f:
                    seg_info = line.strip().split(" ", 3)  # name start end orth
                    assert len(seg_info) == 4
                    rec_name = (
                        seg_info[0].split("-")[0].replace("sw", "sw0")
                    )  # e.g: sw2001A-ms98-a-0022 -> sw02001A
                    rec_to_segs[rec_name].append(seg_info)
        return rec_to_segs


class CreateSwitchboardLexiconTextFileJob(Job):
    """
    This job creates SWB preprocessed dictionary text file consistent with the training corpus given a raw dictionary
    text file downloaded within the transcription directory using `DownloadSwitchboardTranscriptionAndDictJob` Job.
    The resulted dictionary text file will be passed as argument to `LexiconFromTextFileJob` job in order to create
    bliss xml lexicon.
    """

    def __init__(self, raw_dict_file):
        """
        :param tk.Path raw_dict_file: path containing the raw dictionary text file
        """
        self.raw_dict_file = raw_dict_file
        self.out_dict = self.output_path("dict.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.raw_dict_file) as read_f, uopen(self.out_dict, "w") as out_f:
            for line in read_f.readlines()[1:]:
                if line.startswith("#"):  # skip comment
                    continue
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue
                token = parts[0].replace("&amp;", "&")  # e.g A&amp;E -> A&E
                mapped_token = _map_token(token)  # preprocessing as corpus
                out_f.write(mapped_token + " " + parts[1] + "\n")


class SwitchboardSphereToWaveJob(Job):
    """
    Takes an audio folder from one of the switchboard LDC folders and converts dual channel .sph files
    with mulaw encoding to single channel .wav files with s16le encoding
    """

    def __init__(self, sph_audio_folder: tk.Path):
        """
        :param sph_audio_folder:
        """
        self.sph_audio_folder = sph_audio_folder

        self.out_wave_audio_folder = self.output_path("wave_audio", directory=True)

        self.rqmt = {"cpu": 1, "mem": 1, "time": 1.0}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        for sph_file in glob.glob(
            os.path.join(self.sph_audio_folder.get_path(), "**/*.sph"), recursive=True
        ):
            sph_name, ext = os.path.splitext(os.path.basename(sph_file))
            subprocess.call(
                [
                    "ffmpeg",
                    "-i",
                    sph_file,
                    "-filter_complex",
                    "[0:a]channelsplit=channel_layout=stereo[left][right]",
                    "-c:a",
                    "pcm_s16le",
                    "-map",
                    "[left]",
                    os.path.join(
                        self.out_wave_audio_folder.get_path(), f"{sph_name}A.wav"
                    ),
                    "-map",
                    "[right]",
                    os.path.join(
                        self.out_wave_audio_folder.get_path(), f"{sph_name}B.wav"
                    ),
                ]
            )


#### Evaluation Corpus Helper ####


def _process_and_write_stm(stm_in_files: List[str], stm_out_file: str):
    """
    Kaldi-preprocessing (remove double brackets, remove <B_ASIDE> and <E_ASIDE>)

    Change naming pattern to Zoltan style to match the corpus naming with respect to the splitted audio files,
    otherwise there might be conflicts with the ctm, so e.g. from "en_4156 B" -> "en_4156b 1"

    Will write a single target .stm file to be used as reference for the Hub5Scorer

    :param stm_in_files: list of original stm files
    :param stm_out_file: file path to write the final stm to
    """
    remove_extra_tag = re.compile(" *<._ASIDE>")
    remove_double_bracket = re.compile("\(\(")
    channel_a = re.compile(" [A1] ")
    channel_b = re.compile(" [B2] ")
    inter_segment_gap = re.compile("inter_segment_gap")

    with uopen(stm_out_file, "wt") as stm_out:
        for stm_file in stm_in_files:
            with uopen(stm_file, "rt") as stm_in:
                for line in stm_in:
                    if line.startswith(";;"):
                        stm_out.write(line)
                        continue
                    if inter_segment_gap.search(line) is not None:
                        continue
                    line = re.sub(" +", " ", line.strip())
                    # name channel name+channel start end info [TEXT]
                    # in some cases there are arbitrary extra whitespaces
                    fields = line.split(" ", maxsplit=6)
                    header = " ".join(fields[:6])
                    header = channel_a.sub("a 1 ", header)
                    header = channel_b.sub("b 1 ", header)
                    if len(fields) == 6:
                        # rt03 can have empty entries
                        stm_out.write(f"{header}\n")
                        continue
                    content = fields[6]
                    content = remove_extra_tag.sub("", content)
                    content = remove_double_bracket.sub("(", content)
                    stm_out.write(f"{header} {content}\n")


def _get_segment_list_per_file(stm_file: str) -> DefaultDict[str, List[corpus.Segment]]:
    """
    Create corpus segments from the stm

    :param stm_file: reference stm file path
    :return: dict containing lists of segments for each recording
    """
    segment_list_per_file = defaultdict(list)

    for line in uopen(stm_file):
        if line.startswith(";;"):
            continue
        cleaned_line = re.sub(" +", " ", line.strip())
        fields = cleaned_line.split(" ", maxsplit=6)
        # audio filenames have no underscore for us
        name = fields[0]
        segment = corpus.Segment()
        # increasing number starting from 1
        segment.name = len(segment_list_per_file[name]) + 1
        segment.start = float(fields[3])
        segment.end = float(fields[4])
        # there can be empty entries
        segment.orth = fields[6].strip() if len(fields) == 7 else ""
        if segment.orth.startswith("ignore_time_segment_"):
            continue
        segment_list_per_file[name].append(segment)

    return segment_list_per_file


def _fill_corpus_with_segments(
    target_corpus: corpus.Corpus,
    audio_folder: str,
    segment_list_per_file: DefaultDict[str, List[corpus.Segment]],
):
    """
    :param target_corpus: in place filling of corpus
    :param audio_folder: output folder containing wavs from `SwitchboardSphereToWaveJob`
    :param segment_list_per_file: see `_get_segment_list_per_file()`
    :return:
    """
    for wav_file in sorted(glob.glob(os.path.join(audio_folder, "*.wav"))):
        recording = corpus.Recording()
        name = os.path.splitext(os.path.basename(wav_file))[0].lower()
        recording.name = name.lower()  # we are using lowercased names
        recording.audio = wav_file
        for segment in segment_list_per_file[name]:
            recording.add_segment(segment)
        target_corpus.add_recording(recording)


#### Evaluation Corpora Jobs ####


class CreateHub5e00CorpusJob(Job):
    """
    Creates the switchboard hub5e_00 corpus based on LDC2002S09
    No speaker information attached
    """

    def __init__(self, wav_audio_folder: tk.Path, hub5_transcription_folder: tk.Path):
        """
        :param wav_audio_folder: output of SwitchboardSphereToWave called on extracted LDC2002S09.tgz
        :param hub5_transcriptions: extracted LDC2002T43.tgz named "2000_hub5_eng_eval_tr"
        """
        self.wav_audio_folder = wav_audio_folder
        self.hub5_transcription_folder = hub5_transcription_folder

        self.out_bliss_corpus = self.output_path("hub5e_00.xml.gz")
        self.out_stm = self.output_path("hub5e_00.stm")
        self.out_glm = self.output_path("hub5e_00.glm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        base_dir = self.hub5_transcription_folder.get_path()
        glm_file = os.path.join(base_dir, "reference", "en20000405_hub5.glm")
        stm_file = os.path.join(base_dir, "reference", "hub5e00.english.000405.stm")
        assert os.path.isfile(glm_file)
        assert os.path.isfile(stm_file)

        _process_and_write_stm([stm_file], self.out_stm.get_path())

        hub5_corpus = corpus.Corpus()
        hub5_corpus.name = "hub5e_00"

        segment_list_per_file = _get_segment_list_per_file(self.out_stm.get_path())

        _fill_corpus_with_segments(
            hub5_corpus, self.wav_audio_folder.get_path(), segment_list_per_file
        )

        hub5_corpus.dump(self.out_bliss_corpus.get_path())
        shutil.copy(glm_file, self.out_glm.get_path())


class CreateHub5e01CorpusJob(Job):
    """
    Creates the switchboard hub5e_01 corpus based on LDC2002S13

    This corpus provides no glm, as the same as for Hub5e00 should be used

    No speaker information attached
    """

    def __init__(self, wav_audio_folder: tk.Path, hub5e01_folder: tk.Path):
        """
        :param wav_audio_folder: output of SwitchboardSphereToWave called on extracted LDC2002S13.tgz
        :param hub5e01_folder: extracted LDC2002S13 named "hub5e_01"
        """
        self.wav_audio_folder = wav_audio_folder
        self.hub5e_01_folder = hub5e01_folder

        self.out_bliss_corpus = self.output_path("hub5e_01.xml.gz")
        self.out_stm = self.output_path("hub5e_01.stm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        base_dir = self.hub5e_01_folder.get_path()
        stm_file = os.path.join(
            base_dir, "data", "transcr", "hub5e01.english.20010402.stm"
        )
        assert os.path.isfile(stm_file)

        _process_and_write_stm([stm_file], self.out_stm.get_path())

        hub5_corpus = corpus.Corpus()
        hub5_corpus.name = "hub5e_01"

        segment_list_per_file = _get_segment_list_per_file(self.out_stm.get_path())

        _fill_corpus_with_segments(
            hub5_corpus, self.wav_audio_folder.get_path(), segment_list_per_file
        )

        hub5_corpus.dump(self.out_bliss_corpus.get_path())


class CreateRT03sCTSCorpusJob(Job):
    """
    Create the RT03 test set corpus, specifically the "CTS" subset of LDC2007S10

    No speaker information attached
    """

    def __init__(self, wav_audio_folder: tk.Path, rt03_folder: tk.Path):
        """
        :param wav_audio_folder: output of SwitchboardSphereToWave called on extracted LDC2007S10.tgz
        :param rt03_folder: extracted LDC2007S10.tgz
        """
        self.wav_audio_folder = wav_audio_folder
        self.rt03_folder = rt03_folder

        self.out_bliss_corpus = self.output_path("rt03s_cts.xml.gz")
        self.out_stm = self.output_path("rt03s_cts.stm")
        self.out_glm = self.output_path("rt03s_cts.glm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        base_dir = self.rt03_folder.get_path()
        cts_path = os.path.join(
            base_dir, "data", "references", "eval03", "english", "cts"
        )
        glm_file = os.path.join(base_dir, "data", "trans_rules", "en20030506.glm")
        assert os.path.isdir(cts_path)
        assert os.path.isfile(glm_file)

        stm_files = sorted(glob.glob(os.path.join(cts_path, "*.stm")))
        _process_and_write_stm(stm_files, self.out_stm.get_path())

        rt03s_corpus = corpus.Corpus()
        rt03s_corpus.name = "rt03s_cts"

        segment_list_per_file = _get_segment_list_per_file(self.out_stm.get_path())

        _fill_corpus_with_segments(
            rt03s_corpus, self.wav_audio_folder.get_path(), segment_list_per_file
        )

        rt03s_corpus.dump(self.out_bliss_corpus.get_path())
        shutil.copy(glm_file, self.out_glm.get_path())


class CreateSwitchboardSpokenFormBlissCorpusJob(Job):
    """
    Creates a special E2E version of switchboard-1 used for e.g. BPE or Sentencepiece based models.
    It includes:
     - make sure everything is lowercased
     - conversion of numbers to written form
     - conversion of some short forms into spoken forms
     - making special tokens uppercase again
    """

    def __init__(self, switchboard_bliss_corpus: tk.Path):
        """
        :param switchboard_bliss_corpus: out_corpus of `CreateSwitchboardBlissCorpusJob`
        """
        self.switchboard_bliss_corpus = switchboard_bliss_corpus

        self.out_e2e_corpus = self.output_path("swb.e2e.corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        LocalPath = setup_path(__package__)
        map_source_path = LocalPath("switchboard_map_files/map_fsh_swb.txt.part1")
        map_target_path = LocalPath("switchboard_map_files/map_fsh_swb.txt.part2")

        replacement_map = {}

        with uopen(map_source_path) as map_source, uopen(map_target_path) as map_target:
            for source, target in zip(map_source, map_target):
                assert (
                    source is not None and target is not None
                ), "invalid switchboard map files found"
                replacement_map[source.strip()] = target.strip().replace("#", " ")

        special_token_map = {token: token.upper() for token in SPECIAL_TOKENS}

        # sort by longest first to avoid early matching
        map_regex = re.compile(
            "|".join(
                sorted(
                    map(re.escape, replacement_map.keys()),
                    key=lambda x: len(x),
                    reverse=True,
                )
            )
        )
        token_regex = re.compile("|".join(map(re.escape, special_token_map.keys())))

        c = corpus.Corpus()
        c.load(self.switchboard_bliss_corpus.get_path())

        for segment in c.segments():
            orth = segment.orth.lower()
            orth = map_regex.sub(lambda match: replacement_map[match.group(0)], orth)
            orth = token_regex.sub(
                lambda match: special_token_map[match.group(0)], orth
            )
            segment.orth = orth

        c.dump(self.out_e2e_corpus.get_path())


class CreateFisherTranscriptionsJob(Job):
    """
    Create the compressed text data based on the fisher transcriptions which can be used for LM training

    Part 1: https://catalog.ldc.upenn.edu/LDC2004T19
    Part 2: https://catalog.ldc.upenn.edu/LDC2005T19
    """

    def __init__(
        self,
        fisher_transcriptions1_folder: tk.Path,
        fisher_transcriptions2_folder: tk.Path,
    ):
        """
        :param fisher_transcriptions1_folder: path to unpacked LDC2004T19.tgz, usually named fe_03_p1_tran
        :param fisher_transcriptions2_folder: path to unpacked LDC2005T19.tgz, usually named fe_03_p2_tran
        """
        self.fsh_trans1_folder = fisher_transcriptions1_folder
        self.fsh_trans2_folder = fisher_transcriptions2_folder

        self.out = self.output_path("fisher.lm_train.txt.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        files1 = glob.glob(
            os.path.join(
                self.fsh_trans1_folder.get_path(), "data", "trans", "*", "fe_03_*.txt"
            )
        )
        files2 = glob.glob(
            os.path.join(
                self.fsh_trans1_folder.get_path(), "data", "trans", "*", "fe_03_*.txt"
            )
        )
        with uopen(self.out, "wt") as fout:
            for file in sorted(files1 + files2):
                with uopen(file) as fin:
                    for line in fin:
                        split = line.split(":")
                        if len(split) < 2:
                            continue
                        elif len(split) > 2:
                            assert False, "Weird line detected"
                        else:
                            fout.write(split[1].strip() + "\n")
