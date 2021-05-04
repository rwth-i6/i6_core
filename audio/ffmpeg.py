__all__ = ["BlissFFMPEGJob"]

import copy
import os
import subprocess

from sisyphus import *

from recipe.i6_core.lib import corpus


class BlissFFMPEGJob(Job):
    """
    Applies an FFMPEG audio filter to all recordigns of a bliss corpus.
    This Job is extremely generic, as any valid audio option/filter string will work.

    WARNING:
        - This Job requires unique file names for the audio files
        - Changing the duration of the audio files when you have multiple segments per audio,
          as the segment information will be incorrect afterwards. Please use/write specific
          jobs in that case (e.g. see corpus.data_augmentation.ChangeCorpusSpeed).

    Typical applications:

    **Changing Audio Format/Encoding**

        - specify in `output_format` what container you want to use. If
          the filter string is empty (""), ffmepg will automatically use a default encoding option

        - specify specific encoding with `-c:a <codec>`. For a list of available codecs
          and their options see https://ffmpeg.org/ffmpeg-codecs.html#Audio-Encoders

        - specify a fixed bitrate with `-b:a <bit_rate>`, e.g. `64k`. Variable bitrate options depend on the
          used encoder, refer to the online documentation in this case

        - specify a sample rate with `-ar <sample_rate>`. FFMPEG will do proper resampling,
          so the speed of the audio is NOT changed.


    **Changing Channel Layout**

        - for detailed informations see https://trac.ffmpeg.org/wiki/AudioChannelManipulation

        - convert to mono `-ac 1`

        - selecting a specific audio channel:
          `-filter_complex \"[0:a]channelsplit=channel_layout=stereo:channels=FR[right]\" -map \"[right]\"`
          For a list of channels/layouts use `ffmpeg -layouts`


    **Simple Filter Syntax**

    For a list of available filters see: https://ffmpeg.org/ffmpeg-filters.html

    `-af \"<filter_name>=<first_param>=<first_param_value>:<second_param>=<second_param_value>\"`


    **Complex Filter Syntax**

    `-filter_complex \"[<input>]<simple_syntax>[<output>];[<input>]<simple_syntax>[<output>];...\"`

    Inputs and outputs can be namend arbitrarily, but the default stream 0 audio can be accessed with [0:a]

    The output stream that should be written into the audio is defined with `-map "[<output_stream>]"`

    IMPORTANT! Do not forget to escape your quotation marks correctly for `-af` or `-filter_complex`

    """

    def __init__(
        self,
        corpus_file,
        ffmpeg_option_string,
        ffmpeg_binary=None,
        recover_duration=False,
        output_format=None,
    ):
        """

        :param Path corpus_file: bliss corpus
        :param str|None ffmpeg_option_string: audio filter string, "-af" or "-filter_complex" with "-map" may be used
        :param Path|str|None ffmpeg_binary: path to a ffmpeg binary
        :param bool recover_duration: if the filter changes the duration of the audio, set to True
        :param str output_format: output file ending to determine container format (without dot)
        """
        self.corpus_file = corpus_file
        self.ffmpeg_option_string = ffmpeg_option_string
        self.audio_folder = self.output_path("audio/", directory=True)
        self.out = self.output_path("corpus.xml.gz")

        self.ffmpeg_binary = ffmpeg_binary if ffmpeg_binary else "ffmpeg"

        self.recover_duration = recover_duration
        self.output_format = output_format

        self.rqmt = {"time": 4, "cpu": 4, "mem": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
        if self.recover_duration:
            # recovering is not multi-threaded, so force cpu=1
            recover_rqmt = copy.copy(self.rqmt)
            recover_rqmt["cpu"] = 1
            yield Task("run_recover_duration", rqmt=recover_rqmt)

    def run(self):
        in_corpus = corpus.Corpus()
        out_corpus = corpus.Corpus()

        in_corpus.load(tk.uncached_path(self.corpus_file))
        out_corpus.name = in_corpus.name
        out_corpus.speakers = in_corpus.speakers
        out_corpus.default_speaker = in_corpus.default_speaker
        out_corpus.speaker_name = in_corpus.speaker_name

        # store index of last segment
        for r in in_corpus.recordings:
            nr = corpus.Recording()
            nr.name = r.name
            nr.segments = r.segments
            nr.speaker_name = r.speaker_name
            nr.speakers = r.speakers
            nr.default_speaker = r.default_speaker

            audio_name = r.audio.split("/")[-1]

            if self.output_format is not None:
                name, ext = os.path.splitext(audio_name)
                audio_name = name + "." + self.output_format

            nr.audio = os.path.join(tk.uncached_path(self.audio_folder), audio_name)
            out_corpus.add_recording(nr)

        from multiprocessing import pool

        p = pool.Pool(self.rqmt["cpu"])
        p.map(self._perform_ffmpeg, in_corpus.recordings)

        if self.recover_duration:
            out_corpus.dump(tk.uncached_path("temp_corpus.xml.gz"))
        else:
            out_corpus.dump(tk.uncached_path(self.out))

    def run_recover_duration(self):
        """
        Open all files with "soundfile" and extract the length information

        :return:
        """
        import soundfile

        c = corpus.Corpus()
        c.load("temp_corpus.xml.gz")

        for r in c.all_recordings():
            assert len(r.segments) == 1, "needs to be a single segment recording"
            old_duration = r.segments[0].end
            data, sample_rate = soundfile.read(open(r.audio, "rb"))
            new_duration = len(data) / sample_rate
            print(
                "%s: adjusted from %f to %f seconds"
                % (r.segments[0].name, old_duration, new_duration)
            )
            r.segments[0].end = new_duration

        c.dump(tk.uncached_path(self.out))

    def _perform_ffmpeg(self, recording):
        """
        Build and call an FFMPEG command to apply on a recording

        :param corpus.Recording recording:
        :return:
        """
        audio_name = recording.audio.split("/")[-1]

        if self.output_format is not None:
            name, ext = os.path.splitext(audio_name)
            audio_name = name + "." + self.output_format

        target = tk.uncached_path(self.audio_folder) + "/" + audio_name
        if not os.path.exists(target):
            print("try converting %s" % target)
            command_head = [
                self.ffmpeg_binary,
                "-hide_banner",
                "-y",
                "-i",
                recording.audio,
            ]
            command_tail = ["%s/%s" % (self.audio_folder, audio_name)]
            if self.ffmpeg_option_string is None or len(self.ffmpeg_option_string) == 0:
                command = command_head + command_tail
            else:
                command = (
                    command_head + self.ffmpeg_option_string.split(" ") + command_tail
                )
            subprocess.check_call(command)
        else:
            print("skipped existing %s" % target)
