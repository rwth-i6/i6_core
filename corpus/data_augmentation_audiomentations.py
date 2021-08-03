__all__ = ["AddImpulseResponseJob", "MixNoiseJob"]

import os
import audiomentations
import shutil
from scipy.io import wavfile

from sisyphus import *
from recipe.i6_core.lib import corpus


class AddImpulseResponseJob(Job):
    """
    Add impulse responses to each recording in the corpus with a given probability to be affected.
    This is done by convolving the audio with an impulse response to simulate other acoustic conditions.
    Supports .wav files
    """

    def __init__(
        self, corpus_file, new_corpus_name, ir_path, augment_prob, time_rqmt=12
    ):
        """

        :param Path corpus_file: Bliss corpus with wav files
        :param str new_corpus_name: name of the new corpus
        :param ir_path: path to the directory containing impulse responses
        :param augment_prob: probability of each audio to be augmented by impulse response convolution
        :param time_rqmt: time requirement for the Sisypus job
        """
        self.corpus_file = corpus_file
        self.new_corpus_name = new_corpus_name
        self.impulse_responses_dir = ir_path

        self.out_audio_folder = self.output_path("out_audio/", directory=True)
        self.out_corpus = self.output_path("ir_convoluted.corpus.xml.gz")

        self.rqmt = {"time": time_rqmt, "cpu": 2}

        self.add_impulse_response = audiomentations.AddImpulseResponse(
            ir_path=ir_path, p=augment_prob
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        job_id = os.path.basename(self.job_id())
        orig_corpus = corpus.Corpus()
        new_corpus = corpus.Corpus()

        orig_corpus.load(tk.uncached_path(self.corpus_file))
        new_corpus.name = self.new_corpus_name
        new_corpus.speakers = list(orig_corpus.all_speakers())
        new_corpus.default_speaker = orig_corpus.default_speaker
        new_corpus.speaker_name = orig_corpus.speaker_name

        for r in orig_corpus.all_recordings():
            perturbed_audio_name = "perturbed_" + r.audio.split("/")[-1]
            (
                samples,
                sample_rate,
            ) = audiomentations.core.audio_loading_utils.load_wav_file(
                r.audio, sample_rate=None
            )
            augmented_samples = self.add_impulse_response(samples, sample_rate)
            wavfile.write(
                str(self.out_audio_folder) + "/" + perturbed_audio_name,
                sample_rate,
                augmented_samples,
            )

            pr = corpus.Recording()
            pr.name = r.name
            pr.segments = r.segments
            pr.speaker_name = r.speaker_name
            pr.speakers = r.speakers
            pr.default_speaker = r.default_speaker
            pr.audio = str(self.out_audio_folder) + "/" + perturbed_audio_name
            new_corpus.add_recording(pr)

        new_corpus.dump(self.out_corpus.get_path())


class MixNoiseJob(Job):
    """
    Add noise to each recording in the corpus with a given probability to be affected.
    Supports .wav files.
    """

    def __init__(
        self,
        corpus_file,
        new_corpus_name,
        noise_files_dir,
        min_snr=3,
        max_snr=30,
        augment_prob=1,
        time_rqmt=12,
    ):
        """

        :param Path corpus_file: Bliss corpus with wav files
        :param str new_corpus_name: name of the new corpus
        :param noise_files_dir: path to the directory containing noise files
        :param min_snr: minimum signal-to-noise ratio to be used
        :param max_snr: maximum signal-to-noise ratio to be used
        :param augment_prob: probability of each audio to be augmented by noise mixing
        :param time_rqmt: time requirement for the Sisypus job
        """
        self.corpus_file = corpus_file
        self.noise_files_dir = noise_files_dir
        self.new_corpus_name = new_corpus_name

        self.out_audio_folder = self.output_path("out_audio/", directory=True)
        self.out_corpus = self.output_path("noise_mixed.corpus.xml.gz")
        self.out_segment_file = self.output_path("noise_mixed.segments")

        self.rqmt = {"time": time_rqmt, "cpu": 2}

        self.mix_noise = audiomentations.AddBackgroundNoise(
            noise_files_dir, min_snr, max_snr, p=augment_prob
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        self.id = os.path.basename(self.job_id())
        c = corpus.Corpus()
        nc = corpus.Corpus()

        c.load(tk.uncached_path(self.corpus_file))
        nc.name = self.new_corpus_name
        nc.speakers = c.speakers
        nc.default_speaker = c.default_speaker
        nc.speaker_name = c.speaker_name

        for r in c.recordings:
            perturbed_audio_name = "perturbed_" + r.audio.split("/")[-1]
            (
                samples,
                sample_rate,
            ) = audiomentations.core.audio_loading_utils.load_wav_file(
                r.audio, sample_rate=None
            )
            augmented_samples = self.mix_noise(samples, sample_rate)
            wavfile.write(
                str(self.out_audio_folder) + "/" + perturbed_audio_name,
                sample_rate,
                augmented_samples,
            )

            pr = corpus.Recording()
            pr.name = r.name
            pr.segments = r.segments
            pr.speaker_name = r.speaker_name
            pr.speakers = r.speakers
            pr.default_speaker = r.default_speaker
            pr.audio = str(self.out_audio_folder) + "/" + perturbed_audio_name
            nc.add_recording(pr)

        nc.dump(self.out_corpus.get_path())
