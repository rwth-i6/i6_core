__all__ = ["AddImpulseResponseJob", "MixNoiseJob"]

import os
import audiomentations as am
import shutil
import logging
from scipy.io import wavfile


from sisyphus import *
from recipe.i6_core.lib import corpus

class AddImpulseResponseJob(Job):
    def __init__(self, corpus_file, new_corpus_name, ir_path, p):
        self.corpus_file = corpus_file
        self.new_corpus_name = new_corpus_name
        self.impulse_responses_dir = ir_path 
        self.p_impulse_response = p

        self.out_audio_folder = self.output_path("out_audio/", directory=True)
        self.out_corpus = self.output_path("ir_convoluted.corpus.xml.gz")
        self.out_segment_file = self.output_path("ir_convoluted.segments")

        self.rqmt = {"time": 12, "cpu": 2}

        self.add_impulse_response = am.AddImpulseResponse(ir_path=ir_path, p=p)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        self.id = os.path.basename(self.job_id())
        if not os.path.isdir(f"/dev/shm/{self.id}"):
            os.mkdir(f"/dev/shm/{self.id}")
        c = corpus.Corpus()
        nc = corpus.Corpus()
        segment_file_names = []

        c.load(tk.uncached_path(self.corpus_file))
        nc.name = self.new_corpus_name
        nc.speakers = c.speakers
        nc.default_speaker = c.default_speaker
        nc.speaker_name = c.speaker_name

        for r in c.recordings:
            perturbed_audio_name = 'perturbed_' + r.audio.split('/')[-1]
            samples, sample_rate = am.core.audio_loading_utils.load_wav_file(r.audio, sample_rate=None)
            augmented_samples = self.add_impulse_response(samples, sample_rate)
            wavfile.write(str(self.out_audio_folder) + '/' + perturbed_audio_name, sample_rate, augmented_samples)
            
            pr = corpus.Recording()
            pr.name = r.name
            pr.segments = r.segments
            pr.speaker_name = r.speaker_name
            pr.speakers = r.speakers
            pr.default_speaker = r.default_speaker
            pr.audio = str(self.out_audio_folder) + '/' + perturbed_audio_name
            nc.add_recording(pr)
            for s in pr.segments:
                segment_file_names.append(nc.name + '/' + pr.name + '/' + s.name + '\n')

        nc.dump(self.out_corpus.get_path())

        with open(tk.uncached_path(self.out_segment_file), 'w') as segments_outfile:
            segments_outfile.writelines(segment_file_names)

        shutil.rmtree(f"/dev/shm/{self.id}")


class MixNoiseJob(Job):
    def __init__(self, corpus_file, new_corpus_name, noise_files_dir, min_snr=3, max_snr=30, p=0.5):
        self.corpus_file = corpus_file
        self.noise_files_dir = noise_files_dir
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p
        self.new_corpus_name = new_corpus_name

        self.out_audio_folder = self.output_path("out_audio/", directory=True)
        self.out_corpus = self.output_path("noise_mixed.corpus.xml.gz")
        self.out_segment_file = self.output_path("noise_mixed.segments")

        self.rqmt = {"time": 12, "cpu": 2}

        self.mix_noise = am.AddBackgroundNoise(noise_files_dir, min_snr, max_snr, p)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        self.id = os.path.basename(self.job_id())
        if not os.path.isdir(f"/dev/shm/{self.id}"):
            os.mkdir(f"/dev/shm/{self.id}")
        c = corpus.Corpus()
        nc = corpus.Corpus()
        segment_file_names = []

        c.load(tk.uncached_path(self.corpus_file))
        nc.name = self.new_corpus_name
        nc.speakers = c.speakers
        nc.default_speaker = c.default_speaker
        nc.speaker_name = c.speaker_name

        for r in c.recordings:
            perturbed_audio_name = 'perturbed_' + r.audio.split('/')[-1]
            samples, sample_rate = am.core.audio_loading_utils.load_wav_file(r.audio, sample_rate=None)
            augmented_samples = self.mix_noise(samples, sample_rate)
            wavfile.write(str(self.out_audio_folder) + '/' + perturbed_audio_name, sample_rate, augmented_samples)

            pr = corpus.Recording()
            pr.name = r.name
            pr.segments = r.segments
            pr.speaker_name = r.speaker_name
            pr.speakers = r.speakers
            pr.default_speaker = r.default_speaker
            pr.audio = str(self.out_audio_folder) + '/' + perturbed_audio_name
            nc.add_recording(pr)
            for s in pr.segments:
                segment_file_names.append(nc.name + '/' + pr.name + '/' + s.name + '\n')

        nc.dump(self.out_corpus.get_path())

        with open(tk.uncached_path(self.out_segment_file), 'w') as segments_outfile:
            segments_outfile.writelines(segment_file_names)

        shutil.rmtree(f"/dev/shm/{self.id}")

