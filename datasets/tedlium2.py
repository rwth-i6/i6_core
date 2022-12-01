__all__ = [
    "DownloadTEDLIUM2CorpusJob",
    "CreateTEDLIUM2BlissCorpusJob"
]

import os
import re
import shutil
import subprocess

from sisyphus import *

from i6_core.lib import corpus
from i6_core.util import uopen

class DownloadTEDLIUM2CorpusJob(Job):
    """
    Download full TED-LIUM Release 2 corpus from 
    https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/
    (all train/dev/test/LM/dictionary data included)
    """

    def __init__(self):
        self.out_corpus_folders = dict(
            (corpus_key, self.output_path(corpus_key)) for corpus_key in ["train", "dev", "test"])
        self.out_lm_folder = self.output_path("LM")
        self.out_vocab_dict = self.output_path("TEDLIUM.152k.dic")

    def tasks(self):
        yield Task("run", mini_task=True)
        yield Task("process_dict", mini_task=True)

    def run(self):
        subprocess.check_call(
            ["wget", "--no-check-certificate", "https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release2.tar.gz"]
        )
        subprocess.check_call(
            ["tar", "-zxvf", "TEDLIUM_release2.tar.gz"]
        )
        for corpus_key in ["train", "dev", "test"]:
            shutil.move("TEDLIUM_release2/%s" %corpus_key, self.out_corpus_folders[corpus_key].get_path())
        shutil.move("TEDLIUM_release2/LM", self.out_lm_folder.get_path())

    def process_dict(self):
        dict_file = "TEDLIUM_release2/TEDLIUM.152k.dic"
        with uopen(dict_file, 'r') as f:
            data = f.read()
            for n in range(2, 8):
                data = data.replace("(%d)" %n, "")
            data = data.replace("ayışığı EY\n", "")
            # 2 minor pronunciation fixes
            data = data.replace("'d   D IY", "'d   D")
            data = data.replace("'ll   EH L EH L", "'ll   L")

        dict_file = 'mod.TEDLIUM.152k.dic'
        with open(dict_file, 'w') as f:
            f.write(data)
        shutil.move(dict_file, self.out_vocab_dict.get_path())


class CreateTEDLIUM2BlissCorpusJob(Job):
    """
    Processes stm files from TEDLIUM2 corpus folders and creates Bliss corpus files 
    Outputs a stm file and a bliss .xml.gz file for each train/dev/test set
    """
    
    def __init__(self, corpus_folders):
        """
        :param Dict {corpus_key: Path} corpus_folders: 
        """
        self.corpus_folders = corpus_folders

        self.out_corpus_files = {}
        self.out_stm_files = {}
        for corpus_key in ["train", "dev", "test"]:
            assert corpus_key in self.corpus_folders
            self.out_corpus_files[corpus_key] = self.output_path("%s.corpus.xml.gz" %corpus_key)
            self.out_stm_files[corpus_key] = self.output_path("%s.stm" %corpus_key)

    def tasks(self):
        yield Task("make_stm", mini_task=True)
        yield Task("make_corpus", mini_task=True)

    def make_stm(self):
        def extend_segment_time(seg, preSeg, nextSeg, startPad=0.15, endPad=0.1):
            start = float(seg[3])
            end = float(seg[4])
            # start padding (previous seg alread padded)
            if preSeg is not None and seg[0] == preSeg[0]:
                preEnd = float(preSeg[4])
            else:
                preEnd = 0.0
            start = max(start - startPad, preEnd)
            # end padding (next seg not yet padded and start padding is more important)
            nextStart = end + endPad
            if nextSeg is not None and seg[0] == nextSeg[0]:
                nextStart = max(float(nextSeg[3]) - startPad, end)
            end = min(end + endPad, nextStart)
            return "%.2f" %(start), "%.2f" %(end)

        header = [ 
            ';;', 
            ';; LABEL "o" "Overall" "Overall results"',
            ';; LABEL "f0" "f0" "Wideband channel"',
            ';; LABEL "f2" "f2" "Telephone channel"',
            ';; LABEL "male" "Male" "Male Talkers"',
            ';; LABEL "female" "Female" "Female Talkers"',
            ';;'
        ]
        for corpus_key in ["train", "dev", "test"]:
            f = open("%s.stm" %corpus_key, "w")
            f.write("\n".join(header) + "\n")

            stm_folder = os.path.join(self.corpus_folders[corpus_key].get_path(), "stm")
            for stm_file in sorted(os.listdir(stm_folder)):
                if not stm_file.endswith('.stm'):
                    continue
                data = self.load_stm_data(os.path.join(stm_folder, stm_file))
                for idx in range(len(data)):
                    # some text normalization
                    text = data[idx][6]
                    text = text.replace('imiss','i miss')
                    text = text.replace('uptheir','up their')
                    text = re.sub("(\w)'([a-zA-Z])", r"\1 '\2", text) # split apostrophe 
                    data[idx][6] = text
                    
                    # train-only: segment boundary non-overlapping extension (kaldi)
                    if corpus_key == "train":
                        pre_seg = None if idx == 0 else data[idx - 1]
                        next_seg = None if idx == len(data) -1 else data[idx + 1]
                        data[idx][3], data[idx][4] = extend_segment_time(data[idx], pre_seg, next_seg, 0.15, 0.1)                
                
                for seg in data:
                    f.write(' '.join(seg) + '\n')

            f.close()
            shutil.move("%s.stm" %corpus_key, self.out_stm_files[corpus_key].get_path())

    def make_corpus(self):
        """
        create bliss corpus from stm file (always include speakers)
        """
        for corpus_key in ["train", "dev", "test"]:
            audio_dir = os.path.join(self.corpus_folders[corpus_key].get_path(), "sph")
            stm_file = self.out_stm_files[corpus_key].get_path()
            data = self.load_stm_data(stm_file)

            c = corpus.Corpus()
            c.name = "TED-LIUM-realease2"

            speakers = []
            last_rec_name = ""
            recording = None 
            for seg in data:
                rec_name, channel, spk_name, start, end, gender, text = seg

                if not spk_name in speakers:
                    speakers.append(spk_name)
                    speaker = corpus.Speaker()
                    speaker.name = spk_name
                    if 'female' in gender:
                        speaker.attribs["gender"] = 'female'
                    elif 'male' in gender:
                        speaker.attribs["gender"] = 'male'
                    c.add_speaker(speaker)

                if rec_name != last_rec_name:
                    if recording:
                        c.recordings.append(recording)
                    recording = corpus.Recording()
                    recording.name = rec_name
                    recording.audio = os.path.join(audio_dir, "%s.sph" %rec_name)
                    last_rec_name = rec_name
                    seg_id = 1
                
                segment = corpus.Segment()
                segment.name = str(seg_id)
                segment.start = float(start)
                segment.end = float(end)
                segment.speaker_name = spk_name
                segment.orth = text

                recording.segments.append(segment)
                seg_id += 1

            if recording:
                c.recordings.append(recording)
            c.dump(self.out_corpus_files[corpus_key].get_path())

    def load_stm_data(self, stm_file):
        """
        :param str stm_file
        """
        data = []
        with uopen(stm_file, 'r') as f:
            for line in f:
                if not line.strip() or line.strip().startswith(';;'): 
                    continue
                if "ignore_time_segment_in_scoring" in line: 
                    continue
                line = line.replace('<F0_M>', '<o,f0,male>')
                line = line.replace('<F0_F>', '<o,f0,female>')

                tokens = line.split()
                assert len(tokens) >= 7, line
                recName, channel, spkName, start, end, gender = tokens[:6]
                text = ' '.join(tokens[6:])
                data.append([recName, channel, spkName, start, end, gender, text])
        return data


