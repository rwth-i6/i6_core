import os
import tempfile
from typing import Optional

from sisyphus import tk, setup_path

from i6_core.returnn.hdf import ReturnnDumpHDFJob

rel_path = setup_path(__package__)


def test_hdf_dump():
    with tempfile.TemporaryDirectory() as tmpdir:
        from sisyphus import gs

        gs.WORK_DIR = tmpdir
        with open(f"{tmpdir}/tmp.config", 'wt') as f:
            f.write("#!rnn.py\n")
            f.write("train = {'class': 'DummyDataset', 'input_dim': 3, 'output_dim': 4, 'num_seqs': 2}")
        data = rel_path(f"{tmpdir}/tmp.config")
        job = ReturnnDumpHDFJob(data=data)
        for task in ReturnnDumpHDFJob.tasks():
            task()
        
