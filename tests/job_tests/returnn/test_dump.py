import os
import tempfile

from sisyphus import tk, setup_path

from i6_core.returnn.hdf import ReturnnDumpHDFJob

rel_path = setup_path(__package__)


def test_hdf_dump():
    with tempfile.TemporaryDirectory() as tmpdir:
        from sisyphus import gs

        gs.WORK_DIR = tmpdir

        # Case 1: tk.Path
        with open(f"{tmpdir}/tmp.config", "wt") as f:
            f.write("#!rnn.py\n")
            f.write("train = {'class': 'DummyDataset', 'input_dim': 3, 'output_dim': 4, 'num_seqs': 2}")
        data = rel_path(f"{tmpdir}/tmp.config")
        job = ReturnnDumpHDFJob(data=data, returnn_root=tk.Path("returnn/"))
        assert [task.name() for task in job.tasks()] == ["run"]
        job._sis_setup_directory()
        job.run()

        # Case 2: dict
        data2 = {"class": "DummyDataset", "input_dim": 3, "output_dim": 4, "num_seqs": 2}
        job2 = ReturnnDumpHDFJob(data=data2, returnn_root=tk.Path("returnn/"))
        assert [task.name() for task in job2.tasks()] == ["write_config", "run"]
        job2._sis_setup_directory()
        job2.write_config()
        job2.run()

        # Case 3: str
        data3 = "{'class': 'DummyDataset', 'input_dim': 3, 'output_dim': 4, 'num_seqs': 2}"
        job3 = ReturnnDumpHDFJob(data=data3, returnn_root=tk.Path("returnn/"))
        assert [task.name() for task in job3.tasks()] == ["write_config", "run"]
        job3._sis_setup_directory()
        job3.write_config()
        job3.run()

        assert os.path.getsize(job.out_hdf) == os.path.getsize(job2.out_hdf) == os.path.getsize(job3.out_hdf), (
            os.path.getsize(job.out_hdf),
            os.path.getsize(job2.out_hdf),
            os.path.getsize(job3.out_hdf),
        )
