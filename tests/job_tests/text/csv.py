import filecmp
import tempfile

from sisyphus import setup_path

from i6_core.text.csv import GetColumnsFromCsvFileJob
from i6_core.util import uopen

Path = setup_path(__package__)


def test_get_columns_from_csv_file_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        from sisyphus import gs

        gs.WORK_DIR = tmpdir

        job = GetColumnsFromCsvFileJob(csv_file=Path("files/csv/input_file.csv"), columns=[0, 1])

        job._sis_setup_directory()
        job.run()

        # Column 0
        with uopen(Path("files/csv/out_col_0.txt").get_path(), "rt") as f:
            expected_out = f.readlines()
        with uopen(job.out_column_values[0].get_path(), "rt") as f:
            assert expected_out == f.readlines()

        # Column 1
        with uopen(Path("files/csv/out_col_1.txt").get_path(), "rt") as f:
            expected_out = f.readlines()
        with uopen(job.out_column_values[1].get_path(), "rt") as f:
            assert expected_out == f.readlines()
