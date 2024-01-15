import filecmp
import tempfile
from sisyphus import setup_path

from i6_core.text.processing import HeadJob, TailJob

Path = setup_path(__package__)


def test_head_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        from sisyphus import gs

        gs.WORK_DIR = tmpdir

        text_file = Path("files/input_file.txt")

        cases = [
            (None, 0.25, Path("files/out.head.1.txt"), False),
            (None, 0.5, Path("files/out.head.2.txt"), False),
            (None, 0.6, Path("files/out.head.2.txt"), False),
            (2, None, Path("files/out.head.2.txt"), False),
            (1, None, Path("files/out.head.1.txt"), False),
            (None, 0.5, Path("files/out.head.2.txt.gz"), True),
        ]

        for num_lines, ratio, reference_file, zip_output in cases:
            job = HeadJob(text_file=text_file, ratio=ratio, num_lines=num_lines, zip_output=zip_output)

            job._sis_setup_directory()
            job.run()

            assert filecmp.cmp(job.out.get_path(), reference_file.get_path(), shallow=False)


def test_tail_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        from sisyphus import gs

        gs.WORK_DIR = tmpdir

        text_file = Path("files/input_file.txt")

        cases = [
            (None, 0.5, Path("files/out.tail.2.txt"), False),
            (None, 0.75, Path("files/out.tail.3.txt"), False),
            (2, None, Path("files/out.tail.2.txt"), False),
            (2, None, Path("files/out.tail.2.txt.gz"), True),
        ]

        for num_lines, ratio, reference_file, zip_output in cases:
            job = TailJob(text_file=text_file, ratio=ratio, num_lines=num_lines, zip_output=zip_output)

            job._sis_setup_directory()
            job.run()

            assert filecmp.cmp(job.out.get_path(), reference_file.get_path(), shallow=False)
