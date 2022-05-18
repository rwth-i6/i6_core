__all__ = ["MakeJob"]

import logging
import subprocess as sp
import shutil
from tempfile import TemporaryDirectory
from typing import Iterator, Optional, List

from sisyphus import tk, gs, setup_path, Job, Task

Path = setup_path(__package__)


class MakeJob(Job):
    """
    Executes a sequence of make commands in a given folder
    """

    def __init__(
        self,
        folder: tk.Path,
        make_sequence: Optional[List[str]] = None,
        run_configure: bool = False,
        num_processes: int = 1,
        output_folder_name: str = "repository",
    ):
        """

        :param folder: folder in which the make commands are executed
        :param make_sequence: list of options that are given to the make calls.
            defaults to ["all"] i.e. "make all" is executed
        :param run_configure: runs ./configure before make
        :param num_processes: number of parallel running make processes
        :param output_folder_name: name of the output path folder
        """
        self.folder = folder
        self.make_sequence = make_sequence if make_sequence is not None else ["all"]
        self.run_configure = run_configure
        self.num_processes = num_processes

        self.rqmt = {"cpu": num_processes}

        self.out_repository = self.output_path(output_folder_name, True)

    def tasks(self) -> Iterator[Task]:
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        with TemporaryDirectory(prefix=gs.TMP_PREFIX) as temp_dir:
            shutil.rmtree(temp_dir)
            shutil.copytree(self.folder.get_path(), temp_dir, symlinks=True)

            if self.run_configure:
                sp.run(["./configure"], cwd=temp_dir, check=True)

            for command in self.make_sequence:
                args = ["make"]
                args.extend(command.split())
                if "-j" not in args:
                    args.extend(["-j", f"{self.num_processes}"])

                logging.info("running command: %s" % " ".join(args))
                sp.run(args, cwd=temp_dir, check=True)

            shutil.rmtree(self.out_repository.get_path())
            shutil.copytree(temp_dir, self.out_repository.get_path(), symlinks=True)
