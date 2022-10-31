"""
Jobs regarding Sisyphus graph management
"""
import enum
import os
from typing import List

from sisyphus import Job, Task, tk


class MultiJobCleanupJob(Job):
    """
    This Job will remove all given job folders when the provided Path is available.

    Note that the provided jobs should not have any directly registered output, as otherwise they
    will re-run right away.
    """

    class CleanupMode(enum.Enum):
        full_job = 1
        work_folder_only = 2
        output_folder_only = 3
        work_and_output_folder = 4
        output_file_only = 5

    def __init__(
        self, job_output_list: List[tk.Path], trigger: tk.Path, mode: CleanupMode
    ):
        """
        :param job_output_list: Job outputs to delete (one output is sufficient to clean the whole job
            unless you use the `output_file_only` mode)
        :param trigger: trigger Path, this should be the path that causes the MultiCleanUpJob to run
            All the outputs/jobs to be deleted should be dependencies.
        :param mode: what kind of deletion to perform
        """
        self.job_output_list = job_output_list
        self.trigger = trigger
        self.mode = mode
        self.out = self.output_path(os.path.basename(trigger.path))

        # check that the trigger does not share a common job with the job outputs to delete
        for job_output in job_output_list:
            assert trigger.creator is not job_output.creator

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import shutil

        for job_output in self.job_output_list:
            print("remove %s" % str(job_output))
            try:
                if self.mode is self.CleanupMode.output_file_only:
                    shutil.rmtree(job_output.get_path())
                elif self.mode is self.CleanupMode.full_job:
                    shutil.rmtree(job_output.creator._sis_path(abspath=True))
                else:
                    if (
                        self.mode is self.CleanupMode.work_folder_only
                        or self.mode is self.CleanupMode.work_and_output_folder
                    ):
                        shutil.rmtree(
                            os.path.join(
                                job_output.creator._sis_path(abspath=True), "work"
                            )
                        )
                    if (
                        self.mode is self.CleanupMode.output_folder_only
                        or self.mode is self.CleanupMode.work_and_output_folder
                    ):
                        shutil.rmtree(
                            os.path.join(
                                job_output.creator._sis_path(abspath=True), "output"
                            )
                        )
            except Exception as e:
                print("Deletion not possible with:")
                print(e)

        os.symlink(self.trigger.get_path(), self.out.get_path())
