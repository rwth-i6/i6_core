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
        work_only = 2
        output_only = 3
        work_and_output = 4

    def __init__(self, job_list: List[Job], trigger: tk.Path, mode: CleanupMode):
        """
        :param job_list: Job(s) to delete
        :param trigger: trigger Path
        :param mode: what kind of deletion to perform
        """
        self.job_list = job_list
        self.trigger = trigger
        self.mode = mode
        self.out = self.output_path(os.path.basename(trigger.path))

        # check that the trigger does not appear in any of the Jobs that are deleted
        assert trigger.creator not in self.job_list

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import shutil

        for job in self.job_list:
            print("remove %s" % str(job))
            try:
                if self.mode is not self.CleanupMode.full_job:
                    if (
                        self.mode is self.CleanupMode.work_only
                        or self.mode is self.CleanupMode.work_and_output
                    ):
                        shutil.rmtree(os.path.join(job._sis_path(abspath=True), "work"))
                    if (
                        self.mode is self.CleanupMode.output_only
                        or self.mode is self.CleanupMode.work_and_output
                    ):
                        shutil.rmtree(
                            os.path.join(job._sis_path(abspath=True), "output")
                        )
                else:
                    shutil.rmtree(job._sis_path(abspath=True))
            except Exception as e:
                print("Deletion not possible with:")
                print(e)

        os.symlink(self.trigger.get_path(), self.out.get_path())
