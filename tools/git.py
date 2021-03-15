__all__ = ["CloneGitRepository"]

import subprocess as sp

from sisyphus import *

Path = setup_path(__package__)


class CloneGitRepository(Job):
    __sis_hash_exclude__ = {"commit": None, "checkout_folder_name": "repository"}

    def __init__(
        self, url, branch=None, commit=None, checkout_folder_name="repository"
    ):
        self.url = url
        self.branch = branch
        self.commit = commit

        self.repository = self.output_path(checkout_folder_name, True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        args = ["git", "clone", self.url]
        if self.branch is not None:
            args.extend(["-b", self.branch])
        repository_dir = self.repository.get_path()
        args += [repository_dir]
        print(args)
        sp.run(args)
        if self.commit is not None:
            sp.run(["git", "checkout", self.commit], cwd=repository_dir)
