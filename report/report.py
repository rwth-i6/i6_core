from sisyphus import *
from sisyphus import global_settings
import sys
import os
import getpass
import pprint
import gzip


class ReportFinishedOutput(Job):

  def __init__(self,
    name: str,
    output: tk.Path,
  ):
    self.name = name
    self.output = output
    self.out = self.output_path('report.gz')

    if hasattr(global_settings, "GLOBAL_SETTINGS_FILE"):
      with open(global_settings.GLOBAL_SETTINGS_FILE) as f:
        self.settings = f.read()
    else:
      self.settings = ''

    self.sis_command_line = sys.argv
    self.cwd = os.path.abspath('.')
    self.report_format = pprint.pformat
    self.mail_address = getattr(global_settings, 'MAIL_ADDRESS', None)


  def run(self):
    user = getpass.getuser()
    report = {
        "output": self.output.get_path(),
        "user": user,
        "name": self.name,
        "sis_command_line": self.sis_command_line,
        "cwd": self.cwd,
    }

    report = eval(repr(report))
    pprint.pprint(report)

    with gzip.open(str(self.out), 'w') as f:
      f.write(self.report_format(report).encode() + b'\n')

    if self.mail_address:
      self.sh('zcat {out} | mail -s \'Report finished: {name} \' {mail_address}')

  def tasks(self):
    yield Task('run', mini_task=True)
