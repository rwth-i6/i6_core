from sisyphus import *

import getpass
import gzip
from typing import Dict, Union, Callable, Optional
import pprint

_Report_Type = Dict[str, Union[tk.AbstractPath, str]]


class GenerateReportStringJob(Job):
    """
    Job to generate and output a report string
    """

    def __init__(
        self,
        report_values: _Report_Type,
        report_template: Optional[Callable[[_Report_Type], str]] = None,
        compress: bool = True,
    ):
        """

        :param report_values:
        :param report_template:
        :param compress:
        """
        self.report_values = report_values
        self.report_template = report_template
        self.compress = compress

        self.out_report = self.output_path("report.txt.gz" if self.compress else "report.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        if self.report_template:
            report = self.report_template(self.report_values)
        elif callable(self.report_values):
            report = str(self.report_values())
        else:
            report = pprint.pformat(self.report_values, width=140)

        if self.compress:
            with gzip.open(self.out_report.get_path(), "wt") as f:
                f.write(report)
        else:
            with open(self.out_report.get_path(), "wt") as f:
                f.write(report)


class MailJob(Job):
    """
    Job that sends a mail upon completion of an output
    """

    def __init__(
        self,
        result: tk.AbstractPath,
        subject: Optional[str] = None,
        mail_address: str = getpass.getuser(),
        send_contents: bool = False,
    ):
        """

        :param result: graph output that triggers sending the mail
        :param subject: Subject of the mail
        :param mail_address: Mail address of recipient (default: user)
        :param send_contents: send the contents of result in body of the mail
        """
        self.result = result
        self.subject = subject
        self.mail_address = mail_address
        self.send_contents = send_contents

        self.out_status = self.output_var("out_status")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        if self.subject is None:
            subject = f"Output {str(self.result)} is finished"
        else:
            subject = self.subject

        if self.send_contents:
            value = self.sh(
                f"zcat -f {self.result.get_path()} | mail -s '{subject}' {self.mail_address}"
            )
        else:
            value = self.sh(
                f"echo '{subject}' | mail -s '{subject}' {self.mail_address}"
            )

        self.out_status.set(value)
