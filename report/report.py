__all__ = ["GenerateReportStringJob", "MailJob"]
from sisyphus import *

import getpass
import gzip
from typing import Dict, Union, Callable, Optional
import pprint
import subprocess

from i6_core.util import uopen

_Report_Type = Dict[str, Union[tk.AbstractPath, str]]


class GenerateReportStringJob(Job):
    """
    Job to generate and output a report string
    """

    def __init__(
        self,
        report_values: Union[_Report_Type, Callable],
        report_template: Optional[Union[Callable[[_Report_Type], str], str]] = None,
        compress: bool = True,
    ):
        """

        :param report_values: Can be either directly callable or a dict which then is handled by report_template
        :param report_template: Function to handle report_values of type _Report_Type
        :param compress: Whether to zip the report
        """
        if report_template is not None:
            assert not isinstance(report_values, Callable)
        self.report_values = report_values
        self.report_template = report_template
        self.compress = compress

        self.out_report = self.output_path(
            "report.txt.gz" if self.compress else "report.txt"
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        if self.report_template:
            if isinstance(self.report_template, str):
                report = self.report_template.format(**self.report_values)
            else:
                report = self.report_template(self.report_values)
        elif callable(self.report_values):
            report = str(self.report_values())
        else:
            report = pprint.pformat(self.report_values, width=140)

        with uopen(self.out_report.get_path(), "wt") as f:
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
            p1 = subprocess.Popen(
                ["zcat", "-f", self.result.get_path()], stdout=subprocess.PIPE
            )
            value = subprocess.check_output(
                ["mail", "-s", subject, self.mail_address], stdin=p1.stdout
            )
        else:
            out = subprocess.run(
                ["mail", "-s", subject, self.mail_address], input=subject, check=True
            )
            value = out.returncode
        self.out_status.set(value)
