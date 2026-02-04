__all__ = ["GetColumnsFromCsvFileJob"]

import csv

from sisyphus import Job, Task, tk

from i6_core import util


class GetColumnsFromCsvFileJob(Job):
    """
    Dumps the values of a given set of columns from a csv file onto separate text files.
    The csv file must have been previously dumped with :funcref:`csv.writer` and :funcref:`csv.writerow`.

    The i-th output file contains the i-th column.
    """

    def __init__(self, csv_file: tk.Path, columns: list[int], delimiter: str = ","):
        """
        :param csv_file: Csv file for which to retrieve the corresponding column.
        :param column: Zero-based index for the column for which to retrieve the values.
        :param delimiter: Delimiter of the csv file.
        """
        self.csv_file = csv_file
        self.columns = columns
        self.delimiter = delimiter

        self.out_column_values = {i: self.output_path(f"out_col_{i}.txt.gz") for i in self.columns}

        self.rqmt = {"cpu": 1, "mem": 1.0, "time": 1.0}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        # No need to dump the outputs as a csv file, since each output just contains a single column.
        opened_outs = {i: util.uopen(self.out_column_values[i], "wt") for i in self.columns}
        with util.uopen(self.csv_file, "rt") as f_in:
            in_csv = csv.reader(f_in, delimiter=self.delimiter)
            for csv_line in in_csv:
                for column in self.columns:
                    # Encoding with unicode_escape allows special characters like "\n" to be printed as intended.
                    opened_outs[column].write(f"{csv_line[column].encode('unicode_escape').decode('utf-8')}\n")
